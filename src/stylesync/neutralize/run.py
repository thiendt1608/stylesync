"""Phase 2: back-translate each burst into neutral text with the OpenAI Batch API, then build validated pairs.

  submit [--pilot]   send every burst that has no cached result yet (pilot: the review sample only)
  collect            wait for submitted batches to end and cache their results
  build [--pilot]    validate cached results -> pairs.jsonl, rejects.jsonl, stats.json (real cost from usage)
  run [--pilot]      submit + collect + build
  review --n 100     write a random sample of pairs to data/neutral/review.md for the spot-check
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from stylesync.clean.pipeline import read_jsonl, sample, write_jsonl
from stylesync.neutralize.prompt import VARIANTS, cache_key, request
from stylesync.neutralize.validate import check

USAGE = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
DONE = {"completed", "failed", "expired", "cancelled"}


def scope(cfg: dict, base: Path, pilot: bool) -> list[dict]:
    if pilot:
        return sample(base / cfg["in"], cfg["pilot"]["n"], cfg["pilot"]["seed"])
    return read_jsonl(base / cfg["in"])


def key_of(cfg: dict, row: dict) -> str:
    return cache_key(cfg["model"], cfg["prompt_version"], row["text"])


def read_cache(cfg: dict, base: Path) -> dict[str, dict]:
    path = base / cfg["out"]["cache"]
    return {c["key"]: c for c in read_jsonl(path)} if path.exists() else {}


def batch_state(cfg: dict, base: Path) -> dict[str, dict]:
    """batches.jsonl is append-only: a submit line, later a done line with the same id."""
    path = base / cfg["out"]["batches"]
    state: dict[str, dict] = {}
    for line in read_jsonl(path) if path.exists() else []:
        state.setdefault(line["id"], {}).update(line)
    return state


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def submit(client, cfg: dict, base: Path, rows: list[dict]) -> str | None:
    """One request per unique text that is neither cached nor waiting in an open batch."""
    return submit_bodies(client, cfg, base, {key_of(cfg, r): request(cfg, r["text"]) for r in rows})


def submit_bodies(client, cfg: dict, base: Path, bodies: dict[str, dict]) -> str | None:
    """Send the Responses API bodies whose key is neither cached nor waiting in an open batch."""
    cache = read_cache(cfg, base)
    pending = {k for b in batch_state(cfg, base).values() if not b.get("done") for k in b["keys"]}
    todo = {k: body for k, body in bodies.items() if k not in cache and k not in pending}
    if not todo:
        return None
    # The upload copy is local only for the moment it takes to send it.
    path = (base / cfg["out"]["batches"]).with_name("upload.jsonl")
    write_jsonl(path, [{"custom_id": k, "method": "POST", "url": "/v1/responses", "body": body}
                       for k, body in todo.items()])
    with path.open("rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    path.unlink()
    batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/responses", completion_window="24h")
    append_jsonl(base / cfg["out"]["batches"], [{"id": batch.id, "keys": list(todo),
                                                  "submitted": datetime.now(timezone.utc).isoformat()}])
    return batch.id


def variants(d: dict) -> dict | None:
    """Decoded neutralize output: both variants as strings, else None."""
    return {v: d[v].strip() for v in VARIANTS} if all(isinstance(d.get(v), str) for v in VARIANTS) else None


def parse(line: dict, decode=variants) -> dict | None:
    """Cache entry for one output-file line. Failed requests stay uncached so the next submit retries them.

    `decode` turns the JSON reply into the cached `out` (None when it does not fit the expected shape)."""
    resp = line.get("response")
    if line.get("error") or not resp or resp.get("status_code") != 200:
        return None
    body = resp["body"]
    parts = [c for o in body.get("output") or [] if o.get("type") == "message" for c in o.get("content") or []]
    if any(c.get("type") == "refusal" for c in parts):
        stop = "refusal"
    elif body.get("status") == "completed":
        stop = "completed"
    else:  # incomplete: max_output_tokens or content_filter
        stop = (body.get("incomplete_details") or {}).get("reason") or body.get("status")
    out = None
    if stop == "completed":
        try:
            d = json.loads("".join(c["text"] for c in parts if c.get("type") == "output_text"))
            out = decode(d) if isinstance(d, dict) else None
        except ValueError:
            pass
    u = body.get("usage") or {}
    usage = {"input_tokens": u.get("input_tokens") or 0,
             "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens") or 0,
             "output_tokens": u.get("output_tokens") or 0,
             "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0}
    return {"key": line["custom_id"], "stop_reason": stop, "out": out, "usage": usage}


def file_lines(client, file_id: str | None) -> list[dict]:
    if not file_id:
        return []
    return [json.loads(x) for x in client.files.content(file_id).text.split("\n") if x.strip()]


def retrieve(client, bid: str, poll_s: float):
    """Batch status; a dropped connection (laptop asleep, Wi-Fi) is retried instead of ending the wait."""
    import openai

    while True:
        try:
            return client.batches.retrieve(bid)
        except openai.APIConnectionError as e:
            print(f"{bid}: {e}, retrying", flush=True)
            time.sleep(poll_s)


def collect(client, cfg: dict, base: Path, poll_s: float | None = None, decode=variants, wait: bool = True) -> Counter:
    """Wait for every open batch, cache its results, delete its files on OpenAI, mark it done.
    With wait=False, only collect the batches that have already ended.

    Safe to rerun after a crash: a batch is only marked done after its results are cached.
    """
    poll_s = cfg["poll_s"] if poll_s is None else poll_s
    total: Counter = Counter()
    for bid, b in batch_state(cfg, base).items():
        if b.get("done"):
            continue
        while (info := retrieve(client, bid, poll_s)).status not in DONE:
            if not wait:
                break
            rc = info.request_counts
            print(f"{bid}: {info.status}, {rc.completed if rc else 0}/{rc.total if rc else '?'} done", flush=True)
            time.sleep(poll_s)
        if info.status not in DONE:
            continue
        counts: Counter = Counter()
        entries = []
        for line in file_lines(client, info.output_file_id) + file_lines(client, info.error_file_id):
            e = parse(line, decode)
            counts["succeeded" if e else (line.get("error") or {}).get("code") or "errored"] += 1
            if e is not None:
                entries.append(e | {"model": cfg["model"], "prompt_version": cfg["prompt_version"]})
        append_jsonl(base / cfg["out"]["cache"], entries)
        for fid in (info.input_file_id, info.output_file_id, info.error_file_id):
            if fid:
                client.files.delete(fid)
        append_jsonl(base / cfg["out"]["batches"], [{"id": bid, "done": True, "status": info.status,
                                                      "results": dict(counts)}])
        total += counts
    return total


def cost(entries: list[dict], price: dict) -> float:
    """Reasoning tokens are already inside output_tokens; cached tokens inside input_tokens."""
    return sum((e["usage"]["input_tokens"] - e["usage"]["cached_tokens"]) * price["input"]
               + e["usage"]["cached_tokens"] * price["cached_input"]
               + e["usage"]["output_tokens"] * price["output"] for e in entries) / 1e6


def build(cfg: dict, base: Path, rows: list[dict], n_all: int) -> dict:
    """Validated pairs for the rows in scope. n_all is the full row count, for the cost projection."""
    cache = read_cache(cfg, base)
    lo, hi = cfg["validate"]["len_ratio"]
    pairs, rejects, used = [], [], {}
    reasons: Counter = Counter()
    stops: Counter = Counter()
    for r in rows:
        e = cache.get(key_of(cfg, r))
        if e is None:
            continue
        used[e["key"]] = e
        stops[e["stop_reason"]] += 1
        if e["out"] is None:
            rejects.append({"id": r["id"], "variant": None, "reasons": [e["stop_reason"]], "original": r["text"]})
            reasons[e["stop_reason"]] += 1
            continue
        for v in VARIANTS:
            why = check(r["text"], e["out"][v], (lo, hi))
            if why:
                rejects.append({"id": r["id"], "variant": v, "reasons": why, "neutral": e["out"][v],
                                "original": r["text"]})
                reasons.update(why)
                continue
            pairs.append({"id": f"{r['id']}-{v[0]}", "task": "message", "source": r["source"],
                          "thread_id": r["thread_id"], "ts": r["ts"], "weight": r["weight"], "variant": v,
                          "input": e["out"][v], "output": r["text"]})
    write_jsonl(base / cfg["out"]["pairs"], pairs)
    write_jsonl(base / cfg["out"]["rejects"], rejects)
    entries = list(used.values())
    spent = cost(entries, cfg["price"])
    stats = {
        "model": cfg["model"], "prompt_version": cfg["prompt_version"],
        "rows": len(rows), "cached": len(entries), "missing": len(rows) - len(entries),
        "pairs": dict(Counter(p["variant"] for p in pairs)), "rejected": len(rejects),
        "reject_reasons": dict(reasons), "stop_reasons": dict(stops),
        "tokens": {u: sum(e["usage"][u] for e in entries) for u in USAGE},
        "cost_usd": round(spent, 4),
        "cost_usd_projected_all_rows": round(spent / max(len(entries), 1) * n_all, 2),
        "cost_usd_all_cache": round(cost(list(cache.values()), cfg["price"]), 4),
    }
    (base / cfg["out"]["stats"]).write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def review(cfg: dict, base: Path, n: int, seed: int = 0) -> Path:
    """Spot-check sheet (stays under data/, never committed): mark each pair ok / added / lost meaning."""
    pairs = read_jsonl(base / cfg["out"]["pairs"])
    picked = random.Random(seed).sample(pairs, min(n, len(pairs)))
    lines = ["# Spot-check: mark [x] when the neutral text adds or loses meaning\n"]
    for p in picked:
        lines += [f"## {p['id']}  [ ]", "", "original:", "", p["output"], "", f"{p['variant']}:", "", p["input"], ""]
    path = (base / cfg["out"]["stats"]).with_name("review.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("submit", "collect", "build", "run", "review"))
    ap.add_argument("--pilot", action="store_true", help="only the pilot sample (configs: pilot.n)")
    ap.add_argument("--n", type=int, default=100, help="review sample size")
    ap.add_argument("--config", default="configs/neutralize.yaml")
    ap.add_argument("--base", default=".", help="repo root that config paths are relative to")
    args = ap.parse_args()
    base = Path(args.base)
    cfg = yaml.safe_load((base / args.config).read_text(encoding="utf-8"))

    if args.cmd == "review":
        print(review(cfg, base, args.n))
        return
    rows = scope(cfg, base, args.pilot)
    if args.cmd in ("submit", "collect", "run"):
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set in this shell")
        from openai import OpenAI

        client = OpenAI()
        if args.cmd in ("submit", "run"):
            bid = submit(client, cfg, base, rows)
            print(f"submitted {bid}" if bid else "nothing to submit: every burst is cached or pending")
        if args.cmd in ("collect", "run"):
            print(json.dumps(collect(client, cfg, base)))
    if args.cmd in ("build", "run"):
        n_all = len(read_jsonl(base / cfg["in"]))
        print(json.dumps(build(cfg, base, rows, n_all), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
