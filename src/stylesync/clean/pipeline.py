"""Phase 1 pipeline: ingest -> junk -> burst -> PII -> length -> noise -> dedup -> language/diacritics -> cutoff -> threads -> mine.jsonl."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from stylesync.clean import drift
from stylesync.clean.burst import make_bursts
from stylesync.clean.filters import dedup_key, is_english, is_junk, is_noise, is_unaccented, length_ok, strip_markers
from stylesync.clean.pii import Scrubber
from stylesync.ingest import discord, messenger

LOADERS = {"messenger": messenger.load, "discord": discord.load}


def sync_threads(path: Path, infos: list[dict]) -> dict:
    """Load the thread config, add threads it does not know yet with defaults, write it back."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = cfg or {}
    for t in infos:
        if t["id"] in cfg:
            cfg[t["id"]]["n"] = t["n"]
            cfg[t["id"]].pop("tone", None)  # old configs tagged casual/work; now one voice
            continue
        cfg[t["id"]] = {"source": t["source"], "type": t["type"], "title": t["title"], "n": t["n"], "include": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    # Not splitlines(): chat text can hold U+2028 / U+0085, which json.dumps leaves unescaped.
    with path.open(encoding="utf-8", newline="\n") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(cfg: dict, base: Path) -> dict:
    stats: dict = {"steps": {}, "drift": {}}
    rows, infos, names = [], [], set()
    for src, scfg in cfg["sources"].items():
        r, i, n = LOADERS[src](base / scfg["root"])
        rows += r
        infos += i
        names |= n
    write_jsonl(base / cfg["out"]["interim"], rows)

    def count(step: str, items: list[dict]) -> None:
        stats["steps"][step] = dict(Counter(x["source"] for x in items))

    count("own_messages", rows)
    for r in rows:
        r["text"] = strip_markers(r["text"])
    rows = [r for r in rows if not is_junk(r["text"])]
    count("not_junk", rows)
    bursts = make_bursts(rows, cfg["burst_gap_s"])
    count("bursts", bursts)
    scrub = Scrubber(names)
    for b in bursts:
        b["text"] = scrub(b["text"])
    bursts = [b for b in bursts if length_ok(b["text"], cfg["min_len"], cfg["max_len"])]
    count("length_ok", bursts)
    bursts = [b for b in bursts if not is_noise(b["text"])]
    count("not_noise", bursts)
    seen, unique = set(), []
    for b in bursts:
        key = dedup_key(b["text"])
        if key not in seen:
            seen.add(key)
            unique.append(b)
    bursts = unique
    count("dedup", bursts)
    if cfg.get("drop_english", True):
        bursts = [b for b in bursts if not is_english(b["text"])]
        count("vietnamese", bursts)
    if cfg.get("drop_unaccented", True):
        bursts = [b for b in bursts if not is_unaccented(b["text"])]
        count("has_diacritics", bursts)
    write_jsonl(base / cfg["out"]["bursts"], bursts)

    # Drift is measured before the time cutoff: its job is to choose that cutoff.
    for src in cfg["sources"]:
        stats["drift"][src] = drift.by_quarter([b for b in bursts if b["source"] == src])

    cutoffs = {s: str(c["cutoff"]) if c.get("cutoff") else "" for s, c in cfg["sources"].items()}
    bursts = [b for b in bursts if b["ts"][:10] >= cutoffs[b["source"]]]
    count("after_cutoff", bursts)

    threads = sync_threads(base / cfg["threads"], infos)
    bursts = [b for b in bursts if threads[b["thread_id"]]["include"]]
    count("threads_included", bursts)

    def weight(b: dict) -> float:
        """Older data that drifts from today's voice is kept but counts less (plan section 5.3)."""
        scfg = cfg["sources"][b["source"]]
        recent = str(scfg.get("full_weight_from") or "")
        return 1.0 if b["ts"][:10] >= recent else float(scfg.get("old_weight", 1.0))

    out = [{"id": f"{b['source'][0]}{i:05d}", "source": b["source"], "thread_id": b["thread_id"],
            "ts": b["ts"], "n_msgs": b["n_msgs"], "weight": weight(b),
            "text": b["text"]}
           for i, b in enumerate(sorted(bursts, key=lambda b: (b["source"], b["ts"])))]
    write_jsonl(base / cfg["out"]["processed"], out)
    stats["placeholders"] = dict(Counter(p for o in out for p in
                                         ("<NAME>", "<URL>", "<PHONE>", "<EMAIL>", "<NUMBER>", "<SECRET>")
                                         if p in o["text"]))
    stats["weight"] = {f"{s}/{w}": n for (s, w), n in Counter((o["source"], o["weight"]) for o in out).items()}
    stats["mean_len"] = round(sum(len(o["text"]) for o in out) / max(len(out), 1), 1)
    (base / cfg["out"]["stats"]).write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    svg = base / cfg["out"]["drift_svg"]
    svg.parent.mkdir(parents=True, exist_ok=True)
    svg.write_text(drift.to_svg(stats["drift"].get("discord", []), "Discord bursts per quarter (scaled to max)"),
                   encoding="utf-8")
    return stats


def sample(processed: Path, n_by_source: dict[str, int], seed: int = 0) -> list[dict]:
    """Random review sample per source (plan section 5.5)."""
    rows = read_jsonl(processed)
    rng = random.Random(seed)
    picked = []
    for src, n in n_by_source.items():
        pool = [r for r in rows if r["source"] == src]
        picked += rng.sample(pool, min(n, len(pool)))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filter.yaml")
    ap.add_argument("--base", default=".", help="repo root that config paths are relative to")
    args = ap.parse_args()
    base = Path(args.base)
    cfg = yaml.safe_load((base / args.config).read_text(encoding="utf-8"))
    stats = run(cfg, base)
    print(json.dumps({k: stats[k] for k in ("steps", "weight", "placeholders", "mean_len")}, indent=1))


if __name__ == "__main__":
    main()
