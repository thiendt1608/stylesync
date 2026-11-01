"""Phase 6: export the chosen LoRA adapter to GGUF and check the GGUF build against Phase 5 (plan section 10.1).

  stylesync-export convert                        adapter -> LoRA GGUF (llama.cpp converter, separate venv)
  stylesync-export parity --device metal|cpu [--set chat|owner] [--limit N]
  stylesync-export report

`parity` starts its own llama-server (base GGUF + `--lora`) on a spare port and rewrites a sample of the test
set greedily with the training prompt, as Phase 5 `gen` did with PyTorch. Requests go with `cache_prompt` off,
so an output does not depend on the request before it, and each request checks BOS. Training and Phase 5
used exactly one (Unsloth warned that the Gemma template plus the tokenizer gives two and dropped one).
llama-server strips the template's `<bos>` and adds its own, so the check is: the server's `prompt_n` equals
the rendered template tokenized with special tokens added, and that starts with exactly one BOS.
"""
import argparse
import random
import statistics
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import httpx

from stylesync.clean.pipeline import read_jsonl
from stylesync.eval.metrics import checks, stylometry
from stylesync.eval.run import load_cfg
from stylesync.mcp_server import guard, mask, unmask
from stylesync.neutralize.run import append_jsonl

DEVICES = ("metal", "cpu")


def chrf(hyp: str, ref: str, order: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015, as sacrebleu computes it): character n-grams 1..order without spaces, 0-100."""
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    if hyp == ref:
        return 100.0
    precision, recall = [], []
    for n in range(1, order + 1):
        h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        match = sum((h & r).values())
        precision.append(match / sum(h.values()))
        recall.append(match / sum(r.values()))
    if not precision:
        return 0.0
    p, r = statistics.mean(precision), statistics.mean(recall)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def sample(test: list[dict], ids: set[str], n: int | None, seed: int) -> list[dict]:
    """Test rows that have a reference output; a fixed random n of them when n is set, in test order."""
    rows = [r for r in test if r["id"] in ids]
    if n is None or n >= len(rows):
        return rows
    keep = {r["id"] for r in random.Random(seed).sample(rows, n)}
    return [r for r in rows if r["id"] in keep]


def request(row: dict, masked: bool) -> tuple[list[dict], list[str]]:
    """Chat turns for one row (training system prompt) and the code spans taken out of the user turn."""
    turns = row["prompt"]
    if not masked:
        return turns, []
    text, spans = mask(turns[-1]["content"])
    return turns[:-1] + [{"role": "user", "content": text}], spans


def server_args(cfg: dict, base: Path, device: str) -> list[str]:
    s = cfg["server"]
    args = [s["bin"], "-m", str(base / cfg["base_gguf"]), "--lora", str(base / cfg["lora_gguf"]),
            "--host", "127.0.0.1", "--port", str(s["port"]), "-c", str(s["ctx"]), "-np", "1",
            "--reasoning", "off"]
    return args + (["-ngl", "99"] if device == "metal" else ["-ngl", "0", "-dev", "none", "-t", str(s["threads"])])


@contextmanager
def server(args: list[str], log: Path, port: int):
    """Run llama-server until the block ends; its log goes to `log` (scanned for BOS warnings)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=300)
        try:
            t0 = time.perf_counter()
            while True:
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if proc.poll() is not None:
                    raise SystemExit(f"llama-server exited with {proc.returncode}, see {log}")
                time.sleep(0.2)
            yield client, time.perf_counter() - t0
        finally:
            client.close()
            proc.terminate()
            proc.wait(timeout=30)


def leading(tokens: list[dict], piece: str) -> int:
    n = 0
    while n < len(tokens) and tokens[n]["piece"] == piece:
        n += 1
    return n


def rewrite(client: httpx.Client, turns: list[dict], spans: list[str], max_tokens: int, bos: str) -> dict:
    prompt = client.post("/apply-template", json={"messages": turns}).json()["prompt"]
    tokens = client.post("/tokenize", json={"content": prompt, "add_special": True, "parse_special": True,
                                            "with_pieces": True}).json()["tokens"]
    body = {"messages": turns, "temperature": 0, "max_tokens": max_tokens, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.perf_counter()
    r = client.post("/v1/chat/completions", json=body).json()
    wall = time.perf_counter() - t
    tm = r["timings"]
    return {"output": unmask(r["choices"][0]["message"]["content"].strip(), spans), "wall_s": round(wall, 3),
            "prompt_n": tm["prompt_n"], "template_n": len(tokens), "bos": leading(tokens, bos),
            "predicted_n": tm["predicted_n"], "predicted_ms": tm["predicted_ms"],
            "hit_max_tokens": r["choices"][0]["finish_reason"] == "length"}


def out_path(cfg: dict, base: Path, name: str, device: str) -> Path:
    return base / cfg["outputs_dir"] / f"{name}-{device}.jsonl"


def parity(cfg: dict, base: Path, name: str, device: str, limit: int | None) -> Path:
    sc = cfg["sets"][name]
    ref = {r["id"]: r["output"] for r in read_jsonl(base / sc["reference"])}
    rows = sample(read_jsonl(base / sc["test"]), set(ref), limit or sc.get("sample"), cfg["seed"])
    out = out_path(cfg, base, name, device)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    if not rows:
        return out
    log = base / cfg["outputs_dir"] / f"server-{device}.log"
    with server(server_args(cfg, base, device), log, cfg["server"]["port"]) as (client, load_s):
        print(f"llama-server ({device}) ready in {load_s:.1f}s", flush=True)
        bos = client.get("/props").json()["bos_token"]
        for i, row in enumerate(rows, 1):
            turns, spans = request(row, sc.get("mask", False))
            append_jsonl(out, [{"id": row["id"], **rewrite(client, turns, spans, cfg["max_tokens"], bos)}])
            if i % 20 == 0 or i == len(rows):
                print(f"{name}-{device}: {len(done) + i}/{len(done) + len(rows)}", flush=True)
    if "Added a BOS token" in log.read_text(encoding="utf-8", errors="replace"):
        print(f"WARNING: llama-server logged a double BOS, see {log}")
    return out


def summarize(rows: list[dict], test: dict[str, dict], ref: dict[str, str]) -> dict:
    """Aggregates only: agreement with the PyTorch output, guard and hard tokens, BOS, latency."""
    n = len(rows)
    src = {i: test[i]["prompt"][-1]["content"] for i in (r["id"] for r in rows)}
    tok_s = [r["predicted_n"] / r["predicted_ms"] * 1000 for r in rows if r["predicted_ms"]]
    walls = sorted(r["wall_s"] for r in rows)
    with_tokens = [r for r in rows if "`" in src[r["id"]]]
    return {
        "n": n,
        "exact": sum(r["output"] == ref[r["id"]] for r in rows) / n,
        "chrf": statistics.mean(chrf(r["output"], ref[r["id"]]) for r in rows),
        "guard": sum(not guard(src[r["id"]], r["output"]) for r in rows) / n,
        "guard_ref": sum(not guard(src[r["id"]], ref[r["id"]]) for r in rows) / n,
        "hard_tokens": (sum(checks(src[r["id"]], r["output"])["hard_tokens"] for r in with_tokens) / len(with_tokens)
                        if with_tokens else None),
        "bos_ok": sum(r["prompt_n"] == r["template_n"] and r["bos"] == 1 for r in rows) / n,
        "hit_max": sum(r["hit_max_tokens"] for r in rows) / n,
        "wall_mean": statistics.mean(walls), "wall_p90": walls[min(n - 1, int(0.9 * n))],
        "decode_tok_s": statistics.mean(tok_s) if tok_s else None,
        "mean_len": stylometry([r["output"] for r in rows])["mean_len"],
        "mean_len_ref": stylometry([ref[r["id"]] for r in rows])["mean_len"],
    }


COLUMNS = ("n", "exact", "chrf", "guard", "guard_ref", "hard_tokens", "bos_ok", "hit_max",
           "mean_len", "mean_len_ref", "wall_mean", "wall_p90", "decode_tok_s")
PERCENT = {"exact", "guard", "guard_ref", "hard_tokens", "bos_ok", "hit_max"}


def cell(col: str, v) -> str:
    if v is None:
        return "–"
    if col in PERCENT:
        return f"{v:.0%}"
    return f"{v:.2f}s" if col.startswith("wall") else (str(v) if isinstance(v, int) else f"{v:.1f}")


def report(cfg: dict, base: Path) -> Path:
    lines = []
    for name, sc in cfg["sets"].items():
        test = {r["id"]: r for r in read_jsonl(base / sc["test"])}
        ref = {r["id"]: r["output"] for r in read_jsonl(base / sc["reference"])}
        for device in DEVICES:
            path = out_path(cfg, base, name, device)
            if path.exists():
                s = summarize(read_jsonl(path), test, ref)
                lines.append(f"| {name} | {device} | " + " | ".join(cell(c, s[c]) for c in COLUMNS) + " |")
    head = ["| set | device | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 2)]
    path = base / cfg["report"]
    path.write_text("\n".join([
        "# Phase 6 GGUF parity",
        "",
        f"Generated by `stylesync-export report`; aggregates only. Build: `{Path(cfg['base_gguf']).name}` + "
        f"`--lora {Path(cfg['lora_gguf']).name}` (adapter `{cfg['exp']}`), llama-server, greedy, training prompt. "
        "Reference: the Phase 5 PyTorch output of the same adapter (bf16, greedy).",
        "",
        "- `exact`, `chrf`: share identical to the PyTorch output, and mean chrF against it (0-100).",
        "- `guard` / `guard_ref`: MCP guard pass rate of the GGUF / PyTorch outputs on the same rows. "
        "`hard_tokens`: backtick text kept, over rows with a backtick in the input.",
        "- `bos_ok`: the server evaluated the rendered template with exactly one BOS in front (as in training).",
        "- `hit_max`: share that stopped at max_tokens (a sign of a broken template or EOS).",
        "- `mean_len` / `mean_len_ref`: mean characters, GGUF / PyTorch. Latency is per request with "
        "`cache_prompt` off (full prefill every time); the MCP server keeps the cache on, so real calls are faster.",
        "",
        *head, *lines, "",
    ]), encoding="utf-8")
    return path


def convert(cfg: dict, base: Path) -> Path:
    c, out = cfg["converter"], base / cfg["lora_gguf"]
    train = load_cfg(base, cfg["train_config"])
    repo = train["models"][train["experiments"][cfg["exp"]]["model"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(base / c["python"]), str(base / c["script"]), str(base / cfg["adapter"]),
                    "--base-model-id", repo, "--outtype", "f16", "--outfile", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["convert", "parity", "report"])
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--device", choices=DEVICES, default="metal")
    ap.add_argument("--set", choices=["chat", "owner"], dest="sets", action="append",
                    help="repeatable; default: every set in the config")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    base = Path.cwd()
    cfg = load_cfg(base, args.config)
    if args.cmd == "convert":
        print(convert(cfg, base))
    elif args.cmd == "parity":
        for name in args.sets or list(cfg["sets"]):
            print(parity(cfg, base, name, args.device, args.limit))
        print(report(cfg, base))
    else:
        print(report(cfg, base))


if __name__ == "__main__":
    main()
