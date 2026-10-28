"""Phase 3: turn validated pairs into train/val/test files for SFT (plan section 7).

Each line is a TRL conversational prompt-completion example, so the trainer applies the base model's own
chat template and computes loss on the completion only:
  {"id", "source", "variant", "ts", "prompt": [system, user], "completion": [assistant]}
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from stylesync.clean.filters import dedup_key
from stylesync.clean.pipeline import read_jsonl, write_jsonl

# Training and runtime must send the same system prompt. No task or tone tag: one voice (plan 5.4).
SYSTEM = "Bạn là bộ viết lại văn phong. Giữ nguyên ý, code, identifier. Chỉ trả về bản viết lại."
SPLITS = ("train", "val", "test")


def example(p: dict) -> dict:
    return {"id": p["id"], "source": p["source"], "variant": p["variant"], "ts": p["ts"],
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["input"]}],
            "completion": [{"role": "assistant", "content": p["output"]}]}


def burst_id(p: dict) -> str:
    return p["id"].rsplit("-", 1)[0]


def assign(bursts: list[dict], val_frac: float, test_frac: float, embargo: timedelta) -> dict[str, str | None]:
    """Split one source by time: newest test_frac -> test, next val_frac -> val, rest -> train.

    Bursts within `embargo` before a boundary map to None (dropped). Bursts with the same ts stay together.
    """
    order = sorted((datetime.fromisoformat(b["ts"]), b["id"]) for b in bursts)
    n = len(order)
    if not n:
        return {}
    n_test, n_val = round(n * test_frac), round(n * val_frac)
    t_test = order[n - n_test][0] if n_test else None
    t_val = order[n - n_test - n_val][0] if n_val else t_test
    out: dict[str, str | None] = {}
    for t, bid in order:
        if t_test and t >= t_test:
            out[bid] = "test"
        elif t_val and t >= t_val:
            out[bid] = None if t_test and t >= t_test - embargo else "val"
        else:
            out[bid] = None if t_val and t >= t_val - embargo else "train"
    return out


def build(cfg: dict, base: Path) -> dict:
    pairs = read_jsonl(base / cfg["in"])
    by_burst: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_burst[burst_id(p)].append(p)
    bursts = [{"id": k, "source": v[0]["source"], "ts": v[0]["ts"], "weight": v[0]["weight"]}
              for k, v in by_burst.items()]
    split_of: dict[str, str | None] = {}
    for src in sorted({b["source"] for b in bursts}):
        split_of |= assign([b for b in bursts if b["source"] == src], cfg["val_frac"], cfg["test_frac"],
                           timedelta(hours=cfg["embargo_h"]))

    rng = random.Random(cfg["seed"])
    rows: dict[str, list[dict]] = {s: [] for s in SPLITS}
    dropped = Counter()
    for b in sorted(bursts, key=lambda b: b["id"]):
        split, variants = split_of[b["id"]], sorted(by_burst[b["id"]], key=lambda p: p["variant"])
        if split is None:
            dropped["embargo"] += len(variants)
            continue
        if split == "train":  # sample weight -> number of variants kept
            keep = min(len(variants), max(1, round(b["weight"] * 2)))
            if keep < len(variants):
                dropped["weight"] += len(variants) - keep
                variants = rng.sample(variants, keep)
        rows[split] += [example(p) for p in variants]

    out = base / cfg["out"]["dir"]
    for s in SPLITS:
        write_jsonl(out / f"{s}.jsonl", rows[s])
    stats = {"pairs_in": len(pairs), "bursts_in": len(bursts), "dropped": dict(dropped)} | report(rows)
    (base / cfg["out"]["stats"]).write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def report(rows: dict[str, list[dict]]) -> dict:
    """Counts, date ranges and lengths per split, plus leak checks between splits."""
    stats: dict = {"splits": {}}
    for s, ex in rows.items():
        ins = sorted(len(e["prompt"][1]["content"]) for e in ex) or [0]
        outs = sorted(len(e["completion"][0]["content"]) for e in ex) or [0]
        stats["splits"][s] = {
            "examples": len(ex), "bursts": len({e["id"].rsplit("-", 1)[0] for e in ex}),
            "by_source": dict(Counter(e["source"] for e in ex)), "by_variant": dict(Counter(e["variant"] for e in ex)),
            "ts_min": min((e["ts"] for e in ex), default=None), "ts_max": max((e["ts"] for e in ex), default=None),
            "input_chars_p99_max": [ins[int(0.99 * (len(ins) - 1))], ins[-1]],
            "output_chars_p99_max": [outs[int(0.99 * (len(outs) - 1))], outs[-1]],
        }
    keys = {s: {dedup_key(e["completion"][0]["content"]) for e in ex} for s, ex in rows.items()}
    stats["leak_same_output"] = {f"train-{s}": len(keys["train"] & keys[s]) for s in ("val", "test")}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--base", default=".", help="repo root that config paths are relative to")
    args = ap.parse_args()
    base = Path(args.base)
    cfg = yaml.safe_load((base / args.config).read_text(encoding="utf-8"))
    print(json.dumps(build(cfg, base), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
