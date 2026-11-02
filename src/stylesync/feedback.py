"""Usage history -> retraining data (plan section 10.6): `stylesync-feedback stats | review | export`.

  stats   counts per status, task and author, the correction rate, and progress toward the retraining target
  review  go through the corrections one by one in a terminal and drop the bad ones
  export  scrub PII and write training pairs to data/processed/feedback.jsonl, in the Phase 2 pair format plus
          `rejected` (the model's output, for preference training later)

A pair mirrors what the model sees at runtime: a commit's body only, code spans as <CODE1>, <CODE2>..., one
paragraph per pair when the draft and the final text have the same paragraphs.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from stylesync import history
from stylesync.clean.pii import Scrubber
from stylesync.clean.pipeline import write_jsonl
from stylesync.mcp_server import LETTER, PARA, SLOT, mask, split_commit

TARGET = 500  # corrections before the first retraining run (plan section 10.6)


def stats(entries: list[dict]) -> list[str]:
    st = Counter(e["status"] for e in entries)
    judged = st["corrected"] + st["accepted"] + st["silent"]
    corrected = [e for e in entries if e["status"] == "corrected"]
    lines = [f"rewrites: {len(entries)}  " + "  ".join(f"{k}: {st[k]}" for k in
                                                          ("pending", "accepted", "silent", "corrected", "dropped"))]
    if judged:
        lines.append(f"correction rate: {st['corrected'] / judged:.0%} of {judged} judged")
    by_task = Counter(e["task"] for e in corrected)
    by = Counter(e["by"] for e in corrected)
    lines += [f"corrections by task: {dict(by_task)}  by: {dict(by)}",
              f"not reviewed yet: {sum(e['review'] is None for e in corrected)}",
              f"toward retraining: {len(corrected)}/{TARGET}"]
    return lines


def _mask_like(text: str, spans: list[str]) -> str:
    """Put the draft's slots into another text: a span the owner kept becomes the same <CODEn>."""
    for i, span in enumerate(spans, 1):
        text = text.replace(span, f"<CODE{i}>")
    return text


def pairs(e: dict, scrub: Scrubber) -> list[dict]:
    """Training pairs from one corrected (or, with accepted, one used) rewrite."""
    target = e["final"] if e["final"] is not None else e["output"]
    texts = [e["draft"], target, e["output"]]
    if e["task"] == "commit":
        texts = [split_commit(t)[1] for t in texts]
    draft, spans = mask(texts[0].replace("\r\n", "\n"))
    target, rejected = (_mask_like(t.replace("\r\n", "\n"), spans) for t in texts[1:])
    split = [[p.strip("\n") for p in PARA.split(t)] for t in (draft, target, rejected)]
    if len({len(s) for s in split}) == 1:
        chunks = list(zip(*split))
    elif len(split[0]) == len(split[1]):  # the model's output has other paragraphs: no rejected per paragraph
        chunks = [(d, t, None) for d, t in zip(split[0], split[1])]
    else:
        chunks = [(draft, target, rejected)]
    out = []
    for d, t, r in chunks:
        if not LETTER.search(SLOT.sub("", d)) or not t.strip():  # only code, or nothing to learn
            continue
        out.append({"id": f"fb{e['id']}-{len(out) + 1}", "task": e["task"], "source": "feedback",
                    "thread_id": None, "ts": e["ts"], "weight": 1.0, "variant": e["by"] or e["status"],
                    "input": scrub(d), "output": scrub(t), "rejected": scrub(r) if r is not None else None})
    return out


def export(entries: list[dict], out: Path, accepted: bool) -> dict:
    keep = {"corrected", "accepted", "silent"} if accepted else {"corrected"}
    scrub = Scrubber(set())  # patterns only (contacts, secrets, numbers); names are not known here
    rows = [p for e in entries if e["status"] in keep and (e["status"] == "corrected" or e["output"] != e["draft"])
            for p in pairs(e, scrub)]
    write_jsonl(out, rows)
    return {"pairs": len(rows), "from": dict(Counter(r["variant"] for r in rows)), "out": str(out)}


def review(entries: list[dict], p: Path, everything: bool) -> int:
    todo = [e for e in entries if e["status"] == "corrected" and (everything or e["review"] is None)]
    done = 0
    for i, e in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {e['id']}  {e['task']}  {e['ts']}  by {e['by']}")
        for label, text in (("draft", e["draft"]), ("model", e["output"]), ("final", e["final"])):
            print(f"--- {label}\n{text}")
        while (answer := input("[k]eep / [d]rop / [s]kip / [q]uit: ").strip().lower()[:1]) not in ("k", "d", "s", "q"):
            pass
        if answer == "q":
            break
        if answer in ("k", "d"):
            history.record_review(e["id"], "keep" if answer == "k" else "drop", p)
            done += 1
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    rv = sub.add_parser("review")
    rv.add_argument("--all", action="store_true", help="also corrections already reviewed")
    ex = sub.add_parser("export")
    ex.add_argument("--out", type=Path, default=Path("data/processed/feedback.jsonl"))
    ex.add_argument("--accepted", action="store_true", help="also rewrites used as is (accepted or silent)")
    args = ap.parse_args()
    p = history.path()
    if p is None:
        sys.exit("history is off (STYLESYNC_HISTORY=off)")
    entries = history.entries(history.read(p))
    print(f"history: {p}")
    if args.cmd == "stats":
        print("\n".join(stats(entries)))
    elif args.cmd == "review":
        print(f"reviewed: {review(entries, p, args.all)}")
    else:
        print(export(entries, args.out, args.accepted))


if __name__ == "__main__":
    main()
