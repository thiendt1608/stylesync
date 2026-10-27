"""Merge consecutive messages of one thread into bursts."""
from datetime import datetime


def make_bursts(rows: list[dict], gap_s: int) -> list[dict]:
    """Join same-thread messages sent within gap_s of the previous one with "\\n".

    A burst keeps the first message's ts and fields; line breaks are kept because they are part of the voice.
    """
    rows = sorted(rows, key=lambda r: (r["thread_id"], r["ts"]))
    bursts: list[dict] = []
    last_ts = None
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        cur = bursts[-1] if bursts else None
        if cur and cur["thread_id"] == r["thread_id"] and (ts - last_ts).total_seconds() <= gap_s:
            cur["text"] += "\n" + r["text"].strip()
            cur["n_msgs"] += 1
        else:
            bursts.append({**r, "text": r["text"].strip(), "n_msgs": 1})
        last_ts = ts
    return bursts
