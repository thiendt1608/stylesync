"""Usage history for retraining (plan section 10.6): every rewrite_style call and the owner's verdict on it.

An append-only JSONL file outside the repo, `platformdirs.user_data_dir("stylesync")/history.jsonl`, owner-only:
  {"type": "rewrite", "id", "ts", "task", "draft", "output", "guard_passed", "warnings"}
  {"type": "feedback", "id", "ts", "final", "by"}   the text the owner settled on (by: owner | agent)
  {"type": "review", "id", "ts", "action"}           keep | drop, from `stylesync-feedback review`
Appending one line per event keeps concurrent Claude Code sessions safe; for an id the latest feedback and review
win. STYLESYNC_HISTORY overrides the path, and "off" turns logging off.
"""
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import platformdirs

# The owner said nothing about a rewrite this long after it: it was used as is.
SILENT_ACCEPT = timedelta(hours=24)
BY = ("owner", "agent")


def path() -> Path | None:
    value = os.environ.get("STYLESYNC_HISTORY")
    if value == "off":
        return None
    return Path(value) if value else Path(platformdirs.user_data_dir("stylesync")) / "history.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(record: dict, p: Path) -> None:
    """One write call per line, so lines from two sessions never interleave. The file holds private messages."""
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def read(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    records = []
    with p.open(encoding="utf-8", newline="\n") as f:  # not splitlines(): messages can hold U+2028
        lines = [line for line in f if line.strip()]
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:  # a line cut short by a crash
            continue
    return records


def log_rewrite(task: str, draft: str, output: str, guard_passed: bool, warnings: list[str]) -> str | None:
    """Record a rewrite and return its id; None when logging is off or the file cannot be written."""
    p = path()
    if p is None:
        return None
    rid = uuid.uuid4().hex[:12]
    try:
        append({"type": "rewrite", "id": rid, "ts": now(), "task": task, "draft": draft, "output": output,
                "guard_passed": guard_passed, "warnings": warnings}, p)
    except OSError:
        return None
    return rid


def record_feedback(rid: str, final: str, by: str) -> str:
    """Store the text the owner settled on (empty: the rewrite was fine as is) and return the new status."""
    p = path()
    if p is None:
        raise LookupError("history is off (STYLESYNC_HISTORY=off)")
    rewrites = {r["id"]: r for r in read(p) if r.get("type") == "rewrite"}
    if rid not in rewrites:
        raise LookupError(f"unknown rewrite id {rid!r}")
    final = final if final.strip() else rewrites[rid]["output"]
    append({"type": "feedback", "id": rid, "ts": now(), "final": final, "by": by if by in BY else "agent"}, p)
    return status({**rewrites[rid], "final": final, "review": None}, datetime.now(timezone.utc))


def record_review(rid: str, action: str, p: Path) -> None:
    append({"type": "review", "id": rid, "ts": now(), "action": action}, p)


def status(e: dict, at: datetime) -> str:
    """pending | accepted (owner said fine) | silent (no word for 24h) | corrected | dropped (in review)."""
    if e["review"] == "drop":
        return "dropped"
    if e["final"] is not None:
        return "accepted" if e["final"].strip() == e["output"].strip() else "corrected"
    return "silent" if at - datetime.fromisoformat(e["ts"]) >= SILENT_ACCEPT else "pending"


def entries(records: list[dict], at: datetime | None = None) -> list[dict]:
    """One entry per rewrite, oldest first, with its latest feedback and review folded in."""
    at = at or datetime.now(timezone.utc)
    by_id: dict[str, dict] = {}
    for r in records:
        if r.get("type") == "rewrite":
            by_id[r["id"]] = {**r, "final": None, "by": None, "review": None}
        elif (e := by_id.get(r.get("id"))) is not None:
            if r["type"] == "feedback":
                e["final"], e["by"] = r["final"], r["by"]
            elif r["type"] == "review":
                e["review"] = r["action"]
    for e in by_id.values():
        e["status"] = status(e, at)
    return list(by_id.values())
