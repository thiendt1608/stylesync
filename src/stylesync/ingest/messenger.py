"""Messenger export (your_facebook_activity/messages) -> owner's messages in the common schema."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FOLDERS = ("inbox", "archived_threads", "e2ee_cutover")
ATTACHMENTS = ("photos", "videos", "files", "gifs", "audio_files", "sticker", "share")


def fix_mojibake(s: str) -> str:
    """Facebook writes UTF-8 bytes as latin-1 code points; undo it when that is what happened."""
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def read_threads(root: Path) -> list[dict]:
    """One dict per thread folder: path, title, participant names, messages (all message_*.json merged)."""
    threads = []
    for folder in FOLDERS:
        for d in sorted((root / folder).glob("*/")):
            files = sorted(d.glob("message_*.json"))
            if not files:
                continue
            messages, participants, title = [], set(), ""
            for f in files:
                j = json.loads(f.read_text(encoding="utf-8"))
                participants |= {fix_mojibake(p["name"]) for p in j.get("participants", [])}
                title = title or fix_mojibake(j.get("title", ""))
                messages += j.get("messages", [])
            threads.append({"path": f"{folder}/{d.name}", "title": title,
                            "participants": participants, "messages": messages})
    return threads


def detect_me(threads: list[dict]) -> str:
    """The owner is in the most threads and sends the most messages; both must agree.

    A friend can share every thread with the owner, so any participant tied for most threads qualifies.
    """
    in_threads = Counter(p for t in threads for p in t["participants"])
    senders = Counter(fix_mojibake(m.get("sender_name", "")) for t in threads for m in t["messages"])
    top = max(in_threads.values())
    by_sent = senders.most_common(1)[0][0]
    if in_threads[by_sent] != top:
        raise ValueError("cannot detect the owner: top sender is not in the most threads")
    return by_sent


def thread_id(path: str) -> str:
    return "m:" + hashlib.sha1(path.encode()).hexdigest()[:10]


def load(root: Path) -> tuple[list[dict], list[dict], set[str]]:
    """Return (owner's messages, thread infos, every participant name)."""
    threads = read_threads(Path(root))
    me = detect_me(threads)
    rows, infos, names = [], [], set()
    for t in threads:
        tid = thread_id(t["path"])
        ttype = "GROUP" if len(t["participants"]) > 2 else "DM"
        names |= t["participants"]
        mine = [m for m in t["messages"] if fix_mojibake(m.get("sender_name", "")) == me]
        infos.append({"id": tid, "source": "messenger", "type": ttype, "title": t["title"], "n": len(mine)})
        for m in mine:
            rows.append({
                "source": "messenger",
                "thread_id": tid,
                "thread_type": ttype,
                "ts": datetime.fromtimestamp(m["timestamp_ms"] / 1000, timezone.utc).isoformat(),
                "text": fix_mojibake(m.get("content", "")),
                "has_attachment": any(k in m for k in ATTACHMENTS),
            })
    return rows, infos, names
