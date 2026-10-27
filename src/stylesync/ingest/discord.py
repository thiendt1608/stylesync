"""Discord data package (package/Messages) -> owner's messages in the common schema.

The package only contains messages the account sent, so there is no author field to filter on.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

DM_PREFIX = "Direct Message with "


def load(root: Path) -> tuple[list[dict], list[dict], set[str]]:
    """Return (owner's messages, thread infos, DM partner usernames)."""
    root = Path(root)
    index_file = root / "index.json"
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else {}
    rows, infos, names = [], [], set()
    for d in sorted(root.glob("c*/")):
        channel = json.loads((d / "channel.json").read_text(encoding="utf-8"))
        messages = json.loads((d / "messages.json").read_text(encoding="utf-8"))
        tid = "d:" + channel["id"]
        label = index.get(channel["id"]) or ""
        if label.startswith(DM_PREFIX) and "Unknown" not in label:
            names.add(label[len(DM_PREFIX):].split("#")[0])
        guild = channel.get("guild", {}).get("name", "")
        title = " / ".join(x for x in (guild, channel.get("name") or label) if x)
        infos.append({"id": tid, "source": "discord", "type": channel["type"], "title": title, "n": len(messages)})
        for m in messages:
            ts = datetime.fromisoformat(m["Timestamp"])  # "YYYY-MM-DD HH:MM:SS", UTC
            ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            rows.append({
                "source": "discord",
                "thread_id": tid,
                "thread_type": channel["type"],
                "ts": ts.isoformat(),
                "text": m.get("Contents") or "",
                "has_attachment": bool(m.get("Attachments")),
            })
    return rows, infos, names
