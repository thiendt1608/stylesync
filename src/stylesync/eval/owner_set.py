"""Phase 5: the owner-written commit/comment set (plan section 9), turned into test-set examples.

The sheet is private Markdown, one item per `## <id> <commit|comment>` heading with a `neutral:` draft
and the owner's own version under `mine:`. A commit is rewritten the way `rewrite_style` does it: only
the body goes to the model, the conventional prefix and trailers are kept by code, so both sides are
compared on the body.
"""
import re

from stylesync.dataset.build import SYSTEM
from stylesync.mcp_server import split_commit

ITEM = re.compile(r"^## (\S+) (commit|comment)[ \t]*\n(.*?)(?=^## |\Z)", re.S | re.M)
PARTS = re.compile(r"neutral:[ \t]*\n(.*?)\n\s*mine:[ \t]*\n(.*)", re.S)


def parse(text: str) -> list[dict]:
    """[{id, task, neutral, mine}]; items whose `mine:` is still empty are left out."""
    items = []
    for m in ITEM.finditer(text):
        parts = PARTS.search(m.group(3))
        if parts and parts.group(2).strip():
            items.append({"id": m.group(1), "task": m.group(2),
                          "neutral": parts.group(1).strip(), "mine": parts.group(2).strip()})
    return items


def body(text: str, task: str) -> str:
    return split_commit(text)[1] if task == "commit" else text.strip()


def examples(items: list[dict]) -> list[dict]:
    """Test-set rows (same shape as Phase 3 splits); the task is the variant, so style splits by it."""
    return [{"id": it["id"], "source": "owner", "variant": it["task"], "ts": "",
             "prompt": [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": body(it["neutral"], it["task"])}],
             "completion": [{"role": "assistant", "content": body(it["mine"], it["task"])}]}
            for it in items]
