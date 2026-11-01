"""Minimal MCP server (Phase 0 spike): exposes rewrite_style backed by a local llama-server."""
import os
import re

import httpx
from mcp.server.mcpserver import MCPServer

LLAMA_URL = os.environ.get("STYLESYNC_LLAMA_URL", "http://127.0.0.1:18080")
SYSTEM = (
    "Bạn là bộ viết lại văn phong. Viết lại đoạn văn của người dùng theo giọng nhắn tin thân mật "
    "kiểu Gen Z Việt Nam: câu ngắn, dùng viết tắt phổ biến (kh, đc, z, nma, bt, ng, mn, vd), "
    "thuật ngữ kỹ thuật giữ tiếng Anh. Giữ NGUYÊN ý, không thêm, không bớt. "
    "Giữ nguyên mọi nội dung trong dấu backtick. Chỉ trả về bản viết lại, không giải thích.\n"
    "Loại văn bản: {task}"
)
PREFIX = re.compile(r"^[a-z]+(\([^)]*\))?!?:\s*")
TRAILER = re.compile(r"^[A-Za-z-]+: .+$")
VN = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
TASKS = ("message", "commit", "comment")
# Fenced blocks first, then inline spans. The adapters copy <NAME>-style placeholders every time but drop
# backticks (chat data has almost none), so code goes around the model as numbered placeholders.
SPAN = re.compile(r"```.*?```|`[^`\n]+`", re.S)
SLOT = re.compile(r"<CODE(\d+)>")

mcp = MCPServer("stylesync")


def split_commit(text: str) -> tuple[str, str, str]:
    """Split a commit message into (conventional prefix, rewritable body, trailer block)."""
    lines = text.strip().splitlines()
    trailers: list[str] = []
    # As in git, trailers are the last paragraph and need a blank line before them, so a one-line
    # "refactor: ..." subject is never mistaken for a "Token: value" trailer.
    blank = max((i for i, line in enumerate(lines) if not line.strip()), default=None)
    if blank is not None and all(TRAILER.match(line) for line in lines[blank + 1:]):
        trailers, lines = lines[blank + 1:], lines[:blank]
    while lines and not lines[-1].strip():
        lines.pop()
    body = "\n".join(lines)
    m = PREFIX.match(body)
    prefix = m.group(0) if m else ""
    return prefix, body[len(prefix):], "\n".join(trailers)


def mask(text: str) -> tuple[str, list[str]]:
    """Swap each code span for <CODE1>, <CODE2>, ...; returns the masked text and the spans in order.
    A text that already holds such a token is left as it is, so unmask can never misplace a span."""
    if SLOT.search(text):
        return text, []
    spans: list[str] = []

    def slot(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"<CODE{len(spans)}>"

    return SPAN.sub(slot, text), spans


def unmask(text: str, spans: list[str]) -> str:
    """Put the spans back; a slot number the model made up stays, and the guard rejects it."""
    return SLOT.sub(lambda m: spans[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(spans) else m.group(0), text)


def guard(source: str, output: str) -> list[str]:
    """Return guard violations; an empty list means the rewrite can be used."""
    problems = []
    if not output.strip():
        problems.append("empty output")
    spans = re.findall(r"`[^`]+`", source)
    for token in spans:
        if token not in output:
            problems.append(f"lost code span {token}")
    for token in re.findall(r"`[^`]+`", output):
        if token not in spans and token.strip("`") not in source:  # quoting an existing word is fine
            problems.append(f"added code span {token}")
    # A long output without a single Vietnamese diacritic means the model switched to English;
    # short casual replies ("fix `x` r nha") legitimately have none.
    if VN.search(source.lower()) and not VN.search(output.lower()) and len(output.split()) >= 6:
        problems.append("Vietnamese lost (no diacritics in output)")
    if re.search(r"<(NAME|PHONE|EMAIL|CODE\d+)>", output):
        problems.append("placeholder leaked")
    ratio = len(output) / max(len(source), 1)
    if not 0.3 <= ratio <= 2.5:
        problems.append(f"length ratio {ratio:.2f} out of range")
    return problems


def _generate(text: str, task: str) -> str:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM.format(task=task)},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = httpx.post(f"{LLAMA_URL}/v1/chat/completions", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


@mcp.tool()
def rewrite_style(text: str, task: str = "message") -> dict:
    """Rewrite neutral text into the owner's personal voice (Vietnamese, casual, abbreviations).

    Call this before writing a commit message, a PR/issue comment, or a chat message on the
    owner's behalf. Pass a neutral, complete draft as `text` and `task` as one of
    message | commit | comment. Use the returned `text` VERBATIM: do not fix spelling,
    abbreviations or tone. If `guard_passed` is false, `text` is the neutral draft; use it and
    tell the owner.
    """
    if task not in TASKS:
        task = "message"
    prefix, core, trailers = split_commit(text) if task == "commit" else ("", text.strip(), "")
    masked, spans = mask(core)
    try:
        rewritten = unmask(_generate(masked, task), spans)
    except (httpx.HTTPError, KeyError, IndexError) as e:
        return {"text": text, "guard_passed": False, "warnings": [f"local model unavailable: {e}"]}
    problems = guard(core, rewritten)
    if problems:
        return {"text": text, "guard_passed": False, "warnings": problems}
    result = prefix + rewritten + (f"\n\n{trailers}" if trailers else "")
    return {"text": result, "guard_passed": True, "warnings": []}


@mcp.tool()
def health() -> dict:
    """Report whether the local llama-server backend is reachable."""
    try:
        ok = httpx.get(f"{LLAMA_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        ok = False
    return {"backend": LLAMA_URL, "ok": ok}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
