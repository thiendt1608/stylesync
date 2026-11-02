"""stylesync MCP server (plan section 10.3): tools rewrite_style, rewrite_feedback and health over stdio.

A thin layer over the shared llama-server (`stylesync.llama`): the conventional-commit prefix, trailers and code
spans are handled here in code, the model only rewrites the rest, and a guard falls back to the neutral text.
Each rewrite and the text the owner settled on go to the usage history (`stylesync.history`) for retraining.
"""
import re
from collections import Counter

import httpx
from mcp.server.mcpserver import MCPServer

from stylesync import history, llama
# The training system prompt: the adapter was trained with it (Phase 3) and evaluated with it (Phase 5-6).
from stylesync.dataset.build import SYSTEM
PREFIX = re.compile(r"^[a-z]+(\([^)]*\))?!?:\s*")
TRAILER = re.compile(r"^[A-Za-z-]+: .+$")
VN = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
TASKS = ("message", "commit", "comment")
# Fenced blocks first, then inline spans. The adapters copy <NAME>-style placeholders every time but drop
# backticks (chat data has almost none), so code goes around the model as numbered placeholders.
SPAN = re.compile(r"```.*?```|`[^`\n]+`", re.S)
SLOT = re.compile(r"<CODE(\d+)>")
# Contact details the model must not invent (the training data had them scrubbed to placeholders).
PARA = re.compile(r"\n[ \t]*\n")
LETTER = re.compile(r"[^\W\d_]")
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://\S+|(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)")

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


def coverage(source: str, output: str, n: int = 3) -> float:
    """Share of the draft's character n-grams (with their counts) that the rewrite still has."""
    def grams(text: str) -> Counter:
        text = re.sub(r"\s+", " ", text.lower())
        return Counter(text[i:i + n] for i in range(len(text) - n + 1))
    src, out = grams(source), grams(output)
    total = sum(src.values())
    return sum(min(c, out[g]) for g, c in src.items()) / total if total else 1.0


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
    for token in {m.rstrip(".,;:!?)") for m in PII.findall(output)}:  # "…/docs," in the draft, "…/docs" here
        if token not in source:
            problems.append(f"new contact detail {token}")
    # Catches a rewrite that drops a large part of a long draft. On the Phase 5 test set no rewrite the judge kept
    # whole falls under 0.25 at 20+ words (median 0.60); losing one detail cannot be told apart this way.
    if len(source.split()) >= 20 and (c := coverage(source, output)) < 0.25:
        problems.append(f"most of the draft is missing (coverage {c:.2f})")
    ratio = len(output) / max(len(source), 1)
    if not 0.3 <= ratio <= 2.5:
        problems.append(f"length ratio {ratio:.2f} out of range")
    return problems


STATE = {"last_latency_s": None, "started_server": False}


def _generate(text: str) -> str:
    s = llama.Settings.from_env()
    with httpx.Client(base_url=s.url) as client:
        STATE["started_server"] |= llama.ensure(client, s)
        out, seconds = llama.chat(client, SYSTEM, text)
    STATE["last_latency_s"] = round(seconds, 3)
    return out


def _rewrite_paragraphs(masked: str, subject: bool) -> str:
    """Send one paragraph at a time: the adapter learned single chat bursts, and given several it fills the blank
    line with filler ("kk") and runs a commit's subject into its body. A paragraph that is only code is kept."""
    out = []
    for i, para in enumerate(p.strip("\n") for p in PARA.split(masked)):
        if not LETTER.search(SLOT.sub("", para)):
            out.append(para)
            continue
        new = _generate(para)
        if subject and i == 0:  # a commit subject stays one line
            new = " ".join(line.strip() for line in new.splitlines() if line.strip())
        out.append(new)
    return "\n\n".join(p for p in out if p)


def _logged(task: str, draft: str, result: dict) -> dict:
    result["id"] = history.log_rewrite(task, draft, result["text"], result["guard_passed"], result["warnings"])
    return result


@mcp.tool()
def rewrite_style(text: str, task: str = "message") -> dict:
    """Rewrite a neutral Vietnamese draft into the owner's own voice (casual, abbreviations, Gen Z).

    Call this before writing a commit message, a PR/issue comment, or a chat message on the
    owner's behalf. `text` is a complete neutral draft in Vietnamese that addresses people with
    anh/em, as the owner does (not tôi/bạn); keep code, paths and identifiers in backticks.
    `task` is message | commit | comment; for a commit pass the whole message: the
    `type(scope):` prefix and trailers are kept by code. Use the returned `text` VERBATIM: do
    not fix spelling, abbreviations or tone. If `guard_passed` is false, `text` is the neutral
    draft; use it and tell the owner why (`warnings`). Never send a message to another person
    without the owner seeing it first. Keep the returned `id` for rewrite_feedback.
    """
    if task not in TASKS:
        task = "message"
    prefix, core, trailers = split_commit(text) if task == "commit" else ("", text.strip(), "")
    masked, spans = mask(core.replace("\r\n", "\n"))
    prose = SLOT.sub("", masked)
    if not core.strip():
        return {"text": text, "guard_passed": False, "warnings": ["empty draft: nothing to rewrite"], "id": None}
    if not prose.strip():  # only code: already final
        return {"text": text, "guard_passed": True, "warnings": [], "id": None}
    if not VN.search(prose.lower()) and len(prose.split()) >= 4:
        return {"text": text, "guard_passed": False, "warnings": ["draft is not Vietnamese: left unchanged"], "id": None}
    # From here on the owner gets a Vietnamese text to judge, so the call goes to the history even when the
    # neutral draft comes back: a correction of it is as good a training pair as a correction of a rewrite.
    try:
        rewritten = unmask(_rewrite_paragraphs(masked, subject=task == "commit"), spans)
    except llama.Truncated as e:
        return _logged(task, text, {"text": text, "guard_passed": False, "warnings": [f"rewrite incomplete: {e}"]})
    except (llama.BackendError, httpx.HTTPError, KeyError, IndexError) as e:
        return _logged(task, text, {"text": text, "guard_passed": False, "warnings": [f"local model unavailable: {e}"]})
    problems = guard(core, rewritten)
    if problems:
        return _logged(task, text, {"text": text, "guard_passed": False, "warnings": problems})
    result = prefix + rewritten + (f"\n\n{trailers}" if trailers else "")
    return _logged(task, text, {"text": result, "guard_passed": True, "warnings": []})


@mcp.tool()
def rewrite_feedback(id: str, final_text: str = "", by: str = "agent") -> dict:
    """Save the owner's version of an earlier rewrite_style result; it becomes training data.

    Call it right after you apply a change the owner asked for to a rewrite, and again after
    each further change (the latest wins): `id` is the one rewrite_style returned, `final_text`
    the exact new text in full (the whole commit message, comment or message), `by` is "owner"
    when the owner typed that text and "agent" when you edited it on the owner's instructions.
    Call it with an empty `final_text` when the owner says the rewrite is fine as is; when the
    owner says nothing, do not call it (it counts as accepted after 24 hours). To fix a
    rewrite, edit it yourself as the owner asks: calling rewrite_style again returns the same
    text (decoding is greedy).
    """
    try:
        return {"saved": True, "status": history.record_feedback(id, final_text, by), "warnings": []}
    except (LookupError, OSError) as e:
        return {"saved": False, "status": None, "warnings": [str(e)]}


@mcp.tool()
def health() -> dict:
    """Report the local llama-server backend: reachable, stylesync adapter loaded, model files, last latency."""
    s = llama.Settings.from_env()
    with httpx.Client(base_url=s.url) as client:
        ok = llama.healthy(client)
        adapter = ok and llama.adapter_loaded(client, s)
    return {"backend": s.url, "ok": ok, "adapter_loaded": adapter, "base": str(s.base), "lora": str(s.lora),
            "model_files_present": s.base.is_file() and s.lora.is_file(), "history": str(history.path()), **STATE}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
