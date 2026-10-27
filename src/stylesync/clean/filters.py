"""Per-message junk rules, length check and dedup key."""
import re
import unicodedata

from stylesync.clean.drift import text_metrics

URL = re.compile(r"https?://\S+|www\.\S+")
# Messenger writes these as the sender's own messages (English UI strings).
SYSTEM = re.compile(
    r"^(You (sent an attachment|unsent a message|pinned|unpinned|joined|started|named|changed|set|created"
    r"|added|removed|left|missed|called|opened|reacted)\b"
    r"|.*\breacted\b.+\bto (your|\S+'s) message"
    r"|.*\b(started|joined|ended|missed) (a |the )?(video |audio )?call\b"
    r"|(The )?(video |audio )?call ended\.?$"
    r"|.*đã gửi một (file đính kèm|ảnh|nhãn dán)"
    r"|.*đã thu hồi một tin nhắn)",
    re.IGNORECASE,
)


def strip_markers(text: str) -> str:
    """Drop export artifacts that are not part of what was typed."""
    return re.sub(r"\s*\(edited\)\s*$", "", text)


def is_junk(text: str) -> bool:
    """Attachment/sticker-only (empty text), system notices, link-only and emoji-only messages."""
    t = text.strip()
    if not t or SYSTEM.match(t):
        return True
    # No letter or digit left once links are gone: link-only, emoji-only, ":))" alone.
    return not any(ch.isalnum() for ch in URL.sub("", t))


PLACEHOLDER = re.compile(r"<[A-Z]+>")
EMOTICON = re.compile(r"[=:;]+[()]+|[()]+[=:;]+")  # "=))", ":((" are the owner's voice, not code
CODE_CHARS = set("{}[];=()<>")
CODE_LINE = re.compile(r"[;{}]\s*$|^\s*(SELECT|INSERT|UPDATE|DELETE|ALTER|CREATE|DROP)\s+[A-Z]+|^\s*(def|class|import|const|let|var"
                       r"|function|return|local)\s")


def is_noise(text: str) -> bool:
    """Bursts that are not the owner's prose: pasted code, spam repeats, mostly placeholders."""
    tokens = text.split()
    bare = EMOTICON.sub("", PLACEHOLDER.sub("", text))
    code = sum(ch in CODE_CHARS for ch in bare)
    if code >= 4 and code / max(len(bare), 1) >= 0.05:
        return True
    lines = [x for x in bare.split("\n") if x.strip()]
    if lines and sum(bool(CODE_LINE.search(x)) for x in lines) / len(lines) >= 0.3:
        return True
    if len(tokens) >= 6 and len({t.lower() for t in tokens}) / len(tokens) < 0.35:
        return True
    return sum(bool(PLACEHOLDER.fullmatch(t.strip(".,!?/"))) for t in tokens) / max(len(tokens), 1) >= 0.4


def length_ok(text: str, lo: int, hi: int) -> bool:
    return lo <= len(text) <= hi


def dedup_key(text: str) -> str:
    """Lowercase, drop punctuation, squeeze whitespace and long letter runs ("kkkkk" == "kkk")."""
    t = unicodedata.normalize("NFC", text.lower())
    t = re.sub(r"[^\w\s<>]", "", t)
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)
    return " ".join(t.split())


def is_unaccented(text: str) -> bool:
    """Six-plus words and not one diacritic: old no-accent typing, which the runtime guard rejects anyway."""
    m = text_metrics(text)
    return m["words"] >= 6 and m["diacritic_words"] == 0


def is_english(text: str) -> bool:
    """Bursts written in English (no diacritics, mostly non-Vietnamese words) teach the model to switch language."""
    m = text_metrics(text)
    return m["words"] >= 4 and m["diacritic_words"] == 0 and m["english"] / m["words"] >= 0.4
