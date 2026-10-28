"""Drop neutral/original pairs that do not keep the meaning or still carry the voice (plan section 6)."""
import re
from collections import Counter

from stylesync.clean.drift import ABBREV, EMOTICON, VN_CHARS
from stylesync.clean.filters import PLACEHOLDER, dedup_key

# "ok", "vs", "ms" are fine in neutral Vietnamese (and "ms" is milliseconds).
LEFTOVER = re.compile(r"(?<![\w.<`/-])(" + "|".join(sorted(ABBREV - {"ok", "oke", "vs", "ms", "dr"}, key=len,
                                                                 reverse=True)) + r")(?![\w.>`/-])", re.IGNORECASE)
BACKTICK = re.compile(r"`([^`\n]+)`")
DIGITS = re.compile(r"\d+")
# Small numbers the rewrite may spell out ("2 cái" -> "hai cái"): same meaning, so not a lost number.
NUM_WORDS = {"0": ("không",), "1": ("một", "nhất"), "2": ("hai",), "3": ("ba",), "4": ("bốn", "tư"),
             "5": ("năm", "lăm"), "6": ("sáu",), "7": ("bảy",), "8": ("tám",), "9": ("chín",), "10": ("mười",)}
# Code-looking tokens the rewrite must copy verbatim: snake_case, camelCase, file names, paths.
IDENT = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"
                   r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
                   r"|\b[\w-]+\.(?:py|js|ts|tsx|jsx|go|rs|json|ya?ml|md|sql|sh|toml|html|css|txt|csv)\b"
                   r"|(?<![\w/])(?:~|\.{1,2})?/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+")


def hard_tokens(text: str) -> list[str]:
    return BACKTICK.findall(text) + IDENT.findall(text)


def spelled(digits: str, text: str) -> bool:
    return any(re.search(rf"(?<!\w){w}(?!\w)", text.lower()) for w in NUM_WORDS.get(digits, ()))


def check(original: str, neutral: str, len_ratio: tuple[float, float]) -> list[str]:
    """Reasons to reject the pair, empty when it is kept."""
    if not neutral.strip():
        return ["empty"]
    reasons = []
    if dedup_key(neutral) == dedup_key(original):
        reasons.append("unchanged")
    ratio = len(neutral) / max(len(original), 1)
    if not len_ratio[0] <= ratio <= len_ratio[1]:
        reasons.append("len_ratio")
    if Counter(PLACEHOLDER.findall(neutral)) != Counter(PLACEHOLDER.findall(original)):
        reasons.append("placeholder")
    lost = Counter(DIGITS.findall(original)) - Counter(DIGITS.findall(neutral))
    if any(not spelled(d, neutral) for d in lost):
        reasons.append("number")
    if any(t not in neutral for t in hard_tokens(original)):
        reasons.append("code")
    if LEFTOVER.search(BACKTICK.sub("", neutral)):
        reasons.append("abbrev")
    if EMOTICON.search(BACKTICK.sub("", neutral)):
        reasons.append("emoticon")
    if VN_CHARS.search(original.lower()) and not VN_CHARS.search(neutral.lower()):
        reasons.append("no_diacritics")
    return reasons
