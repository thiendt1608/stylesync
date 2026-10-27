"""Replace personal data with placeholders so the model keeps the voice but not other people's details."""
import re

PATTERNS = [
    ("<SECRET>", re.compile(r"\b(sk-[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|AKIA[0-9A-Z]{16}"
                            r"|eyJ[\w-]{10,}\.[\w-]+\.?[\w-]*)")),  # last one: JWT
    # Any other long opaque token (hashes, keys, base64): needs both letters and digits.
    ("<SECRET>", re.compile(r"\b(?=[\w+/=-]*\d)(?=[\w+/=-]*[A-Za-z])[\w+/=-]{32,}")),
    ("<URL>", re.compile(r"https?://\S+|www\.\S+")),
    ("<EMAIL>", re.compile(r"[\w.+-]+@[\w-]+(\.[\w.-]+)?")),  # also "name@gmail" typed without the TLD
    ("<NAME>", re.compile(r"<@!?\d+>")),  # Discord user mention
    ("<CHANNEL>", re.compile(r"<#\d+>")),
    ("<PHONE>", re.compile(r"(?<!\d)(\+84|0)([ .-]?\d){9,10}(?!\d)")),
    ("<NUMBER>", re.compile(r"(?<!\d)\d(?:[ .-]?\d){8,}(?!\d)")),  # bank accounts, IDs
    ("<NUMBER>", re.compile(r"(?<![\d.,])\d{6,8}(?![\d.,])")),  # OTPs, student IDs
]
# "pw: abc123" / "mk là x9y": keep the keyword, hide a value that has a digit in it.
CREDENTIAL = re.compile(r"(?i)\b(mk|pw|pass|passwd|password|mật khẩu|otp)(\s*(?::|=|là)\s*)(?=\S*\d)\S+")
CUSTOM_EMOJI = re.compile(r"<a?(:\w+:)\d+>")
# Name parts that are also everyday words; replacing them would wreck normal sentences.
COMMON_WORDS = {
    "anh", "em", "chị", "cô", "chú", "bác", "ông", "bà", "con", "bạn", "mình", "tôi", "nam", "an", "vui",
    "long", "bình", "minh", "thành", "công", "đức", "phúc", "lộc", "tài", "hòa", "hoà", "xuân", "thu", "hạ",
    "đông", "my", "ngọc", "hồng", "lan", "mai", "hà", "giang", "sơn", "hải", "tâm", "thiện", "nhân",
    "đại", "nhật", "tiền", "quốc", "việt",
}
# Participants that are pages/shops/bots rather than people; their words are not names.
ORG_WORDS = {"shop", "store", "official", "page", "club", "team", "group", "center", "centre", "studio", "bot",
             "ai", "google", "cloud", "english", "fpt", "university", "đại học", "trường", "công ty", "cửa hàng",
             "academy", "community", "support", "facebook", "meta", "user", "người dùng"}


class Scrubber:
    """Full names match in any case; single parts only when capitalized and not a common word.

    Lowercase parts are left alone: participant lists include shops and pages, so their words ("tiền", "quán",
    "gia đình") are everyday chat. Lowercase given names in casual chat therefore stay (known gap).
    """

    def __init__(self, names: set[str]):
        full = {n.strip() for n in names if len(n.strip()) >= 4}
        people = [n for n in names if not any(w in ORG_WORDS for w in n.lower().split())
                  and not any(w in n.lower() for w in ORG_WORDS if " " in w)]
        # Title-case only: "AI", "FPT" in a display name are not something to scrub from every message.
        parts = {p for n in people for p in n.split()
                 if len(p) >= 2 and p[0].isupper() and p[1:].islower() and p.lower() not in COMMON_WORDS}
        self.full = _alternation(full, re.IGNORECASE)
        self.caps = _alternation(parts)

    def __call__(self, text: str) -> str:
        text = CUSTOM_EMOJI.sub(r"\1", text)
        text = CREDENTIAL.sub(r"\1\2<SECRET>", text)
        for placeholder, pattern in PATTERNS:
            text = pattern.sub(placeholder, text)
        for pattern in (self.full, self.caps):
            if pattern:
                text = pattern.sub("<NAME>", text)
        return text


def _alternation(words: set[str], flags: int = 0) -> re.Pattern | None:
    if not words:
        return None
    return re.compile(r"@?\b(" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")\b", flags)
