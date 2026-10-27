"""Per-quarter style metrics, used to pick the Discord cutoff (plan section 5.3)."""
import re
from collections import defaultdict

ABBREV = {"kh", "k", "ko", "hong", "đc", "dc", "z", "v", "bt", "bth", "nma", "ng", "ngta", "mn", "vd", "thoi",
          "dk", "r", "j", "cx", "ns", "trc", "ms", "vs", "lm", "kk", "kkk", "dr", "vl", "vc", "clm", "oke", "ok"}
VN_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
EMOTICON = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]|=\)\)|:\)\)|:v|:3|:d\b|=\)", re.IGNORECASE)
# A Vietnamese syllable typed without diacritics; an ASCII word that is not one is (usually) English.
VN_SYLLABLE = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?"
                         r"[aeiouy]{1,3}(ng|nh|ch|[cmnpt])?$")
# English function words that also pass as unaccented Vietnamese syllables ("the", "to"...).
EN_COMMON = {"the", "you", "and", "is", "are", "was", "for", "that", "this", "but", "with", "have", "what", "how",
             "not", "can", "of", "it", "i", "please", "thank", "sorry", "yes", "will", "do", "to", "if", "so"}
WORD = re.compile(r"[^\W\d_]+")
METRICS = ("n", "mean_len", "abbrev", "diacritics", "emoticon", "codemix")


def quarter(ts: str) -> str:
    return f"{ts[:4]}Q{(int(ts[5:7]) - 1) // 3 + 1}"


def text_metrics(text: str) -> dict:
    words = [w for w in WORD.findall(text.lower()) if w not in {"name", "url", "phone", "email"}]
    ascii_words = [w for w in words if w.isascii() and w not in ABBREV]
    return {
        "len": len(text),
        "words": len(words),
        "abbrev": sum(w in ABBREV for w in words),
        "diacritic_words": sum(bool(VN_CHARS.search(w)) for w in words),
        "emoticon": bool(EMOTICON.search(text)),
        "english": sum(w in EN_COMMON or (len(w) >= 3 and not VN_SYLLABLE.match(w)) for w in ascii_words),
    }


def by_quarter(rows: list[dict]) -> list[dict]:
    """Rates per quarter: abbrev/diacritics/codemix are shares of words, emoticon is the share of bursts."""
    acc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        acc[quarter(r["ts"])].append(text_metrics(r["text"]))
    out = []
    for q in sorted(acc):
        ms = acc[q]
        words = max(sum(m["words"] for m in ms), 1)
        out.append({
            "quarter": q,
            "n": len(ms),
            "mean_len": round(sum(m["len"] for m in ms) / len(ms), 1),
            "abbrev": round(sum(m["abbrev"] for m in ms) / words, 3),
            "diacritics": round(sum(m["diacritic_words"] for m in ms) / words, 3),
            "emoticon": round(sum(m["emoticon"] for m in ms) / len(ms), 3),
            "codemix": round(sum(m["english"] for m in ms) / words, 3),
        })
    return out


def to_svg(rows: list[dict], title: str, min_n: int = 30) -> str:
    """Small line chart, every metric scaled to its own max so the shapes can be compared."""
    rows = [r for r in rows if r["n"] >= min_n]
    if not rows:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'
    keys = ["mean_len", "abbrev", "diacritics", "emoticon", "codemix"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
    w, h, pad = 760, 300, 40
    step = (w - 2 * pad) / max(len(rows) - 1, 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + 60}" font-family="sans-serif" '
             f'font-size="11"><rect width="100%" height="100%" fill="#fff"/>'
             f'<text x="{pad}" y="20" font-size="13">{title}</text>']
    for i, r in enumerate(rows):
        x = pad + i * step
        parts.append(f'<text x="{x:.0f}" y="{h + 5}" transform="rotate(60 {x:.0f} {h + 5})">{r["quarter"]}</text>')
    for k, c in zip(keys, colors):
        top = max(r[k] for r in rows) or 1
        pts = " ".join(f"{pad + i * step:.0f},{h - 20 - (h - 60) * r[k] / top:.0f}" for i, r in enumerate(rows))
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>')
    for j, (k, c) in enumerate(zip(keys, colors)):
        parts.append(f'<text x="{pad + 120 * j}" y="{h + 55}" fill="{c}">{k} (max {max(r[k] for r in rows)})</text>')
    return "".join(parts) + "</svg>"
