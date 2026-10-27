"""Shared prompt, test set loader and output checks for the Phase 0 benches (stdlib only)."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SYSTEM = (
    "Bạn là bộ viết lại văn phong. Viết lại đoạn văn của người dùng theo giọng nhắn tin thân mật "
    "kiểu Gen Z Việt Nam: câu ngắn, dùng viết tắt phổ biến (kh, đc, z, nma, bt, ng, mn, vd), "
    "xưng t/m khi phù hợp, thuật ngữ kỹ thuật giữ tiếng Anh. Giữ NGUYÊN ý, không thêm, không bớt. "
    "Giữ nguyên mọi nội dung trong dấu backtick. Nếu đầu vào bắt đầu bằng prefix conventional commit "
    "thì giữ nguyên prefix đó; không tự thêm prefix khi đầu vào không có. "
    "Chỉ trả về bản viết lại, không giải thích.\nLoại văn bản: {task}"
)
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
VN = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
PREFIX = re.compile(r"^[a-z]+(\([^)]*\))?:")
ABBR = {"kh", "k", "ko", "đc", "dc", "z", "v", "bt", "nma", "ng", "mn", "vd", "thoi", "dk", "t", "m", "ok", "r"}


def load_cases() -> list[dict]:
    return [json.loads(line) for line in (HERE / "testset.jsonl").read_text().splitlines() if line]


def messages(case: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM.format(task=case["task"])},
        {"role": "user", "content": case["text"]},
    ]


def check(src: str, out: str) -> dict:
    words = re.findall(r"\w+", out.lower())
    prefix = PREFIX.match(src)
    return {
        "empty": not out.strip(),
        "cjk": bool(CJK.search(out)),
        "think_leak": "<think" in out or "thought" in out[:20].lower(),
        "has_diacritics": bool(VN.search(out.lower())),
        "backticks_kept": all(t in out for t in re.findall(r"`[^`]+`", src)),
        "prefix_kept": (out.startswith(prefix.group(0)) if prefix else None),
        "prefix_leak": bool(not prefix and PREFIX.match(out.lstrip("`* "))),
        "abbr_count": sum(w in ABBR for w in words),
        "len_ratio": round(len(out) / max(len(src), 1), 2),
    }


def summarize(name: str, device: str, rows: list[dict], load_s: float, peak_rss: int) -> dict:
    walls = sorted(r["wall_s"] for r in rows)
    tg = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    return {
        "model": name, "device": device, "load_s": round(load_s, 2),
        "peak_rss_mb": round(peak_rss / 1e6),
        "wall_mean_s": round(sum(walls) / len(walls), 3), "wall_p90_s": walls[int(len(walls) * 0.9)],
        "decode_tok_s": round(sum(tg) / len(tg), 1),
        "cjk": sum(r["cjk"] for r in rows), "empty": sum(r["empty"] for r in rows),
        "think_leak": sum(r["think_leak"] for r in rows),
        "no_diacritics": sum(not r["has_diacritics"] for r in rows),
        "backticks_lost": sum(not r["backticks_kept"] for r in rows),
        "prefix_lost": sum(r["prefix_kept"] is False for r in rows),
        "prefix_leak": sum(r["prefix_leak"] for r in rows),
        "abbr_mean": round(sum(r["abbr_count"] for r in rows) / len(rows), 2),
    }


def save(name: str, device: str, rows: list[dict], summary: dict) -> None:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}_{device}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    with (out_dir / "summary.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
