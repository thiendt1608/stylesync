import json
from pathlib import Path

import yaml

from stylesync.clean import drift
from stylesync.clean.burst import make_bursts
from stylesync.clean.filters import dedup_key, is_english, is_junk, is_noise, is_unaccented, strip_markers
from stylesync.clean.pii import Scrubber
from stylesync.clean.pipeline import read_jsonl, run, write_jsonl
from stylesync.ingest import discord, messenger

ME = "Chủ Nhân"


def mojibake(s: str) -> str:
    return s.encode("utf-8").decode("latin1")


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def make_messenger(root: Path) -> None:
    def thread(folder, name, others, msgs):
        write(root / folder / name / "message_1.json", {
            "participants": [{"name": mojibake(p)} for p in [ME, *others]],
            "title": mojibake(others[0]),
            "messages": [{"sender_name": mojibake(s), "timestamp_ms": t, "content": mojibake(c)} for s, t, c in msgs],
        })

    base = 1_750_000_000_000
    thread("inbox", "ban_1", ["Trần Văn Bảo"], [
        (ME, base, "ê Bảo ơi mai đi cafe kh"),
        (ME, base + 30_000, "t rảnh từ 9h nha"),
        ("Trần Văn Bảo", base + 60_000, "oke"),
        (ME, base + 400_000, "You sent an attachment."),
        (ME, base + 500_000, "😆😆"),
    ])
    thread("archived_threads", "nhom_2", ["Lê Thị Cúc", "Trần Văn Bảo"], [
        (ME, base + 1_000_000, "số t là 0912345678 nha mn, gọi đc á"),
        ("Lê Thị Cúc", base + 1_000_100, "ok"),
    ])


def make_discord(root: Path) -> None:
    write(root / "index.json", {"111": "Direct Message with friendly#0", "222": "general in Dev Team"})
    write(root / "c111" / "channel.json", {"id": "111", "type": "DM", "recipients": ["1", "2"]})
    write(root / "c111" / "messages.json", [
        {"ID": 1, "Timestamp": "2022-01-01 10:00:00", "Contents": "tin cũ quá rồi nè trước cutoff", "Attachments": ""},
        {"ID": 2, "Timestamp": "2024-05-01 10:00:00", "Contents": "", "Attachments": "https://cdn/x.png"},
        {"ID": 3, "Timestamp": "2024-05-01 10:05:00", "Contents": "https://github.com/x/y", "Attachments": ""},
        {"ID": 4, "Timestamp": "2024-05-02 10:00:00", "Contents": "<@123> xem PR này giúp t với nha", "Attachments": ""},
    ])
    write(root / "c222" / "channel.json", {"id": "222", "type": "GUILD_TEXT", "name": "general",
                                           "guild": {"id": "9", "name": "Dev Team"}})
    write(root / "c222" / "messages.json", [
        {"ID": 5, "Timestamp": "2024-06-01 10:00:00", "Contents": "deploy xong rồi mn check giúp nha", "Attachments": ""},
        {"ID": 6, "Timestamp": "2024-06-02 10:00:00", "Contents": "Deploy xong rồi   mn check giúp nha!!", "Attachments": ""},
        {"ID": 7, "Timestamp": "2024-06-03 10:00:00", "Contents": "thank you for your offer, I am busy", "Attachments": ""},
    ])


def test_messenger_fixes_mojibake_and_keeps_only_owner(tmp_path):
    make_messenger(tmp_path)
    rows, infos, names = messenger.load(tmp_path)
    assert {r["text"] for r in rows} >= {"ê Bảo ơi mai đi cafe kh", "t rảnh từ 9h nha"}
    assert "oke" not in {r["text"] for r in rows}
    assert {i["type"] for i in infos} == {"DM", "GROUP"}
    assert "Lê Thị Cúc" in names
    assert rows[0]["ts"].endswith("+00:00")


def test_messenger_owner_must_be_unambiguous():
    threads = [
        {"participants": {"A", "B"}, "messages": [{"sender_name": "B"}, {"sender_name": "B"}]},
        {"participants": {"A", "C"}, "messages": [{"sender_name": "A"}]},
    ]
    try:
        messenger.detect_me(threads)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_discord_load(tmp_path):
    make_discord(tmp_path)
    rows, infos, names = discord.load(tmp_path)
    assert len(rows) == 7
    assert names == {"friendly"}
    assert {i["title"] for i in infos} == {"Direct Message with friendly#0", "Dev Team / general"}
    assert rows[1]["has_attachment"] and rows[1]["ts"] == "2024-05-01T10:00:00+00:00"


def test_is_junk():
    for t in ["", "You sent an attachment.", "Bảo reacted 😆 to your message ", "https://x.y/z", "😆😆", ":))",
              "You started a video call."]:
        assert is_junk(t), t
    for t in ["kh", "ok nha", "xem https://x.y nè", "9h nha"]:
        assert not is_junk(t), t


def test_strip_markers():
    assert strip_markers("cần gì nữa (edited)") == "cần gì nữa"
    assert strip_markers("(edited) ở giữa vẫn giữ") == "(edited) ở giữa vẫn giữ"


def test_burst_merges_within_gap_and_thread():
    rows = [
        {"thread_id": "a", "ts": "2024-01-01T00:00:00+00:00", "text": "một", "source": "x"},
        {"thread_id": "a", "ts": "2024-01-01T00:01:00+00:00", "text": "hai", "source": "x"},
        {"thread_id": "a", "ts": "2024-01-01T00:10:00+00:00", "text": "ba", "source": "x"},
        {"thread_id": "b", "ts": "2024-01-01T00:01:30+00:00", "text": "khác", "source": "x"},
    ]
    out = make_bursts(rows, 120)
    assert [b["text"] for b in out] == ["một\nhai", "ba", "khác"]
    assert out[0]["n_msgs"] == 2


def test_scrubber():
    s = Scrubber({"Trần Văn Bảo", "Lê Thị Cúc", "friendly", "Hà Anh", "Harry KTX", "Google Cloud Club"})
    assert s("ê Bảo ơi, hỏi Trần Văn Bảo đi") == "ê <NAME> ơi, hỏi <NAME> đi"
    assert s("@Lê Thị Cúc gọi 0912 345 678 hoặc +84912345678") == "<NAME> gọi <PHONE> hoặc <PHONE>"
    assert s("mail a@b.com, stk 1903 1234 5678 9") == "mail <EMAIL>, stk <NUMBER>"
    assert s("email là abc123@gmail nha") == "email là <EMAIL> nha"
    assert s("key sk-abcdefghijklmnop ở https://x.y/z") == "key <SECRET> ở <URL>"
    assert s("<@123> <#456> <:kekw:789>") == "<NAME> <CHANNEL> :kekw:"
    assert s("anh em đi hà nội kh") == "anh em đi hà nội kh"  # common words stay in lowercase
    assert s("hỏi cúc đi") == "hỏi cúc đi"  # lowercase single parts are not matched
    assert s("Harry ở KTX hỏi Google Cloud") == "<NAME> ở KTX hỏi Google Cloud"  # all-caps part, org name
    assert s("t bảo rồi mà") == "t bảo rồi mà"  # "bảo" is also "to tell"
    assert s("hỏi friendly đi") == "hỏi <NAME> đi"


def test_scrubber_secrets():
    s = Scrubber(set())
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcDEF123_-x"
    assert s(f"token nè {jwt} xài đi") == "token nè <SECRET> xài đi"
    assert s("hash a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 nè") == "hash <SECRET> nè"
    assert s("pw: hello9ab1234 nha") == "pw: <SECRET> nha"
    assert s("mk là abc12") == "mk là <SECRET>"
    assert s("pass là xong") == "pass là xong"  # slang, no digit
    assert s("mã 123456 nè, giá 1.500.000") == "mã <NUMBER> nè, giá 1.500.000"
    assert s("supercalifragilisticexpialidocious") == "supercalifragilisticexpialidocious"  # no digits


def test_is_english():
    assert is_english("please try downloading each source individually")
    assert not is_english("deploy xong rồi mn check giúp nha")
    assert not is_english("ok bro")  # too short to judge
    assert is_english("i'm sorry but i don't know how to do as you said")
    assert not is_english("mai di choi kh, t ranh nha")  # unaccented Vietnamese is is_unaccented's job


def test_is_noise():
    assert is_noise("['cardealer'] = { / bossRank = 4, / locations = {} / },")
    assert is_noise("nghe / nghe / nghe / ro / nghe / nghe / nghe")
    assert is_noise("<NAME> <NAME> <NAME> <NAME> đâu")
    assert is_noise('ALTER TABLE "documents" ADD COLUMN "summary" TEXT;')
    assert not is_noise("<NAME> ơi mai đi cafe kh, t rảnh từ 9h nha")
    assert not is_noise("fix cái hàm get_user() rồi nha =))")


def test_is_unaccented():
    assert is_unaccented("da toi net / het van doi xiu nha dang mo may")
    assert not is_unaccented("co ai choi kh z")  # short replies (< 6 words) are fine
    assert not is_unaccented("kh biết / dc nghe full ma / adu vc z")


def test_dedup_key():
    assert dedup_key("Deploy xong rồi   mn!!") == dedup_key("deploy xong rồi mn")
    assert dedup_key("kkkkkk") == dedup_key("kkk")


def test_drift_metrics():
    rows = [{"ts": "2024-02-01T00:00:00+00:00", "text": "kh bt nữa"},
            {"ts": "2024-05-01T00:00:00+00:00", "text": "deploy server =))"}]
    q = drift.by_quarter(rows)
    assert [r["quarter"] for r in q] == ["2024Q1", "2024Q2"]
    assert q[0]["abbrev"] == round(2 / 3, 3) and q[0]["diacritics"] == round(1 / 3, 3)
    assert q[1]["emoticon"] == 1.0 and q[1]["codemix"] == 1.0
    assert drift.to_svg(q, "t", min_n=1).startswith("<svg")


def test_pipeline_end_to_end(tmp_path):
    make_messenger(tmp_path / "fb")
    make_discord(tmp_path / "dc")
    cfg = {
        "sources": {"messenger": {"root": "fb", "cutoff": None}, "discord": {"root": "dc", "cutoff": "2023-10-27",
                                                                         "full_weight_from": "2024-06-01",
                                                                         "old_weight": 0.5}},
        "burst_gap_s": 120, "min_len": 20, "max_len": 400, "threads": "threads.yaml",
        "out": {"interim": "i/m.jsonl", "bursts": "i/b.jsonl", "processed": "p/mine.jsonl",
                "stats": "i/stats.json", "drift_svg": "r/drift.svg"},
    }
    stats = run(cfg, tmp_path)
    out = read_jsonl(tmp_path / "p/mine.jsonl")
    texts = {o["text"] for o in out}
    assert "ê <NAME> ơi mai đi cafe kh\nt rảnh từ 9h nha" in texts
    assert "số t là <PHONE> nha mn, gọi đc á" in texts
    assert "<NAME> xem PR này giúp t với nha" in texts
    assert not any("cutoff" in t for t in texts)  # 2022 Discord message dropped
    assert sum("deploy" in t.lower() for t in texts) == 1  # near-duplicate dropped
    assert all("tone" not in o for o in out)  # one voice, no tone tag
    assert not any("offer" in t for t in texts)  # English burst dropped
    weights = {o["text"][:10]: o["weight"] for o in out}
    assert weights["<NAME> xem"] == 0.5 and weights["deploy xon"] == 1.0 and weights["ê <NAME> ơ"] == 1.0
    assert stats["steps"]["own_messages"] == {"messenger": 5, "discord": 7}

    # Excluding a thread by hand is respected on the next run.
    threads = yaml.safe_load((tmp_path / "threads.yaml").read_text(encoding="utf-8"))
    threads["d:222"]["include"] = False
    (tmp_path / "threads.yaml").write_text(yaml.safe_dump(threads, allow_unicode=True), encoding="utf-8")
    run(cfg, tmp_path)
    out = (tmp_path / "p/mine.jsonl").read_text(encoding="utf-8")
    assert "deploy" not in out.lower()


def test_jsonl_roundtrip_keeps_unicode_line_separators(tmp_path):
    rows = [{"text": "một\u2028hai\x85ba"}, {"text": "bốn"}]
    write_jsonl(tmp_path / "x.jsonl", rows)
    assert read_jsonl(tmp_path / "x.jsonl") == rows
