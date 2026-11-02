import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

from stylesync import feedback, history, mcp_server
from stylesync.clean.pii import Scrubber


def _rewrite(monkeypatch, out, text="Đã sửa lỗi null trong `get_user`.", task="message"):
    monkeypatch.setattr(mcp_server, "_generate", lambda para: out)
    return mcp_server.rewrite_style(text, task)


def test_rewrite_style_logs_each_rewrite_and_returns_its_id(monkeypatch):
    res = _rewrite(monkeypatch, "sửa null trong <CODE1> r nha")
    assert res["guard_passed"] and res["id"]
    (rec,) = history.read(history.path())
    assert rec["type"] == "rewrite" and rec["id"] == res["id"] and rec["task"] == "message"
    assert rec["draft"] == "Đã sửa lỗi null trong `get_user`." and rec["output"] == "sửa null trong `get_user` r nha"


def test_history_file_is_owner_only(monkeypatch):
    _rewrite(monkeypatch, "sửa null trong <CODE1> r nha")
    if sys.platform != "win32":
        assert os.stat(history.path()).st_mode & 0o777 == 0o600


def test_fallbacks_are_logged_but_drafts_with_nothing_to_rewrite_are_not(monkeypatch):
    def down(para):
        raise mcp_server.llama.BackendError("down")

    monkeypatch.setattr(mcp_server, "_generate", down)
    assert mcp_server.rewrite_style("Đã sửa lỗi rồi.", "message")["id"]
    for text in ("", "`make test`", "Fixed the null check in the parser module."):
        assert mcp_server.rewrite_style(text, "message")["id"] is None
    assert len(history.read(history.path())) == 1


def test_history_off_logs_nothing(monkeypatch):
    monkeypatch.setenv("STYLESYNC_HISTORY", "off")
    assert _rewrite(monkeypatch, "sửa null trong <CODE1> r nha")["id"] is None
    res = mcp_server.rewrite_feedback("abc", "x")
    assert res["saved"] is False and "off" in res["warnings"][0]


def test_feedback_marks_corrected_accepted_and_unknown(monkeypatch):
    a = _rewrite(monkeypatch, "sửa null trong <CODE1> r nha")["id"]
    b = _rewrite(monkeypatch, "sửa null trong <CODE1> r nha")["id"]
    assert mcp_server.rewrite_feedback(a, "fix null ở `get_user` r", "owner") == {
        "saved": True, "status": "corrected", "warnings": []}
    assert mcp_server.rewrite_feedback(b)["status"] == "accepted"
    assert mcp_server.rewrite_feedback("nope", "x")["saved"] is False
    by_id = {e["id"]: e for e in history.entries(history.read(history.path()))}
    assert by_id[a]["by"] == "owner" and by_id[a]["final"] == "fix null ở `get_user` r"
    assert by_id[b]["final"] == by_id[b]["output"]


def test_status_silent_after_a_day_and_latest_feedback_wins():
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    recs = [{"type": "rewrite", "id": "a", "ts": t0.isoformat(), "task": "message", "draft": "d", "output": "o"},
            {"type": "rewrite", "id": "b", "ts": t0.isoformat(), "task": "message", "draft": "d", "output": "o"},
            {"type": "feedback", "id": "b", "final": "x", "by": "agent"},
            {"type": "feedback", "id": "b", "final": "o", "by": "agent"},
            {"type": "review", "id": "zzz", "action": "drop"}]  # unknown id: ignored
    at = t0 + timedelta(hours=23)
    assert {e["id"]: e["status"] for e in history.entries(recs, at)} == {"a": "pending", "b": "accepted"}
    assert history.entries(recs, t0 + history.SILENT_ACCEPT)[0]["status"] == "silent"
    recs.append({"type": "review", "id": "b", "action": "drop"})
    assert history.entries(recs, at)[1]["status"] == "dropped"


def test_commit_feedback_is_compared_without_trailers():
    out = "fix(api): sửa null r\n\nCo-Authored-By: X <x@example.com>"
    rec = {"type": "rewrite", "id": "a", "ts": "2026-11-01T00:00:00+00:00", "task": "commit", "draft": "d",
           "output": out}
    same = [rec, {"type": "feedback", "id": "a", "final": "fix(api): sửa null r", "by": "agent"}]
    other = [rec, {"type": "feedback", "id": "a", "final": "fix(api): fix null\n\nCo-Authored-By: X <x@example.com>",
                   "by": "agent"}]
    assert history.entries(same)[0]["status"] == "accepted"
    assert history.entries(other)[0]["status"] == "corrected"
    assert history.entries([{**rec, "task": "comment"}, same[1]])[0]["status"] == "corrected"


def test_read_skips_a_cut_off_line_and_keeps_line_separators(tmp_path):
    p = tmp_path / "h.jsonl"
    history.append({"type": "rewrite", "id": "a", "text": "dòng 1 dòng 2"}, p)
    with p.open("a", encoding="utf-8") as f:
        f.write('{"type": "rewr')
    assert history.read(p) == [{"type": "rewrite", "id": "a", "text": "dòng 1 dòng 2"}]


def _entry(**kw):
    e = {"id": "abc", "ts": "2026-11-01T00:00:00+00:00", "task": "message", "by": "owner", "status": "corrected",
         "review": None, "final": None}
    return {**e, **kw}


def test_pairs_mask_code_and_use_commit_body_per_paragraph():
    e = _entry(task="commit",
               draft="fix(api): Sửa lỗi null trong `get_user`.\n\nĐã thêm test.\n\nCo-Authored-By: X <x@example.com>",
               output="fix(api): sửa null trong `get_user` nha\n\nthêm test r\n\nCo-Authored-By: X <x@example.com>",
               final="fix(api): fix null ở `get_user`\n\nthêm test luôn\n\nCo-Authored-By: X <x@example.com>")
    rows = feedback.pairs(e, Scrubber(set()))
    assert [(r["input"], r["output"], r["rejected"]) for r in rows] == [
        ("Sửa lỗi null trong <CODE1>.", "fix null ở <CODE1>", "sửa null trong <CODE1> nha"),
        ("Đã thêm test.", "thêm test luôn", "thêm test r")]
    assert [r["id"] for r in rows] == ["fbabc-1", "fbabc-2"] and rows[0]["source"] == "feedback"
    assert rows[0]["variant"] == "owner"


def test_pairs_keep_whole_text_when_paragraphs_differ_and_scrub_contacts():
    e = _entry(draft="Gửi anh file.\n\nEmail em là a.b@example.com nhé.", output="gửi a file nha\n\nmail e a.b@example.com",
               final="gửi file nha, mail a.b@example.com")
    (row,) = feedback.pairs(e, Scrubber(set()))
    assert row["input"] == "Gửi anh file.\n\nEmail em là <EMAIL> nhé." and row["output"] == "gửi file nha, mail <EMAIL>"


def test_export_takes_corrections_and_optionally_used_rewrites(tmp_path):
    entries = [_entry(id="a", draft="Đã xong.", output="xong r", final="xong r nha"),
               _entry(id="b", draft="Đã xong.", output="xong r", status="silent", by=None),
               _entry(id="c", draft="Đã xong.", output="Đã xong.", status="silent", by=None),  # fallback, used as is
               _entry(id="d", draft="Đã xong.", output="xong r", final="ok", status="dropped")]
    out = tmp_path / "fb.jsonl"
    assert feedback.export(entries, out, accepted=False)["pairs"] == 1
    assert feedback.export(entries, out, accepted=True) == {"pairs": 2, "from": {"owner": 1, "silent": 1},
                                                            "out": str(out)}
    assert [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()] == ["fba-1", "fbb-1"]


def test_stats_and_review(monkeypatch, tmp_path):
    p = tmp_path / "h.jsonl"
    for rid in ("a", "b"):
        history.append({"type": "rewrite", "id": rid, "ts": history.now(), "task": "commit", "draft": "Đã xong.",
                        "output": "xong r"}, p)
        history.append({"type": "feedback", "id": rid, "final": "xong r nha", "by": "owner"}, p)
    entries = history.entries(history.read(p))
    assert "toward retraining: 2/500" in feedback.stats(entries)
    answers = iter(["x", "d", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert feedback.review(entries, p, everything=False) == 1
    entries = history.entries(history.read(p))
    assert [e["status"] for e in entries] == ["dropped", "corrected"]
    assert "not reviewed yet: 1" in feedback.stats(entries)


@pytest.mark.parametrize("by", ["owner", "agent", "someone"])
def test_feedback_author_is_owner_or_agent(monkeypatch, by):
    rid = _rewrite(monkeypatch, "sửa null trong <CODE1> r nha")["id"]
    mcp_server.rewrite_feedback(rid, "khác", by)
    (e,) = history.entries(history.read(history.path()))
    assert e["by"] == (by if by != "someone" else "agent")
