import pytest

from stylesync import mcp_server
from stylesync.mcp_server import guard, mask, split_commit, unmask


def test_split_commit_keeps_prefix_and_trailers():
    msg = "fix(auth): Sửa lỗi null trong `get_user`\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
    prefix, body, trailers = split_commit(msg)
    assert prefix == "fix(auth): "
    assert body == "Sửa lỗi null trong `get_user`"
    assert trailers == "Co-Authored-By: Claude <noreply@anthropic.com>"


def test_split_commit_without_prefix():
    assert split_commit("Cập nhật README") == ("", "Cập nhật README", "")


def test_split_commit_subject_is_never_a_trailer():
    assert split_commit("refactor: Tách logic ra module riêng.") == ("refactor: ", "Tách logic ra module riêng.", "")
    assert split_commit("docs: update README\nRefs: #12") == ("docs: ", "update README\nRefs: #12", "")
    msg = "revert: Hoàn tác `a1b2c3d`\n\nLý do: build hỏng\n\nSigned-off-by: Dev <dev@example.com>"
    assert split_commit(msg) == ("revert: ", "Hoàn tác `a1b2c3d`\n\nLý do: build hỏng", "Signed-off-by: Dev <dev@example.com>")


def test_guard_rejects_lost_code_span():
    assert guard("sửa `parse_date` sai múi giờ", "sửa hàm parse date nha") != []


def test_guard_rejects_placeholder_and_empty():
    assert "placeholder leaked" in guard("hello bạn", "hi <NAME> nha")
    assert "empty output" in guard("abc", "  ")


def test_guard_rejects_added_code_span():
    assert "added code span `commit`" in guard("Sửa lỗi null trong `get_user`.", "`commit`: sửa null trong `get_user`")


def test_guard_allows_quoting_existing_identifier():
    assert guard("Hàm parse_date đang trả về sai múi giờ.", "hàm `parse_date` trả sai múi giờ nha") == []


def test_guard_rejects_switch_to_english():
    assert "Vietnamese lost (no diacritics in output)" in guard(
        "Sửa lỗi đăng nhập do thiếu kiểm tra null.", "fix login bug, missing null check")


def test_guard_allows_english_source():
    assert guard("Update README", "update README nha") == []


def test_guard_accepts_good_rewrite():
    assert guard("Đã sửa `get_user` rồi.", "fix `get_user` r nha") == []


def test_mask_round_trip_inline_and_fenced():
    src = "Sửa `get_user` và `parse_date`:\n```py\nx = `y`\n```\nxong."
    masked, spans = mask(src)
    assert masked == "Sửa <CODE1> và <CODE2>:\n<CODE3>\nxong." and len(spans) == 3
    assert unmask(masked, spans) == src
    assert unmask("<CODE2> rồi <CODE1>", spans) == "`parse_date` rồi `get_user`"


def test_mask_leaves_text_with_a_slot_token_alone():
    assert mask("giữ `<CODE1>` nha") == ("giữ `<CODE1>` nha", [])


def test_made_up_slot_is_rejected_by_guard():
    out = unmask("sửa <CODE1> với <CODE9>", ["`a`"])
    assert out == "sửa `a` với <CODE9>" and "placeholder leaked" in guard("Sửa `a`.", out)


def test_rewrite_style_sends_masked_text_and_restores_backticks(monkeypatch):
    seen = []

    def fake(text):
        seen.append(text)
        return "fix null trong <CODE1> rồi nha"  # what an adapter does with a placeholder: copy it

    monkeypatch.setattr(mcp_server, "_generate", fake)
    res = mcp_server.rewrite_style("fix(auth): Sửa lỗi null trong `get_user`.\n\nRefs: #12", "commit")
    assert seen == ["Sửa lỗi null trong <CODE1>."]
    assert res == {"text": "fix(auth): fix null trong `get_user` rồi nha\n\nRefs: #12", "guard_passed": True, "warnings": []}


def test_guard_rejects_new_contact_details_only():
    assert "new contact detail dev@example.com" in guard("Gửi file cho anh nhé.", "gửi file cho a nha dev@example.com")
    assert any("https://x.example/a" in p for p in guard("Xem link nhé.", "xem https://x.example/a nha"))
    assert any(p.startswith("new contact detail") for p in guard("Gọi anh nhé.", "gọi a 0912345678 nha"))
    assert guard("Gửi về dev@example.com nhé.", "gửi về dev@example.com nha") == []
    # the draft's URL ends with a comma, the rewrite drops it: still the same URL
    assert guard("Docs ở https://example.com/docs/cart, em đọc nhé.", "docs ở https://example.com/docs/cart e đọc nha") == []


def test_rewrite_style_falls_back_when_backend_is_down(monkeypatch):
    def down(text):
        raise mcp_server.llama.BackendError("model file missing: x.gguf")

    monkeypatch.setattr(mcp_server, "_generate", down)
    res = mcp_server.rewrite_style("Đã sửa lỗi rồi.", "message")
    assert res == {"text": "Đã sửa lỗi rồi.", "guard_passed": False,
                   "warnings": ["local model unavailable: model file missing: x.gguf"]}


def test_rewrite_style_uses_the_training_prompt(monkeypatch):
    seen = {}

    def chat(client, system, text, max_tokens=256):
        seen.update(system=system, text=text)
        return "fix r nha", 0.4

    monkeypatch.setattr(mcp_server.llama, "ensure", lambda client, s: False)
    monkeypatch.setattr(mcp_server.llama, "chat", chat)
    res = mcp_server.rewrite_style("Tôi đã sửa xong.", "comment")
    assert seen == {"system": mcp_server.SYSTEM, "text": "Tôi đã sửa xong."} and res["guard_passed"]
    assert mcp_server.STATE["last_latency_s"] == 0.4
    from stylesync.dataset.build import SYSTEM
    assert mcp_server.SYSTEM == SYSTEM


def test_tools_over_an_mcp_client(monkeypatch):
    import asyncio
    import json

    from mcp.client import Client

    monkeypatch.setattr(mcp_server, "_generate", lambda text: "sửa null trong <CODE1> r nha")

    async def run():
        async with Client(mcp_server.mcp) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            res = await client.call_tool("rewrite_style", {"text": "fix: Sửa lỗi null trong `get_user`.", "task": "commit"})
            return names, json.loads(res.content[0].text)

    names, res = asyncio.run(run())
    assert {"rewrite_style", "health"} <= names
    assert res == {"text": "fix: sửa null trong `get_user` r nha", "guard_passed": True, "warnings": []}


def test_rewrite_style_sends_one_paragraph_at_a_time(monkeypatch):
    seen = []

    def fake(text):
        seen.append(text)
        return {"Thêm <CODE1> cho <CODE2>.": "thêm <CODE1>\ncho <CODE2> nha",
                "Hàm <CODE3> giờ nhận thêm <CODE4>.": "hàm <CODE3> giờ nhận thêm <CODE4> r"}[text]

    monkeypatch.setattr(mcp_server, "_generate", fake)
    msg = ("feat(cart): Thêm `discount` cho `total`.\r\n\r\nHàm `total` giờ nhận thêm `discount`.\r\n\r\n"
           "```py\ntotal(items, 0.1)\n```\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    res = mcp_server.rewrite_style(msg, "commit")
    assert seen == ["Thêm <CODE1> cho <CODE2>.", "Hàm <CODE3> giờ nhận thêm <CODE4>."]  # the code block is not sent
    assert res["guard_passed"] and res["text"] == (
        "feat(cart): thêm `discount` cho `total` nha\n\n"  # the subject stays one line, the blank line stays
        "hàm `total` giờ nhận thêm `discount` r\n\n```py\ntotal(items, 0.1)\n```\n\n"
        "Co-Authored-By: Claude <noreply@anthropic.com>")


def test_rewrite_style_leaves_empty_and_non_vietnamese_drafts(monkeypatch):
    monkeypatch.setattr(mcp_server, "_generate", lambda text: pytest.fail("the model must not be called"))
    assert mcp_server.rewrite_style("  ", "message") == {
        "text": "  ", "guard_passed": False, "warnings": ["empty draft: nothing to rewrite"]}
    assert mcp_server.rewrite_style("fix(ci): ", "commit")["warnings"] == ["empty draft: nothing to rewrite"]
    assert mcp_server.rewrite_style("`make test`", "comment") == {"text": "`make test`", "guard_passed": True, "warnings": []}
    en = "Please review the pull request when you have time."
    assert mcp_server.rewrite_style(en, "comment") == {
        "text": en, "guard_passed": False, "warnings": ["draft is not Vietnamese: left unchanged"]}


def test_rewrite_style_rejects_a_cut_off_rewrite(monkeypatch):
    def cut(text):
        raise mcp_server.llama.Truncated("output hit max_tokens=512")

    monkeypatch.setattr(mcp_server, "_generate", cut)
    assert mcp_server.rewrite_style("Đã sửa lỗi rồi.", "message")["warnings"] == [
        "rewrite incomplete: output hit max_tokens=512"]


def test_guard_rejects_a_rewrite_that_drops_most_of_a_long_draft():
    sentence = "Anh đã kiểm tra lại luồng thanh toán và thấy phần giảm giá bị làm tròn hai lần nên tổng tiền lệch."
    draft = " ".join([sentence] * 4)
    kept_one = "a check lại luồng thanh toán thấy phần giảm giá bị làm tròn 2 lần nên tổng tiền lệch"
    assert any(p.startswith("most of the draft is missing") for p in guard(draft, kept_one))
    assert not any(p.startswith("most of") for p in guard(draft, "\n".join([kept_one] * 4)))
    short = "Anh đã sửa xong lỗi làm tròn rồi, em xem lại giúp anh nhé."
    assert not any(p.startswith("most of") for p in guard(short, "ok nha"))  # under 20 words: length rule only


def test_coverage_counts_repeats():
    from stylesync.mcp_server import coverage

    assert coverage("abc abc", "abc abc") == 1.0 and coverage("", "x") == 1.0
    assert coverage("abcabc", "abc") < coverage("abcabc", "abcabc")
