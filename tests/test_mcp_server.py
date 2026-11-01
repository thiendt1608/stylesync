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

    def fake(text, task):
        seen.append(text)
        return "fix null trong <CODE1> rồi nha"  # what an adapter does with a placeholder: copy it

    monkeypatch.setattr(mcp_server, "_generate", fake)
    res = mcp_server.rewrite_style("fix(auth): Sửa lỗi null trong `get_user`.\n\nRefs: #12", "commit")
    assert seen == ["Sửa lỗi null trong <CODE1>."]
    assert res == {"text": "fix(auth): fix null trong `get_user` rồi nha\n\nRefs: #12", "guard_passed": True, "warnings": []}
