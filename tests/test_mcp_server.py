from stylesync.mcp_server import guard, split_commit


def test_split_commit_keeps_prefix_and_trailers():
    msg = "fix(auth): Sửa lỗi null trong `get_user`\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
    prefix, body, trailers = split_commit(msg)
    assert prefix == "fix(auth): "
    assert body == "Sửa lỗi null trong `get_user`"
    assert trailers == "Co-Authored-By: Claude <noreply@anthropic.com>"


def test_split_commit_without_prefix():
    assert split_commit("Cập nhật README") == ("", "Cập nhật README", "")


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
