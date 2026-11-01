## stylesync
- Trước `git commit`, `gh pr comment`, `gh issue comment` (và `gh pr create`/`gh pr review` có body) hoặc gửi tin
  thay owner: mở skill `stylesync`, viết bản trung tính tiếng Việt (xưng anh/em, kh dùng tôi/bạn; code, path,
  identifier để trong backtick), gọi `mcp__stylesync_rewrite_style` (task: commit | comment | message) và dùng
  nguyên văn `text` trả về.
- `guard_passed: false` → dùng `text` (bản trung tính) và báo owner lý do trong `warnings`.
- Tin gửi người thật: đưa owner xem trước, kh tự gửi.
