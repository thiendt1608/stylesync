## stylesync
- Trước `git commit`, `gh pr comment`, `gh issue comment` (và `gh pr create`/`gh pr review` có body) hoặc gửi tin
  thay owner: mở skill `stylesync`, viết bản trung tính tiếng Việt (xưng anh/em, kh dùng tôi/bạn; code, path,
  identifier để trong backtick), gọi `mcp__stylesync_rewrite_style` (task: commit | comment | message) và dùng
  nguyên văn `text` trả về.
- `guard_passed: false` → dùng `text` (bản trung tính) và báo owner lý do trong `warnings`.
- Owner chê bản viết lại ("viết lại", "kh giống", "bỏ chữ X"…) hoặc tự gõ bản khác: sửa tay theo ý owner
  (kh gọi lại `rewrite_style`, ra y chang), dùng bản mới, rồi gọi ngay `mcp__stylesync_rewrite_feedback` (`id`
  của lần rewrite, `final_text` = nguyên văn bản mới đầy đủ, `by`: `owner` nếu owner gõ nguyên câu, `agent` nếu
  sửa theo lời owner). Mỗi lần sửa tiếp lại gọi với bản mới nhất. Owner nói ok luôn → gọi với `final_text` rỗng;
  kh nói gì thì thôi.
- Tin gửi người thật: đưa owner xem trước, kh tự gửi.
