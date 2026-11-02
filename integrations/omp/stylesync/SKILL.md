---
name: stylesync
description: Viết commit message, PR/issue comment hoặc tin nhắn thay owner bằng giọng của owner qua MCP stylesync (mcp__stylesync_rewrite_style). Dùng TRƯỚC mọi `git commit`, `gh pr comment`, `gh issue comment`, `gh pr create`/`gh pr review` có body, hoặc khi soạn tin gửi người khác thay owner.
---

# stylesync — viết đúng giọng owner

Model local (LoRA Gemma 4 E2B qua llama-server) đổi bản nháp trung tính tiếng Việt thành giọng owner.
Chạy hoàn toàn trên máy, kh gửi text ra ngoài.

## Quy trình

1. Viết bản trung tính **tiếng Việt**: xưng anh/em như owner (kh dùng tôi/bạn/mình); code, path, identifier,
   lệnh để trong backtick hoặc code block. Commit giữ prefix conventional (`fix(x): ...`) và trailer
   (`Co-Authored-By: ...`) ở đoạn cuối, cách body 1 dòng trống. Đưa cả message (subject + body + trailer)
   trong 1 lần gọi: tool tự rewrite từng đoạn và giữ code block/trailer.
2. Gọi `mcp__stylesync_rewrite_style` với `text` = bản nháp, `task` = `commit` | `comment` | `message`.
3. Dùng **nguyên văn** `text` trả về, kh sửa, kh dịch, kh thêm câu.
   - `guard_passed: false` → `text` lúc này là bản trung tính: dùng nó và báo owner lý do trong `warnings`.
   - Bản nháp tiếng Anh sẽ được trả nguyên kèm warning: muốn đúng giọng thì viết tiếng Việt.
4. Ghi ra git/gh mà giữ đúng xuống dòng: `git commit -F <file>` hoặc heredoc `-F -`,
   `gh pr comment --body-file <file>`; kh nhét nhiều đoạn vào `-m "..."`.
5. Tin gửi người thật (chat, email, DM): đưa owner xem trước, kh tự gửi.
6. Giữ `id` tool trả về. Owner chê bản viết lại hoặc tự gõ bản khác: sửa tay theo ý owner (gọi lại
   `rewrite_style` ra y chang vì decode greedy), dùng bản mới (amend commit, ghi lại file…), rồi gọi **ngay**
   `mcp__stylesync_rewrite_feedback` với `id`, `final_text` = nguyên văn bản mới đầy đủ (cả commit message), `by` =
   `owner` (owner gõ nguyên câu) hoặc `agent` (sửa theo lời owner). Mỗi lần sửa tiếp lại gọi với bản mới nhất (bản
   cuối đè). Owner nói ok luôn → `final_text` rỗng; owner kh nói gì thì kh gọi. Bản này là data để train tiếp, kh
   được tự bịa.

## Lỗi thường gặp

- `local model unavailable: ...` → backend chưa sẵn sàng (thiếu model, llama-server lỗi). Dùng `text` trung tính,
  báo owner; `mcp__stylesync_health` cho biết model file, server, adapter.
- Lần gọi đầu sau khi máy khởi động mất vài giây (spawn llama-server), các lần sau ~1s.
