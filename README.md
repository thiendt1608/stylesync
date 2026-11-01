# stylesync

Private style rewriter: a LoRA on Gemma 4 E2B turns a neutral Vietnamese draft into the owner's own voice.
Claude Code calls it through the `stylesync` MCP server before writing a commit message, a PR/issue comment
or a chat message. Everything runs locally; the training data and the adapter never leave this machine.
Plan and results: `IMPLEMENTATION_PLAN.md`, `reports/`.

## Use with Claude Code

Needs `llama-server` (llama.cpp) on `PATH` and the GGUF build from Phase 6 (`models/gguf/`, not in git).

```bash
uv run stylesync-export install
claude mcp add --scope user stylesync -- uv run --directory /path/to/fine-tune-chat stylesync-mcp
```

`install` copies the base and LoRA GGUF to `platformdirs.user_cache_dir("stylesync")` (macOS:
`~/Library/Caches/stylesync`). The first `rewrite_style` call starts one `llama-server` on `127.0.0.1:18080`,
detached, and later sessions reuse it; its log is `llama-server.log` next to the models. Overrides:
`STYLESYNC_LLAMA_URL`, `STYLESYNC_MODEL_DIR`, `STYLESYNC_BASE_GGUF`, `STYLESYNC_LORA_GGUF`, `STYLESYNC_LLAMA_SERVER`.

The tool description alone is not enough for Claude Code to call a deferred MCP tool (Phase 0), so add a rule to
`~/.claude/CLAUDE.md`:

```markdown
## stylesync
- Trước `git commit`, `gh pr comment`, `gh issue comment` hoặc gửi tin thay t: viết bản trung tính tiếng Việt
  (xưng anh/em như t, kh dùng tôi/bạn; code, path, identifier để trong backtick), gọi
  `mcp__stylesync__rewrite_style` (task: commit | comment | message) và dùng nguyên văn `text` trả về.
- `guard_passed: false` → dùng `text` (bản trung tính) và báo t lý do trong `warnings`.
- Tin gửi người thật: đưa t xem trước, kh tự gửi.
```

Stop the backend: `pkill -f "llama-server.*stylesync-lora"` (Windows: end `llama-server.exe` in Task Manager).
