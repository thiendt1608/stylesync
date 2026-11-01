# stylesync

A small local model that rewrites a neutral Vietnamese draft in your own voice: your abbreviations, pronouns,
code-mixing and sentence length. Coding agents (Claude Code, omp) call it through an MCP server before they write
a commit message, a PR/issue comment or a chat message, so what they write sounds like you, not like an LLM.

- **Model:** a LoRA (r16) on Gemma 4 E2B, served as GGUF (base Q4_K_M + a 48MB LoRA) by `llama-server`. One
  server on `127.0.0.1` is shared by every agent session.
- **Safety net:** the model only changes the voice. The MCP server keeps the conventional-commit prefix, trailers
  and code spans out of the model's reach and checks the result. If anything fails it returns the neutral draft.
- **Data:** your own Messenger and Discord exports. A large model (OpenAI Batch here) back-translates each of your
  messages into a neutral version, which gives (neutral → yours) training pairs.

This repository holds the code, configs, tests and the aggregate reports of one run. It ships **no data and no
model weights**: an adapter trained on private chats memorises them, so the owner's adapter stays private, and
yours should too.

## Results of the reference run

From `reports/` (owner's data, aggregates only):

| | |
|---|---|
| Training data | 8,769 own messages → 17,038 neutral/own pairs; train 12,707 / val 1,649 / test 1,708 (time-based split, 24h embargo) |
| Training | 31 min on an RTX 3060 12GB, 1 epoch |
| Style on the test set | 100% classified as the owner's (base model zero-shot: 6%) |
| Meaning kept (LLM judge, 1–5) | 4.81, against 4.90 for GPT few-shot with 10 retrieved examples |
| Owner-written commit/comment set (40) | guard pass 100% with code-span masking, meaning 4.97 |
| Blind test (owner, 50 pairs) | 68% told apart (goal ≈ 50%: not reached) |
| Latency | 0.4–0.9s per call on an M4 (Metal); ~1.3s on Windows/Linux after the first call |
| API cost | ≈ $1.62 of OpenAI Batch for neutralizing and judging |

Full write-up (Vietnamese): `reports/project_report.md`. Plan and design notes: `IMPLEMENTATION_PLAN.md`.

## Layout

| Path | What |
|---|---|
| `src/stylesync/ingest/`, `clean/` | Phase 1: parse exports, burst merge, PII scrub, filters, style drift |
| `src/stylesync/neutralize/` | Phase 2: back-translation with the OpenAI Batch API, validator |
| `src/stylesync/dataset/` | Phase 3: time-based split, TRL prompt-completion files |
| `src/stylesync/train/` | Phase 4: LoRA SFT with Unsloth + TRL (CUDA, Linux/WSL2) |
| `src/stylesync/eval/` | Phase 5: style classifier, stylometry, LLM judge, gate |
| `src/stylesync/export/` | Phase 6: LoRA → GGUF, parity against PyTorch, install |
| `src/stylesync/mcp_server.py`, `llama.py` | Phase 7: MCP tools `rewrite_style` and `health`, shared llama-server backend |
| `configs/` | One YAML per phase; `threads.example.yaml` shows the shape of the generated thread list |
| `integrations/omp/` | omp skill and `AGENTS.md` rule |
| `spike/phase0/` | Model/runtime benchmark with 40 synthetic sentences |
| `reports/` | Aggregate reports of the reference run |

## Train your own voice

Only train on messages you wrote, keep `data/`, `runs/` and `models/` private (they are gitignored), and do not
publish the adapter. Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

1. **Exports.** Put your Facebook export (`your_facebook_activity/`, JSON) and your Discord data package
   (`package/`) in the repo root, or edit the paths and cutoffs in `configs/filter.yaml`.
2. **Clean (Phase 1).** `uv run stylesync-data` writes `data/processed/mine.jsonl`, `configs/threads.yaml` (set
   `include: false` to drop a thread, then run again) and `reports/data.md`.
3. **Neutralize (Phase 2).** Needs `OPENAI_API_KEY` in the environment; only PII-scrubbed text is sent, with
   `store: false`. Try `uv run stylesync-neutralize run --pilot` and `review --n 100` first, then
   `uv run stylesync-neutralize run`. Batches can take hours.
4. **Dataset (Phase 3).** `uv run stylesync-dataset`.
5. **Train (Phase 4), on a CUDA GPU under Linux/WSL2 (8–12GB VRAM).** `uv sync --group train`, then
   `uv run --group train stylesync-train --exp gemma4-e2b-r16-e1`. Experiments live in `configs/train.yaml`;
   add `--max-steps 20 --limit 400` for a smoke test and `--resume` to continue from a checkpoint.
6. **Eval (Phase 5, optional).** `uv run stylesync-eval gen --system <name>` (GPU), then `fewshot`, `judge` and
   `report` (OpenAI Batch). Settings in `configs/eval.yaml`.
7. **Export (Phase 6).** Put the base GGUF (`gemma-4-E2B-it-Q4_K_M.gguf`, Unsloth's Q4_K_M of
   `google/gemma-4-E2B-it` in the reference run) in `models/gguf/`, copy `runs/<exp>/adapter` to this machine,
   and set up the converter that `configs/export.yaml` points at: a llama.cpp checkout in `tools/llama.cpp` and a
   venv in `tools/venv-convert` with `requirements/requirements-convert_lora_to_gguf.txt` and
   `transformers>=5.17`. Then `uv run stylesync-export convert` and, optionally,
   `parity` and `report`.

Tests need no model and no data: `uv run pytest -q`.

## Use with Claude Code

Needs `llama-server` (llama.cpp) on `PATH` and the GGUF build from step 7.

```bash
uv run stylesync-export install
claude mcp add --scope user stylesync -- uv run --directory /path/to/fine-tune-chat stylesync-mcp
```

If Claude Code runs from an app whose `PATH` lacks Homebrew, give absolute paths: `/opt/homebrew/bin/uv` as the
command and `-e STYLESYNC_LLAMA_SERVER=/opt/homebrew/bin/llama-server` (after the server name).

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

Change the pronouns in the rule to the ones you use.

## Use with omp

omp (oh-my-pi) reads its own config, not Claude Code's. Add the server to `~/.omp/agent/mcp.json` under
`mcpServers` (the tool is then `mcp__stylesync_rewrite_style`, called through `xd://`):

```json
"stylesync": {
  "type": "stdio",
  "command": "/opt/homebrew/bin/uv",
  "args": ["run", "--directory", "/path/to/fine-tune-chat", "stylesync-mcp"],
  "env": {"STYLESYNC_LLAMA_SERVER": "/opt/homebrew/bin/llama-server"}
}
```

Then link the skill and append the rule to the user-level context file:

```bash
ln -s "$PWD/integrations/omp/stylesync" ~/.agents/skills/stylesync
cat integrations/omp/AGENTS.md >> ~/.omp/agent/AGENTS.md
```

Stop the backend: `pkill -f "llama-server.*stylesync-lora"` (Windows: end `llama-server.exe` in Task Manager).

## Limits

- Vietnamese only; one voice per adapter (no tone switch).
- The blind test did not reach chance level, and the judge shares a model with the strongest baseline.
- CI (`.github/workflows/ci.yml`) runs the tests on Linux, macOS and Windows; the model itself was run by hand on
  macOS (Metal), Windows and WSL2.

## License

MIT, see `LICENSE`. The license covers this code only. Base models keep their own licenses (Gemma 4: Apache 2.0),
and a model trained on chat exports carries the privacy of everyone in those chats.
