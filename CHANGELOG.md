# Changelog

## [Unreleased]

### Added
- Implementation plan (`IMPLEMENTATION_PLAN.md`): Discord data window is 36 months (from 2023-10-27); Messenger keeps the full export.
- Phase 0 spike (`spike/phase0/`): GGUF (llama-server, Metal/CPU) and ONNX Runtime GenAI benches over a 40-sentence synthetic Vietnamese test set.
- `stylesync` MCP server (`src/stylesync/mcp_server.py`): tool `rewrite_style` backed by a local llama-server. It keeps the conventional-commit prefix and trailers, and a guard falls back to the neutral text when output is empty, loses or adds code spans, switches away from Vietnamese, leaks placeholders, or changes length too much.
- Phase 0 report (`reports/phase0.md`): runtime is GGUF + llama-server; Gemma 4 E2B and Qwen3.5-2B go to the LoRA pilot.
- Phase 0 agent test: Claude Code calls `rewrite_style` only when a CLAUDE.md-style rule asks for it, because the docstring of a deferred MCP tool is not enough.
- Phase 1 data pipeline (`uv run stylesync-data`, `src/stylesync/{ingest,clean}/`):
  - ingests the Messenger and Discord exports;
  - drops junk, code, spam, English and no-diacritic bursts;
  - merges bursts, scrubs PII (names, phones, emails, URLs, tokens, credentials) into placeholders, and dedups;
  - applies per-source cutoffs and sample weights;
  - writes `data/processed/mine.jsonl` (8,769 samples).
- Phase 1 report (`reports/data.md`, `reports/drift.svg`): counts per step, per-quarter style drift, and three rounds of manual review (final error rate 0.5%).
- Phase 2 neutralize (`uv run stylesync-neutralize`, `src/stylesync/neutralize/`):
  - back-translates each burst into a formal and a concise neutral variant with the OpenAI Batch API (gpt-6-luna);
  - caches results by model, prompt version and text, and resumes after a restart or an expired batch;
  - deletes batch files on OpenAI after collecting;
  - validates pairs, writes `data/processed/pairs.jsonl`, and computes real cost from token usage;
  - writes a 100-pair spot-check sheet.
- Phase 3 dataset build (`uv run stylesync-dataset`, `src/stylesync/dataset/`):
  - splits pairs by time per source (newest 10% test, next 10% val) with a 24h embargo at each boundary;
  - applies sample weights as kept variants;
  - writes TRL prompt-completion files to `data/processed/splits/` with a leak check.
- Phase 3 report (`reports/dataset.md`): train 12,707 / val 1,649 / test 1,708 examples; the longest is about 390 tokens.
- Phase 4 training (`uv run stylesync-train --exp <name>`, `src/stylesync/train/`, `configs/train.yaml`):
  - LoRA bf16 SFT with Unsloth + TRL on a CUDA GPU, experiments defined in the config;
  - renders each example with the base model's chat template (non-thinking, as the MCP server sends it) and trains on the completion only;
  - evaluates on val, keeps the best checkpoint, stops early, resumes with `--resume`;
  - loads the exact model repo from the config, and turns off Unsloth's torch.compile per model (`no_compile`, needed for Gemma 4 E2B);
  - writes `runs/<exp>/adapter/`, `log.jsonl` and `metrics.json`.
- Phase 5 eval (`uv run stylesync-eval gen|fewshot|judge|report`, `src/stylesync/eval/`, `configs/eval.yaml`):
  - `gen` rewrites the test set with each base model (Baseline A, descriptive prompt) and each LoRA adapter, greedy, resumable (CUDA GPU);
  - `fewshot` runs Baseline B: gpt-6-luna with the 10 most similar train pairs (char n-gram TF-IDF, one per burst), through the Phase 2 Batch runner;
  - `judge` scores meaning (1–5, added / lost flags) of every system's rewrites with gpt-6-luna, ignoring style;
  - `report` writes `reports/eval.md` with aggregates only: style classifier rate, stylometry, hard-token / placeholder / guard pass rates, judge scores, the plan's gate (the "added" flag rate is compared with the real messages' rate plus 3 points, since the judge flags 20% of them) and OpenAI cost. Good/bad examples and a 50-pair blind sheet stay private in `data/eval/`.
- Phase 5 owner set (`stylesync-eval <cmd> --set owner`, `src/stylesync/eval/owner_set.py`, `reports/eval_owner.md`): the 40 owner-written commit/comment items run through the same systems, metrics and gate; only a commit's body is rewritten and compared, as in `rewrite_style`. The private sheet stays in `data/eval/`.
- Phase 6 export (`uv run stylesync-export convert|parity|report`, `src/stylesync/export/`, `configs/export.yaml`):
  - `convert` turns the chosen LoRA adapter into a LoRA GGUF with llama.cpp's converter, to load as `llama-server --lora` on top of the base Q4_K_M GGUF;
  - `parity` starts its own llama-server (Metal or CPU), rewrites a fixed sample of the test set and the owner set greedily with the training prompt, and checks each request has exactly one BOS, as in training;
  - `report` writes `reports/export.md`: agreement with the Phase 5 PyTorch outputs (exact, chrF), guard and hard-token rates, BOS, runaway rate and latency, aggregates only.
- Phase 7 local backend (`src/stylesync/llama.py`): `rewrite_style` starts one shared `llama-server` (base Q4_K_M GGUF + stylesync LoRA GGUF) on `127.0.0.1:18080` when nothing is listening, detached so later Claude Code sessions reuse it, and refuses a server on that port without the stylesync adapter. Model files live in `platformdirs.user_cache_dir("stylesync")`; `uv run stylesync-export install` copies them there. `STYLESYNC_*` environment variables override the URL and paths.
- `README.md`: Claude Code setup (`claude mcp add`) and the `~/.claude/CLAUDE.md` rule that makes the agent call `rewrite_style`.
- CI (`.github/workflows/ci.yml`): tests on Ubuntu, macOS and Windows.
- Phase 2 report (`reports/neutralize.md`): pilot v1 vs v2, full run 17,038 pairs from 8,769 bursts, real cost $1.06.
- omp setup: `integrations/omp/` holds a `stylesync` skill and an `AGENTS.md` rule so omp calls `rewrite_style` before a commit, a comment or a message; README shows the `mcp.json` entry.
- `reports/project_report.md`: one Vietnamese report of the whole project, Phase 0 to omp (results, costs, privacy, decisions, what is not verified); aggregates only.

### Changed
- Phase 2 provider: OpenAI gpt-6-luna Batch instead of Claude Haiku 5.5 Batch.
- Neutralize prompt v2: keeps IT terms in English ("push code", not "đẩy mã nguồn"), keeps anh/em/chị pronouns, rewrites slang only when its meaning is clear, and adds no subject, question tag or connective the original lacks.
- Plan: GitHub commits/comments dropped as a data source (they were AI-written). Training uses chat data only; commit/comment rewrites are evaluated on a small owner-written set.
- `rewrite_style` sends code spans (inline and fenced) to the model as `<CODE1>`, `<CODE2>`… and puts them back after, since the adapters keep a span's text but drop its backticks; the guard rejects a slot the model made up. On the owner set the guard pass rate goes from 48% to 100% (ft-gemma4-e2b-r16-e1) with meaning unchanged (4.97); `stylesync-eval gen` does the same for systems with `mask: true`.
- `rewrite_style` sends the training system prompt and decodes greedily, as in eval and parity, instead of a descriptive prompt; the guard also rejects a new email, URL or phone number, and a backend failure returns the neutral text with a warning. `health` reports the adapter, model files, last latency and whether this session started the server.
- Train a single voice: no casual/work tone tag in the data pipeline, dataset or `rewrite_style` (work-style data was 8 of 8,769 samples). `configs/threads.yaml` keeps only `include`.

### Fixed
- `rewrite_style` no longer treats a one-line commit such as `refactor: ...` as a trailer and rewrites an empty body: trailers must be the last paragraph, after a blank line, as in git.
- Guard no longer rejects a rewrite that wraps an identifier already in the source in backticks.
- `rewrite_style` rewrites one paragraph at a time. Given a whole commit message, the adapter put filler on the blank line after the subject and broke the body into lines, so git read the whole message as its subject. A commit subject now stays one line, and paragraphs that are only code are not sent to the model.
- `rewrite_style` returns an empty or non-Vietnamese draft unchanged with a warning (it used to translate English into Vietnamese), passes a code-only draft through as is, and rejects a rewrite that hit `max_tokens` instead of returning a cut-off text.
- Guard no longer reports a URL or email from the draft as a new contact detail when punctuation right after it changes.
- A llama-server that exits at start-up (a broken model file, say) is reported at once instead of after 30 seconds, unless another session's server is still loading on the port.
- Guard rejects a rewrite of a draft of 20+ words that keeps less than a quarter of its character trigrams, so a rewrite that drops most of the draft falls back to the neutral text.
- `stylesync-eval report` no longer overwrites a blind sheet the owner has started answering.
