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
- Phase 2 report (`reports/neutralize.md`): pilot v1 vs v2, full run 17,038 pairs from 8,769 bursts, real cost $1.06.

### Changed
- Phase 2 provider: OpenAI gpt-6-luna Batch instead of Claude Haiku 5.5 Batch.
- Neutralize prompt v2: keeps IT terms in English ("push code", not "đẩy mã nguồn"), keeps anh/em/chị pronouns, rewrites slang only when its meaning is clear, and adds no subject, question tag or connective the original lacks.
- Plan: GitHub commits/comments dropped as a data source (they were AI-written). Training uses chat data only; commit/comment rewrites are evaluated on a small owner-written set.
- `rewrite_style` sends code spans (inline and fenced) to the model as `<CODE1>`, `<CODE2>`… and puts them back after, since the adapters keep a span's text but drop its backticks; the guard rejects a slot the model made up. On the owner set the guard pass rate goes from 48% to 100% (ft-gemma4-e2b-r16-e1) with meaning unchanged (4.97); `stylesync-eval gen` does the same for systems with `mask: true`.
- Train a single voice: no casual/work tone tag in the data pipeline, dataset or `rewrite_style` (work-style data was 8 of 8,769 samples). `configs/threads.yaml` keeps only `include`.

### Fixed
- `rewrite_style` no longer treats a one-line commit such as `refactor: ...` as a trailer and rewrites an empty body: trailers must be the last paragraph, after a blank line, as in git.
- Guard no longer rejects a rewrite that wraps an identifier already in the source in backticks.
- `stylesync-eval report` no longer overwrites a blind sheet the owner has started answering.
