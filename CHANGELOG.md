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

### Changed
- Plan: GitHub commits/comments dropped as a data source (they were AI-written). Training uses chat data only; commit/comment rewrites are evaluated on a small owner-written set.
- Train a single voice: no casual/work tone tag in the data pipeline, dataset or `rewrite_style` (work-style data was 8 of 8,769 samples). `configs/threads.yaml` keeps only `include`.

### Fixed
- Guard no longer rejects a rewrite that wraps an identifier already in the source in backticks.
