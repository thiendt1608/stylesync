# Changelog

## [Unreleased]

### Added
- Implementation plan (`IMPLEMENTATION_PLAN.md`): Discord data window is 36 months (from 2023-10-27); Messenger keeps the full export.
- Phase 0 spike (`spike/phase0/`): GGUF (llama-server, Metal/CPU) and ONNX Runtime GenAI benches over a 40-sentence synthetic Vietnamese test set.
- `stylesync` MCP server (`src/stylesync/mcp_server.py`): tool `rewrite_style` backed by a local llama-server. It keeps the conventional-commit prefix and trailers, and a guard falls back to the neutral text when output is empty, loses or adds code spans, switches away from Vietnamese, leaks placeholders, or changes length too much.
- Phase 0 report (`reports/phase0.md`): runtime is GGUF + llama-server; Gemma 4 E2B and Qwen3.5-2B go to the LoRA pilot.
