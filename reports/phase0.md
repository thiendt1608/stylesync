# Phase 0: Spike chọn model + runtime

Ngày đo: 2026-10-27. Script: `spike/phase0/` (bench_gguf.py, bench_onnx.py, mcp_smoke.py, testset.jsonl). Output thô ở `spike/phase0/results/` (gitignore, kh commit).

## TL;DR

- **Model đề xuất:** Gemma 4 E2B (zero-shot tiếng Việt tốt nhất, rõ rệt). **Dự phòng:** Qwen3.5-2B (nhanh hơn, nhẹ hơn nhưng zero-shot hay nhảy sang tiếng Anh). Bước tiếp: train LoRA pilot trên **cả hai** cùng data rồi chọn bằng eval. Phase 0 chưa đủ căn cứ để loại Qwen3.5-2B, vì lỗi chính của nó (đổi ngôn ngữ, bỏ style) đúng là loại lỗi mà SFT sửa được (not verified).
- **Runtime:** GGUF Q4_K_M + `llama-server` (Metal trên Mac). ONNX chạy được với Qwen3.5 nhưng phải dùng 2 workaround và kh nhanh hơn llama.cpp CPU khi tính end-to-end.
- **Gate:** latency đạt cho mọi model. **RAM: Gemma 4 E2B vượt ngưỡng 3GB** (3.3GB trên Metal, 4.8GB trên CPU). **Lỗi ngôn ngữ: kh model nào đạt 0** (zero-shot). Owner quyết: nới ngưỡng RAM cho Gemma, hoặc ưu tiên Qwen3.5-2B.
- **Rủi ro lớn nhất:** model nhỏ **đảo nghĩa** (vd chèn "kh"/"ko" thành phủ định, biến câu hỏi thành câu khẳng định). Guard hiện tại kh bắt được loại lỗi này → Phase 2 bắt buộc có eval giữ nghĩa (LLM judge) trước khi bật dùng thật.
- **Agent test (Claude Code headless, Sonnet 5.5):** nếu chỉ dựa vào docstring thì Claude **kh gọi tool** (0/3), vì MCP tool bị deferred và Claude kh thấy docstring lúc quyết định. Thêm 1 dòng luật kiểu CLAUDE.md thì gọi 3/3 và dùng nguyên văn. → Phase 4 bắt buộc phải có luật trong CLAUDE.md (hoặc skill/hook), kh trông vào docstring được.
- **Chưa verify:** x64 CPU, ONNX cho Gemma 4 E2B và Qwen3-1.7B.

## 1. Môi trường

| | |
|---|---|
| Máy | Apple M4 (4 P-core + 6 E-core), 24GB unified memory, macOS (Darwin 27) |
| llama.cpp | `llama-server` 0.7.0 (build 11669, commit 620d2225c), brew |
| ONNX | onnxruntime-genai 0.18.0, onnxruntime 1.31.1, CPU EP |
| Quant | GGUF Q4_K_M; ONNX int4 (builder `-p int4 -e cpu`) |
| Cấu hình llama | `-c 2048 -np 1 --reasoning off`; Metal `-ngl 99`; CPU `-ngl 0 -dev none -t 4` |
| Sinh | temperature 0, max_tokens 160, `enable_thinking=false` |
| Test set | 40 câu trung tính tiếng Việt do Claude viết (kh phải data thật): 20 message, 10 commit (có prefix conventional + backtick), 10 comment review |

## 2. Export

| Model | GGUF Q4_K_M | ONNX int4 CPU |
|---|---|---|
| Qwen3.5-2B | ✅ tự convert (`convert_hf_to_gguf.py` + `llama-quantize`), 1.31GB | ✅ 1.25GB, cần 2 workaround |
| Qwen3.5-0.8B | ✅ tự convert, 542MB | ✅ 541MB, cần 2 workaround |
| Gemma 4 E2B | ✅ bản unsloth, 3.1GB | ⏭ chưa thử (bản HF bf16 ~10GB, không đáng tải chỉ để so khi GGUF đã là runtime chính) |
| Qwen3-1.7B | ✅ bản unsloth, 1.1GB | ⏭ chưa thử (fallback, chất lượng zero-shot kém nhất) |

Workaround ONNX cho Qwen3.5 (onnxruntime-genai 0.18.0):
1. Builder phải có `--extra_options exclude_mtp=true`. Không có thì load lỗi `GetMemoryInfo for CPU should not be used` (do head MTP).
2. Sửa `tokenizer_config.json` trong thư mục output: `"tokenizer_class": "Qwen3_5Tokenizer"` → `"Qwen2Tokenizer"`. Không sửa thì lỗi `Unsupported tokenizer class`.

Tự convert HF → GGUF cho Qwen3.5 chạy ổn. Đây là đường đi của Phase 3: merge LoRA → convert → quant, hoặc convert adapter rồi `--lora`.

## 3. Tốc độ và RAM (40 câu, mỗi model 1 lần chạy)

`wall` = thời gian 1 request HTTP end-to-end (prefill + decode), tính khi server đã load xong. `RSS` = peak RSS của process (llama-server hoặc python ONNX). Trên Metal, RSS có tính cả weights đã mmap.

| Model | Thiết bị | Load | Peak RSS | wall mean | wall p90 | commit mean / max | decode tok/s |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B | Metal | 0.85s | 1.57GB | 0.33s | 0.48s | 0.33 / 0.48s | 63 |
| Qwen3.5-2B | CPU (llama, 4 thread) | 1.05s | 2.84GB | 0.76s | 0.93s | 0.77 / 1.97s | 45 |
| Qwen3.5-2B | ONNX CPU | 0.93s | 2.50GB | 1.06s | 1.19s | — | 69 |
| Gemma 4 E2B | Metal | 2.52s | **3.33GB** | 0.51s | 0.62s | 0.49 / 0.66s | 52 |
| Gemma 4 E2B | CPU (llama, 4 thread) | 1.65s | **4.78GB** | 1.14s | 1.39s | 1.15 / 2.87s | 37 |
| Qwen3.5-0.8B | Metal | 1.04s | 1.01GB | 0.31s | 0.37s | 0.22 / 0.37s | 111 |
| Qwen3.5-0.8B | CPU (llama, 4 thread) | 0.84s | 1.67GB | 0.62s | 2.11s | 0.61 / 2.12s | 85 |
| Qwen3.5-0.8B | ONNX CPU | 1.36s | 1.58GB | 0.52s | 0.59s | — | 126 |
| Qwen3-1.7B | Metal | 0.43s | 1.62GB | 0.57s | 1.63s | 0.48 / 1.63s | 72 |
| Qwen3-1.7B | CPU (llama, 4 thread) | 0.84s | 2.68GB | 0.99s | 2.50s | 1.14 / 3.56s | 54 |

Ghi chú:
- p90 cao ở Qwen3.5-0.8B/Qwen3-1.7B là do có câu bị lặp tới max_tokens (3–5 câu/40). Gemma và Qwen3.5-2B kh có câu nào chạm max_tokens.
- ONNX decode tok/s cao hơn, nma wall chậm hơn vì bench ONNX prefill lại toàn bộ prompt mỗi request (~154 token), còn `llama-server` cache system prompt. ONNX dùng mặc định toàn bộ core, llama CPU chỉ dùng 4 thread, nên 2 cột này kh so ngang hàng 100%.
- Load model < 3s mọi trường hợp. Do `llama-server` chạy thường trú, load chỉ tốn 1 lần.
- Qwen3-1.7B cho output khác nhau giữa Metal và CPU dù temperature 0 (lệch số học float giữa backend), nên eval phải chạy trên đúng backend sẽ deploy.

## 4. Chất lượng tiếng Việt zero-shot

Chỉ số tự động (Metal; CPU tương tự):

| Model | lẫn CJK | think leak | kh dấu (đổi sang EN) | mất backtick | mất prefix commit | viết tắt/câu |
|---|---|---|---|---|---|---|
| Gemma 4 E2B | 0 | 0 | 1 | 1 | 2 | 0.4 |
| Qwen3.5-2B | 0 | 0 | 10 | 7 | 2 | 0.2 |
| Qwen3.5-0.8B | 0 | 0 | 5 | 2 | 3 | 3.7 |
| Qwen3-1.7B | 0 | 0 | 2 | 1 | 2 | 6.2 |

Chấm tay: Claude chấm, owner chưa duyệt lại. Tiêu chí: giữ đúng ý, ra tiếng Việt, kh lặp, kh đảo nghĩa. Style chưa tính vì zero-shot chưa có style của owner.

| Model | Đạt (ước lượng) | Lỗi chính |
|---|---|---|
| **Gemma 4 E2B** | **~34/40** | Trả lời câu hỏi thay vì viết lại (m18); 1 commit chuyển sang EN (c04); đổi prefix thành `commit:`/`commit perf(db):` (c07, c08); bỏ backtick quanh `next` (c05). Giọng "tui/nha/á/bồ" chứ kh phải t/m của owner. |
| Qwen3.5-2B | ~20/40 | 10/40 câu chuyển hẳn sang tiếng Anh (chủ yếu ở commit). **Đảo nghĩa:** r01 "nên tách" → "ko nên tách", m11 câu hỏi → "chưa push lên dev", r09 "đề xuất đóng" → "đã đóng". Hay bỏ ý (m13, m20, r10). |
| Qwen3.5-0.8B | ~15/40 | Lặp danh sách viết tắt tới max_tokens; token rác "zhi/zhiệt/chk"; bỏ ý. |
| Qwen3-1.7B | ~10/40 | Hầu như chép lại input; **chèn "kh" ở đầu câu** (23/40 câu), đọc như phủ định; lặp tới max_tokens. |

Lỗi đo trong lúc làm: bản prompt đầu có ví dụ `` `fix(auth):` `` → Gemma/Qwen3-1.7B chép prefix này vào cả tin nhắn thường. Đã bỏ ví dụ khỏi prompt và thêm chỉ số `prefix_leak`; sau khi sửa thì `prefix_leak = 0` ở mọi model. Số liệu bên trên là của prompt đã sửa. Bản cũ để ở `results/v1_prompt_with_example/`. Bài học cho data/inference: **kh đặt ví dụ cụ thể trong system prompt**, model nhỏ sẽ chép y nguyên.

## 5. MCP

- `src/stylesync/mcp_server.py` (mcp SDK 2.x, `MCPServer`): tool `rewrite_style(text, task)` + `health()`. Với commit: tách prefix conventional và trailer (`Co-Authored-By:`…), chỉ rewrite phần thân rồi ghép lại.
- Guard: rỗng, mất code span, **thêm code span lạ** (bọc backtick quanh chữ đã có sẵn trong nguồn thì vẫn cho qua), **mất tiếng Việt** (output ≥6 từ mà kh có dấu), lộ placeholder, tỉ lệ độ dài 0.3–2.5. Fail thì trả lại bản trung tính kèm `guard_passed: false`. Unit test: 9 pass (`pytest -q`).
- `mcp_smoke.py` (stdio client thật): list tools ok, health ok. Prefix và trailer được giữ nguyên. Backend Gemma cho ra `` fix(auth): `commit`: Fix login bug… ``, guard bản cũ cho qua → đã thêm 2 check "thêm code span lạ" và "mất tiếng Việt", câu này giờ bị chặn.
- Chạy lại guard trên output Metal: chặn Gemma 3/40, Qwen3.5-2B 14/40, Qwen3.5-0.8B 8/40, Qwen3-1.7B 8/40. **Guard kh bắt được đảo nghĩa** (vd "nên tách" → "ko nên tách" vẫn qua).
- **Agent test** (2026-10-27, `claude -p` headless, model mặc định Sonnet 5.5, backend Gemma 4 E2B Metal, `--strict-mcp-config`, chỉ cho phép `mcp__stylesync__rewrite_style`). 3 prompt (commit, comment PR, tin nhắn team), kh prompt nào nhắc tới tool:

  | Lượt | Cấu hình | Gọi tool | Kết quả |
  |---|---|---|---|
  | 1 | Chỉ có docstring | 0/3 | MCP connected và tool có trong danh sách, nma tool bị **deferred** (phải ToolSearch mới load schema) nên Claude kh thấy docstring, tự viết luôn. |
  | 2 | Thêm 1 dòng luật qua `--append-system-prompt` (giả lập CLAUDE.md) | 3/3 | Tin nhắn: dùng nguyên văn ✅. Commit + comment: guard **bắt nhầm** vì model bọc `get_user`/`parse_date` trong backtick → Claude dùng bản trung tính và báo owner, đúng như luật. |
  | 3 | Như lượt 2, sau khi sửa guard (bọc backtick quanh chữ đã có trong nguồn thì cho qua) | 2/2 (commit, comment) | Comment: trùng nguyên văn ✅. Commit: phần thân trùng nguyên văn, Claude gắn thêm trailer `Co-Authored-By` ở cuối (theo quy ước của Claude Code, chấp nhận được). |

  Kết luận: luồng MCP → llama-server → guard → agent dùng nguyên văn đã chạy được. **Nhưng phải có luật trong CLAUDE.md** (hoặc skill/hook), chỉ có docstring là kh đủ. Mẫu câu luật đã dùng: "trước khi đưa ra commit message, comment PR/issue hoặc tin nhắn thay mặt owner, viết bản trung tính rồi gọi tool mcp__stylesync__rewrite_style (task = commit|comment|message) và dùng NGUYÊN VĂN field text trả về, không sửa gì. Nếu guard_passed=false thì dùng bản trung tính và báo owner." Cỡ mẫu nhỏ (3 prompt, 1 model), chưa thử Opus/Haiku.

## 6. Gate

| Ngưỡng (gợi ý trong plan) | Gemma 4 E2B | Qwen3.5-2B |
|---|---|---|
| Commit ≤1.5s Metal | ✅ 0.49s mean, 0.66s max | ✅ 0.33s / 0.48s |
| Commit ≤3s CPU (M4) | ✅ 1.15s mean, 2.87s max | ✅ 0.77s / 1.97s |
| Commit ≤3s CPU x64 | not verified | not verified |
| RAM ≤3GB | ❌ 3.33GB Metal, 4.78GB CPU | ✅ 1.57GB Metal, 2.84GB CPU |
| 0 lỗi dấu/lẫn ngôn ngữ | ❌ 1 câu EN | ❌ 10 câu EN |

Không model nào qua hết gate ở trạng thái zero-shot. Cái này đã đoán trước, vì zero-shot chỉ đo nền. Đề xuất:
1. Chốt runtime **GGUF + llama-server** (kh còn tranh cãi).
2. Giữ **2 ứng viên vào Phase 2: Gemma 4 E2B và Qwen3.5-2B**. LoRA pilot nhỏ (~1–2k cặp) cho cả hai trên 3060/Colab rồi so bằng eval. Loại Qwen3.5-0.8B và Qwen3-1.7B.
3. Nếu chọn Gemma: nới ngưỡng RAM lên ~3.5GB cho Metal (máy 24GB dư sức), hoặc thử quant nhỏ hơn (IQ4_XS/Q3_K_M) ở Phase 3 (not verified).
4. Thêm **eval giữ nghĩa** (LLM judge so nghĩa nguồn ↔ output, đặc biệt phủ định/câu hỏi) làm gate cứng ở Phase 2. Guard regex kh đủ.

## 7. Việc còn lại / cách chạy lại

- x64 CPU: chạy `bench_gguf.py … cpu` trên một máy x64 hoặc GitHub Actions ubuntu runner (repo chưa có remote, chưa làm).
- Chạy lại agent test:
  ```bash
  llama-server -m models/gguf/gemma-4-E2B-it-Q4_K_M.gguf --port 18080 -c 2048 --reasoning off -np 1 -ngl 99
  claude -p "<prompt>" --append-system-prompt "<luật ở mục 5>" --mcp-config mcp.json --strict-mcp-config --allowedTools mcp__stylesync__rewrite_style --output-format stream-json --verbose --max-turns 4
  ```
  với `mcp.json` = `{"mcpServers":{"stylesync":{"command":"<repo>/.venv/bin/python","args":["-m","stylesync.mcp_server"]}}}`. Cần kiểm: có `tool_use` gọi `rewrite_style` kh, và output cuối có trùng nguyên văn với `text` trả về kh.
- Agent test với Opus/Haiku và cỡ mẫu lớn hơn: chưa làm.
- Đa nền tảng: chỉ đo trên macOS arm64. **not verified: Windows, Linux.**
