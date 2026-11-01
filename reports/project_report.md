# Báo cáo toàn bộ project stylesync (fine-tune-chat)

Viết ngày 2026-11-01, sau commit `3b1f3ce`. Báo cáo chỉ có số tổng hợp, kh có nội dung tin nhắn hay tên người.
Nguồn số liệu: `reports/*.md`, `IMPLEMENTATION_PLAN.md`, `CHANGELOG.md`, git log, và log các lần chạy e2e trong phiên làm việc.
Chỗ nào chưa kiểm được thì ghi **not verified**.

---

## 0. TL;DR

- **Mục tiêu:** cho Claude Code (và omp) viết commit, comment PR/issue, tin nhắn đúng giọng của owner. Claude viết bản trung tính tiếng Việt, một model nhỏ chạy local viết lại thành giọng owner.
- **Kết quả:** LoRA r16 trên **Gemma 4 E2B**, chạy bằng GGUF (base Q4_K_M + adapter LoRA f16 48MB) qua `llama-server`. MCP server `stylesync` có tool `rewrite_style`, đã đăng ký cho Claude Code (scope user) và omp.
- **Chất lượng (test 1,708 mẫu chat):** style 100% (bằng tin thật), giữ ý 4.81/5 so với Baseline B (gpt-6-luna few-shot) 4.90. Qua toàn bộ gate.
- **Bộ commit/comment owner tự gõ (40 mục):** guard pass 100% sau khi che code span, giữ ý 4.97/5.
- **Blind test:** owner đoán đúng 34/50 (68%, p ≈ 0.008). Owner vẫn phân biệt được, **chưa đạt** mục tiêu ~50%.
- **Tốc độ dùng thật:** Mac (Metal) 0.4–0.9s/lần. Windows/Linux (CPU + GPU 3060) khoảng 1.3s/lần sau lần gọi đầu.
- **Chi phí API:** tổng ≈ **$1.62** OpenAI Batch: Phase 2 $1.08, Phase 5 $0.54. Train trên GPU nhà (RTX 3060), kh tốn tiền cloud.
- **Riêng tư:** data, adapter và model kh lên git, kh public. Commit chỉ có code, config mẫu, test giả và report tổng hợp.
- **Còn thiếu:** CI 3 OS chưa chạy lần nào (repo chưa có remote). Claude Code interactive trên Windows chưa test. omp mới thử với model mặc định. Blind test chưa đạt.

---

## 1. Bối cảnh và mục tiêu

### 1.1. Bài toán

Owner muốn agent AI (Claude Code, omp) viết commit message, comment GitHub, tin nhắn bằng **đúng giọng của mình**: viết tắt (t/m/kh/đc…), xưng anh/em, code-mix tiếng Anh, độ dài câu… thay vì giọng "văn mẫu" của LLM.

### 1.2. Thiết kế tổng thể

```
Claude Code / omp
  │  1. viết bản trung tính tiếng Việt (xưng anh/em, code trong backtick)
  ▼
MCP server `stylesync` (stdio, Python, FastMCP)
  │  2. tách prefix `type(scope):` + trailer, che code span → <CODE1>…
  │  3. gửi từng đoạn tới llama-server dùng chung (127.0.0.1:18080)
  ▼
llama-server (base Gemma 4 E2B Q4_K_M + --lora stylesync)
  │  4. greedy rewrite, system prompt y như lúc train
  ▼
MCP server: gắn lại code, prefix, trailer → guard kiểm → trả `text` + `guard_passed` + `warnings`
  │  guard fail → trả lại bản trung tính (fallback an toàn)
  ▼
Agent dùng nguyên văn `text` để commit / comment / nhắn tin
```

**Nguyên tắc chính:**
- Model chỉ đổi văn phong. Format cứng (prefix commit, trailer, code, path, identifier) do code giữ, model kh được đụng vào.
- Lỗi gì cũng **fallback về bản trung tính**, kh bao giờ để agent kẹt.
- 1 `llama-server` dùng chung cho mọi session, kh load model riêng cho từng session.

### 1.3. Máy móc

| Máy | Vai trò |
|---|---|
| Laptop Apple M4, 24GB | Máy code; chạy MCP + llama-server (Metal), convert GGUF, eval nhẹ. Kh train nặng. |
| Desktop RTX 3060 12GB, Windows 11 + WSL2 Ubuntu 24.04 | Train LoRA, gen eval, test e2e Windows/Linux. Truy cập qua Tailscale + OpenSSH (`ssh pc`), job chạy trong `tmux`. |

---

## 2. Dòng thời gian và commit

| Commit | Thời gian | Nội dung | Thay đổi |
|---|---|---|---|
| `9689621` | 2026-10-27 17:57 | Phase 0 spike, MCP server bản đầu, report | 16 file, +2,121 |
| `7ced576` | 2026-10-27 18:04 | Guard cho phép backtick quanh từ có sẵn; kết quả agent test | 4 file |
| `7d7ca8f` | 2026-10-27 18:14 | Bỏ commit/comment GitHub khỏi nguồn data train | 2 file |
| `985e0e9` | 2026-10-27 18:41 | Phase 1: ingest + clean, chốt 1 giọng | 19 file, +911 |
| `cb709e6` | 2026-10-28 16:47 | Phase 2: neutralize bằng OpenAI Batch | 11 file, +793 |
| `a6dc4f8` | 2026-10-28 16:56 | Phase 3: build train/val/test | 8 file, +268 |
| `9e668b9` | 2026-10-30 10:22 | Phase 4: LoRA SFT với Unsloth | 9 file, +1,933 |
| `a4c7fe2` | 2026-11-01 09:17 | Phase 5: eval + che code span trong `rewrite_style` | 20 file, +1,477 |
| `943ba4f` | 2026-11-01 09:53 | Phase 6: export GGUF + parity | 8 file, +405 |
| `6975e53` | 2026-11-01 09:58 | Phase 7: backend llama-server dùng chung | 12 file, +424 |
| `e0fdb52` | 2026-11-01 10:27 | Hardening sau e2e thật (5 bug + coverage guard) | 7 file, +178 |
| `3b1f3ce` | 2026-11-01 11:25 | Cấu hình stylesync cho omp | 5 file, +59 |

Tổng cộng 12 commit trong khoảng 6 ngày (2026-10-27 → 2026-11-01). Repo chưa có remote.

---

## 3. Phase 0: spike chọn model và runtime (2026-10-27)

**Làm gì:** bench 4 model nhỏ trên 40 câu trung tính giả do Claude viết (20 message, 10 commit, 10 comment), với 2 runtime (GGUF/llama.cpp và ONNX Runtime GenAI). Script ở `spike/phase0/`, report `reports/phase0.md`.

### 3.1. Tốc độ và RAM (M4)

| Model | Thiết bị | Peak RSS | wall TB | decode tok/s |
|---|---|---|---|---|
| Qwen3.5-2B | Metal | 1.57GB | 0.33s | 63 |
| Qwen3.5-2B | CPU 4 thread | 2.84GB | 0.76s | 45 |
| Gemma 4 E2B | Metal | **3.33GB** | 0.51s | 52 |
| Gemma 4 E2B | CPU 4 thread | **4.78GB** | 1.14s | 37 |
| Qwen3.5-0.8B | Metal | 1.01GB | 0.31s | 111 |
| Qwen3-1.7B | Metal | 1.62GB | 0.57s | 72 |

### 3.2. Chất lượng tiếng Việt zero-shot (Claude chấm tay)

| Model | Đạt (ước lượng) | Lỗi chính |
|---|---|---|
| **Gemma 4 E2B** | **~34/40** | Trả lời câu hỏi thay vì viết lại, 1 commit sang tiếng Anh, đổi prefix commit |
| Qwen3.5-2B | ~20/40 | 10/40 câu nhảy sang tiếng Anh; **đảo nghĩa** (thêm "ko", câu hỏi thành khẳng định) |

### 3.3. Quyết định

- **Runtime: GGUF Q4_K_M + `llama-server`.** ONNX chạy được Qwen3.5 nhưng cần 2 workaround (`exclude_mtp=true`, sửa `tokenizer_class`) và kh nhanh hơn end-to-end.
- **Model:** train cả Gemma 4 E2B và Qwen3.5-2B rồi chọn bằng eval thật. Gemma vượt ngưỡng RAM 3GB (3.3GB), chấp nhận vì Mac có 24GB.
- **Agent test (Claude Code headless):** chỉ có docstring của tool → Claude **0/3** lần gọi tool, vì MCP tool bị deferred nên Claude kh thấy mô tả. Thêm 1 dòng luật kiểu CLAUDE.md → **3/3** và dùng nguyên văn. Từ đó mọi tích hợp đều kèm rule (CLAUDE.md, AGENTS.md, skill).
- **Rủi ro lớn nhất được ghi nhận sớm:** model nhỏ đảo nghĩa. Vì vậy Phase 5 bắt buộc có LLM-judge chấm giữ ý.

### 3.4. MCP server bản đầu

Commit đầu đã có `stylesync-mcp` với tool `rewrite_style(text, task)` và guard: giữ prefix `type(scope):`, trailer `Co-Authored-By`, backtick. Commit `7ced576` sửa guard: cho phép model thêm backtick quanh từ đã có sẵn trong bản gốc.

### 3.5. Đổi plan: bỏ data GitHub

Owner xác nhận commit/comment cũ trên GitHub đều do AI viết, kh phải giọng mình. Vì vậy **chỉ train trên chat** (Messenger + Discord), commit/comment dùng chung giọng chat, còn format cứng do code giữ (commit `7d7ca8f`).

---

## 4. Phase 1: ingest + clean data (2026-10-27, `985e0e9`)

**Lệnh:** `uv run stylesync-data` (config `configs/filter.yaml`), chạy ~6s trên M4. Report: `reports/data.md`.

### 4.1. Pipeline

| Bước | Messenger | Discord |
|---|---:|---:|
| Tin của owner (ingest) | 57,526 | 82,748 |
| Bỏ rác (rỗng, hệ thống, chỉ link/emoji) | 55,381 | 77,107 |
| Gộp burst (tin liên tiếp ≤120s) | 10,737 | 23,021 |
| PII → placeholder, độ dài 20–400 ký tự | 7,025 | 14,955 |
| Bỏ nhiễu (code dán vào, spam, burst toàn placeholder) | 7,008 | 14,734 |
| Dedup | 7,002 | 14,615 |
| Bỏ burst tiếng Anh | 6,986 | 13,406 |
| Bỏ burst kh dấu (≥6 từ) | 6,951 | 10,901 |
| Cutoff thời gian (Discord ≥ 2023-10-27) | **6,951** | **1,818** |

**Tổng: 8,769 mẫu**, dài trung bình 92.8 ký tự.

### 4.2. Chi tiết kỹ thuật

- **Ingest:** Messenger tự nhận "me" (top sender + có mặt ở nhiều thread nhất); sửa mojibake của export Facebook. Discord: package chỉ chứa tin mình gửi.
- **PII scrub:** `<NAME>` 701, `<PHONE>` 15, `<EMAIL>` 15, `<NUMBER>` 10, `<URL>` 9, `<SECRET>` 3. Có thêm rule cho JWT/token dài, credential dạng `pw:`, số 6–8 chữ số (OTP/MSSV).
- **`thread_id` được hash**, nên file data kh mang tên thread.
- **Style drift theo quý:**
  - Đo viết tắt, có dấu, emoticon, code-mix, độ dài (`reports/drift.svg`).
  - Discord từ 2023Q4 đã gần giọng hiện tại → chốt cutoff 2023-10-27.
  - Emoticon cũ cao gấp ~9 lần hiện tại → burst Discord trước 2025-04-01 có **weight 0.5** (883 mẫu).
- **Review tay 3 lượt × 200 mẫu.** Mỗi lỗi tìm được biến thành rule có test:

  | Lượt | Tỉ lệ lỗi |
  |---|---|
  | 1 | 2.5% |
  | 2 | 4.5% |
  | 3 | **0.5%** |

  Ngưỡng plan là 10%, đạt.
- **Tone:** work chỉ có 8/8,769 mẫu (0.09%). **Owner chốt train 1 giọng, kh tag tone.**

### 4.3. Bảo mật

- Credential và token dán trong chat được che bằng rule (`<SECRET>`, rule credential) trước khi gửi đi hay train.
- Tên gọi viết thường (vd "ê <tên> ơi") còn sót ~1/10 mẫu. Lý do: participant list lẫn tên page/shop, nên rule bắt tên viết thường che nhầm từ thường. Chấp nhận vì adapter private.

---

## 5. Phase 2: neutralize / back-translation (2026-10-28, `cb709e6`)

**Ý tưởng:** có tin thật của owner (output) nhưng thiếu bản trung tính (input) → nhờ LLM lớn "dịch ngược" tin thật ra văn trung tính, tạo cặp (trung tính → giọng owner).

**Cách làm:**
- **Model:** OpenAI **gpt-6-luna** qua **Batch API**, `reasoning.effort: low`, output JSON schema strict `{formal, concise}`. Owner đổi từ Claude Haiku vì có sẵn credit OpenAI.
- **Privacy:** chỉ gửi text đã scrub PII, `store: false`, xoá file batch trên OpenAI sau khi collect. Key nằm trong Keychain/env, kh ghi vào repo.
- **2 biến thể cho mỗi mẫu** (trang trọng + trung tính ngắn gọn): data x2, model kh overfit vào 1 kiểu câu.
- **Validator tự động** loại cặp khi:
  - rỗng hoặc y bản gốc,
  - tỉ lệ độ dài ngoài [0.5, 3],
  - mất số/code/placeholder,
  - còn viết tắt hoặc emoticon,
  - mất dấu,
  - model từ chối hoặc bị cắt.
- **Cache theo hash** (model, prompt_version, text): chạy lại kh tốn tiền.

**Pilot 200 burst:**

| | v1 | v2 |
|---|---:|---:|
| Pair qua validator / 400 | 383 | 383 |
| Cặp lệch nghĩa (spot-check tay /100) | **7** | kh chấm lại (owner chốt chạy full) |
| Thuật ngữ tiếng Anh còn giữ | 43% | 59% |
| Chi phí | $0.0187 | $0.0242 |

Prompt v2 sửa các lỗi của v1:
- giữ thuật ngữ IT bằng tiếng Anh;
- t/m → tôi/bạn nhưng giữ anh/em/chị;
- kh tự thêm chủ ngữ;
- kh gộp dòng;
- câu cảm thán kh biến thành mệnh lệnh.

**Full run (prompt v2):**

| | Số |
|---|---:|
| Request completed / lỗi / refusal / bị cắt | 8,769 / 0 / 0 / 0 |
| **Pair qua validator** | **17,038** (formal 8,550, concise 8,488) |
| Burst còn ≥1 pair | 8,567 / 8,769 (97.7%) |
| Variant bị loại | 500 (2.9%); nhiều nhất do mất số (206) và "y bản gốc" (188) |
| Token | input 8.74M, output 2.50M (reasoning 1.65M) |
| Thời gian batch | ~4.3 giờ |
| **Chi phí** | **$1.06** (cả Phase 2: **$1.08**) |

---

## 6. Phase 3: dataset build (2026-10-28, `a6dc4f8`)

**Lệnh:** `uv run stylesync-dataset`. Report: `reports/dataset.md`.

- **Format:**
  - Dạng prompt-completion của TRL, system prompt cố định: "Bạn là bộ viết lại văn phong. Giữ nguyên ý, code, identifier. Chỉ trả về bản viết lại."
  - Chỉ tính loss trên completion.
  - MCP server Phase 7 dùng **đúng system prompt này** (`stylesync.dataset.build.SYSTEM`).
- **Split theo thời gian, riêng từng nguồn:**
  - 10% burst mới nhất làm test, 10% kế tiếp làm val.
  - **Embargo 24h** ở ranh giới split.
  - 2 variant của 1 burst luôn cùng split.
  - Kh split random, vì burst cùng hội thoại rất giống nhau, split random sẽ leak.
- **Weight 0.5** làm bằng cách chỉ giữ 1/2 variant trong train, nên trainer kh cần hỗ trợ sample weight.

| Split | Example | Burst | Messenger | Discord |
|---|---:|---:|---:|---:|
| train | 12,707 | 6,823 | 10,775 | 1,932 |
| val | 1,649 | 826 | 1,315 | 334 |
| test | 1,708 | 857 | 1,356 | 352 |

- Bỏ 852 variant do weight, 122 example do embargo.
- **Leak check:** 0 bản gốc trùng giữa train–val và train–test.
- Dài nhất 388 token (Gemma), cộng template vẫn < `max_seq_len` 512.

---

## 7. Phase 4: fine-tune LoRA (2026-10-29 → 30, `9e668b9`)

### 7.1. Setup

- **Máy:** RTX 3060 12GB, WSL2 Ubuntu 24.04. Từ laptop vào qua `ssh pc`, job chạy trong `tmux`.
- **WSL tự tắt distro khi kh còn `wsl.exe` nào mở, kéo theo tmux.** Owner đồng ý tạo scheduled task `wsl-keepalive` (S4U, chạy lúc khởi động) để giữ distro sống.
- **Checkpoint mỗi 100 step + `--resume`**, phòng PC bị tắt giữa chừng.
- **Stack:** Unsloth 2026.10.10, torch 2.14.1 (CUDA 13.0), transformers 5.17.0, TRL 1.15.0, PEFT 0.21.2. Dependency group `train` chỉ cài trên Linux.
- **Hyperparameter:**

  | Tham số | Giá trị |
  |---|---|
  | Kiểu | LoRA bf16, **kh QLoRA** (Unsloth khuyến cáo cho Qwen3.5) |
  | `r` / `lora_alpha` | 16 / 16 |
  | Target modules | attention + MLP |
  | LR | 2e-4, cosine, warmup 3% |
  | Effective batch | 16 |
  | Max seq len | 512 |
  | Eval val | mỗi 100 step |
  | Early stop | 2 lần eval liên tiếp kh tốt hơn |

- **Chat template:** render bằng template của chính base model, non-thinking, giống hệt cái MCP gửi cho llama-server. Đã kiểm template prefix-stable trên cả 14,356 example train+val.

### 7.2. Kết quả

| Run | Step | Thời gian | Val loss | Ghi chú |
|---|---|---|---|---|
| smoke (20 step, 400 ex) | 20 | — | 2.17 → 2.05 | VRAM ~5.4GB |
| `qwen35-2b-r16-e1` | 795 | 19.9 phút | 1.369 → **1.089** | best ở step cuối |
| `qwen35-2b-r16-e2` | 1000/1590 (early stop) | 23.4 phút | best **1.085** @ step 800 | epoch 2 overfit (1.111 → 1.114) → 1 epoch là đủ |
| `gemma4-e2b-r16-e1` | 795 | 31.3 phút | 1.164 → **0.941** | best ở step cuối |

Val loss của Gemma và Qwen kh so thẳng được vì khác tokenizer, nên chọn model ở Phase 5.

### 7.3. Lỗi gặp và cách sửa

- **Gemma 4 E2B kh gated.** Lúc đầu nói nhầm là gated, sau đó sửa lại; chỉ cần `hf auth login`.
- Unsloth tự đổi repo sang `unsloth/gemma-4-E2B-it` → load với `use_exact_model_name=True`.
- torch.compile của Unsloth làm backward đầu tiên lỗi (checkpoint recompute lệch graph) → tắt compile cho Gemma (`no_compile`).
- Unsloth báo template Gemma tạo double BOS, tự bỏ 1 cái lúc train. Phase 6 phải kiểm BOS (đã kiểm, xem mục 9).

---

## 8. Phase 5: eval, gate trước khi làm MCP (2026-10-30 → 31, `a4c7fe2`)

**Lệnh:** `uv run stylesync-eval gen|fewshot|judge|report`. Report: `reports/eval.md`, `reports/eval_owner.md`.

### 8.1. Hệ thống so sánh

| Hệ thống | Là gì |
|---|---|
| `neutral` | Input trung tính, giữ nguyên (sàn) |
| `reference` | Tin thật của owner (trần) |
| `base-*` | Baseline A: base model zero-shot |
| `ft-*` | 3 model fine-tune của Phase 4 |
| `fewshot-gpt6luna` | Baseline B: gpt-6-luna + 10 ví dụ train giống nhất (TF-IDF char n-gram) |

- **Gen local:** 5 hệ thống × 1,708 mẫu, greedy, tổng ~20 phút trên 3060.
- **Judge giữ ý:** gpt-6-luna Batch. Judge cùng model với B nên có thể thiên vị B, coi điểm của B là cận trên.
- **Style classifier:** char n-gram + logistic regression (tin thật vs bản trung tính).

### 8.2. Kết quả trên test chat (1,708 mẫu)

| Hệ thống | Style | Giữ ý (1–5) | Added | Lost | ≤2 điểm |
|---|---:|---:|---:|---:|---:|
| reference (tin thật) | 100% | 4.65 | 20% | 6% | 1% |
| base-qwen35-2b | 16% | 4.14 | 13% | 42% | 13% |
| base-gemma4-e2b | 6% | 4.58 | 10% | 22% | 4% |
| ft-qwen35-2b-r16-e1 | 100% | 4.75 | 8% | 10% | 3% |
| ft-qwen35-2b-r16-e2 | 100% | 4.76 | 8% | 9% | 2% |
| **ft-gemma4-e2b-r16-e1** | **100%** | **4.81** | 7% | 7% | 1% |
| fewshot-gpt6luna (B) | 99% | 4.90 | 4% | 3% | 1% |

**Stylometry của ft-gemma rất sát tin thật:**

| | ft-gemma | Tin thật |
|---|---:|---:|
| Độ dài TB | 86.3 | 87.1 |
| Viết tắt | 12% | 11% |
| Có dấu | 67% | 64% |

**Hiệu chuẩn judge:** judge gắn cờ "thêm ý" cho 20% tin thật, vì bản trung tính Phase 2 hay rớt chi tiết. → **Owner chốt:** ngưỡng added = tỉ lệ của tin thật + 3 điểm %, kh phải 3% tuyệt đối.

**Gate:** cả 3 ft đều pass mọi rule:
- thắng base về style;
- style ≥ 70%;
- giữ ý ≥ 4.3;
- added trong ngưỡng;
- kém B ≤ 0.3 điểm (ft-gemma chỉ kém 0.09).

Style classifier bão hoà (ft = tin thật = 100%), nên xếp hạng giữa các ft dựa vào judge + stylometry → chọn **ft-gemma4-e2b-r16-e1**.

### 8.3. Blind test

- Owner tự chấm 50 cặp (ft-gemma vs tin thật trộn ngẫu nhiên): đoán đúng **34/50 = 68%**, p ≈ 0.008.
- → Owner vẫn phân biệt được, **chưa đạt** mục tiêu "gần 50%".
- Lưu ý: người chấm là chính chủ, có thể nhớ tin mình từng gửi, nên đây là bài khó nhất. Nên cho thêm 1–2 người khác chấm.

### 8.4. Bộ commit/comment owner tự gõ (40 mục)

Test chat gần như kh có code (chỉ 6 input có hard token), nên owner tự viết 20 commit + 20 comment, bản trung tính xưng anh/em.

- **Giữ ý:** ft-gemma 4.97, B 5.0, bản của owner 4.6.
- **Chữ trong backtick:** giữ đủ 23/23 ở mọi ft.
- **Vấn đề:** ft hay **bỏ dấu backtick** vì data chat gần như kh có backtick. Guard pass chỉ 48% (ft-gemma), so với B 98% → khoảng ½ commit/comment sẽ rơi về bản trung tính.
- **Cách sửa: che code span.** `rewrite_style` thay mỗi code span (inline + fenced) bằng `<CODE1>`, `<CODE2>`… trước khi đưa model, rồi gắn lại bằng code. Guard từ chối slot do model tự bịa. **Kh cần train lại**, vì lúc train model đã giữ placeholder 100%.
- **Kết quả sau khi che:**

  | Model | Guard trước → sau | Giữ ý |
  |---|---|---|
  | ft-gemma | 48% → **100%** | 4.97, lost 0% |
  | ft-qwen e1 | 72% → 100% | — |
  | ft-qwen e2 | 65% → 95% | — |

**Chi phí OpenAI Phase 5:** $0.54 (Baseline B + judge, cả 2 bộ: 6.17M input / 0.93M output token).

Batch OpenAI có lúc kẹt hàng đợi hàng giờ: B đứng ~5 giờ ở 1,505/1,546, bộ owner ~11 giờ. Owner chọn chờ chứ kh chạy API thường. Runner Batch được thêm retry khi rớt mạng và `--no-wait`.

---

## 9. Phase 6: export GGUF + parity (2026-11-01, `943ba4f`)

**Lệnh:** `uv run stylesync-export convert|parity|report|install`. Report: `reports/export.md`.

- **Build:**
  - Base `gemma-4-E2B-it-Q4_K_M.gguf` (bản Unsloth) + `--lora stylesync-...-lora-f16.gguf` (**48MB**, 205 module LoRA, chỉ phần language).
  - Convert bằng `convert_lora_to_gguf.py` ngay trên laptop, chỉ cần config, kh phải tải bf16 10GB.
  - venv converter phải lên transformers 5.17 (4.57 kh biết `gemma4`).
- **Chat template:** template trong GGUF chỉ khác template HF ở phần tool-call. System/user render giống nhau.
- **BOS:** llama-server bỏ `<bos>` của template rồi tự thêm 1 cái → đúng 1 BOS như lúc train, ở 100% request.
- **Parity vs PyTorch bf16:**

  | Bộ | Trùng y chang (Metal / CPU) | chrF | Guard |
  |---|---|---|---|
  | Chat 200 mẫu | 59% / 57% | 95.4 / 95.1 | 96–97% (PyTorch 97%) |
  | Owner 40 mẫu (mask) | 60% / 55% | ~95 | 100% |

  Metal và CPU trùng nhau 85% → phần lệch chủ yếu do base Q4_K_M + khác kernel, kh phải lỗi template/EOS. Kh câu nào chạm max_tokens.
- **Latency** (cache prompt tắt, M4):

  | | TB / p90 (chat) | TB / p90 (owner) | tok/s |
  |---|---|---|---|
  | Metal | 0.93s / 1.93s | 0.60s / 0.74s | 45–48 |
  | CPU 4 thread | 2.69s / 4.48s | 1.64s / 1.98s | 29.6–32 |

- **Not verified:** chưa chấm judge cho output GGUF; chưa có bản merge 1 file.

---

## 10. Phase 7: MCP server + backend dùng chung (2026-11-01, `6975e53`)

- **`src/stylesync/llama.py`:**
  - MCP server đầu tiên kh thấy server nào đang chạy thì spawn `llama-server` **detached** ở `127.0.0.1:18080` (Windows: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, Unix: `start_new_session`).
  - Session sau dùng lại server đó.
  - Từ chối server trên port mà kh load đúng LoRA (vd server bench cũ của Phase 0).
  - Chỉ tự start khi URL là localhost.
- **Model:** nằm ở `platformdirs.user_cache_dir("stylesync")` (macOS `~/Library/Caches/stylesync`), do `stylesync-export install` copy vào. Đổi được qua env `STYLESYNC_LLAMA_URL`, `STYLESYNC_MODEL_DIR`, `STYLESYNC_BASE_GGUF`, `STYLESYNC_LORA_GGUF`, `STYLESYNC_LLAMA_SERVER`.
- **Sinh:** greedy (temperature 0), `enable_thinking=False`, system prompt y như lúc train.
- **Tool:**
  - `rewrite_style(text, task)` trả `text`, `guard_passed`, `warnings`.
  - `health()` báo trạng thái backend và `started_server`.
- **Guard:**
  - prefix commit, trailer, code span;
  - placeholder bịa;
  - **PII** (model kh được tự thêm email/sđt/URL);
  - kh dấu, dịch sang tiếng Anh.
- **Smoke stdio trên Mac:** 0.4–0.9s/lần.
- **Đăng ký Claude Code:**
  - `claude mcp add --scope user stylesync`, dùng path tuyệt đối `/opt/homebrew/bin/uv` + env `STYLESYNC_LLAMA_SERVER`, vì app desktop kh có Homebrew trong PATH.
  - Thêm rule `## stylesync` ở cuối `~/.claude/CLAUDE.md`, vì Phase 0 đã chứng minh chỉ có docstring thì agent kh gọi tool.

---

## 11. E2E thật và hardening (2026-11-01, `e0fdb52`)

Owner hỏi "đã test kĩ chưa, test e2e dùng thật xem lỗi gì kh". Chạy thật qua `claude -p` (Claude tự gọi tool theo rule) và client MCP stdio (script `e2e_os.py`, chỉ dùng text giả). Phát hiện và sửa **5 bug:**

1. **Văn bản nhiều đoạn bị gộp hoặc mất đoạn.** → Viết lại **từng đoạn** riêng, giữ dòng trống giữa subject và body của commit.
2. **Input tiếng Anh bị dịch sang tiếng Việt.** → Kh dịch: bản tiếng Anh trả nguyên, kèm warning.
3. **Output bị cắt ở max_tokens vẫn được dùng.** → `chat` raise `Truncated` khi `finish_reason == "length"`, fallback về bản trung tính.
4. **Guard PII báo nhầm vì dấu câu dính vào token.** → Chuẩn hoá dấu câu trước khi so.
5. **Server chết lúc load (vd LoRA hỏng) thì MCP đứng chờ hết 90s timeout.** → `ensure` báo lỗi **ngay** khi process thoát và kh còn ai listen trên port (`listening()`). Vẫn xử lý được trường hợp 2 session cùng start: process của mình thoát vì port bị chiếm thì dùng server của session kia.

Thêm **coverage guard:** với bản nháp ≥20 từ, nếu output giữ dưới 25% trigram ký tự của bản nháp thì coi là model viết lạc đề và từ chối.

### 11.1. Test trên Windows và Linux (PC 3060)

Setup: test env ở `E:\xfer\e2e` (llama.cpp Windows, uv Windows, code, models); WSL ở `~/e2e`. Code đưa sang bằng tarball (`COPYFILE_DISABLE=1` để macOS kh nhét file `._*`).

| | Linux (WSL2) | Windows 11 |
|---|---|---|
| pytest | 96 pass | 96 pass |
| Lần gọi đầu (cold, có start server) | 18.6s (model trên `/mnt/e`, đọc qua 9p chậm) | 6.6s |
| Các lần sau | ~1.3–1.4s | ~1.3s |
| LoRA hỏng → báo lỗi | 7.3s | 8.5s |
| Server sống sau khi session đóng | Có (`started_server: false` ở lần sau) | **Kh** khi chạy qua ssh |
| Guard 4 case (commit/comment/message/tiếng Anh) | pass | pass |

**Windows:** OpenSSH server kill cả cây process khi phiên ssh đóng. Đã thử job breakaway, kh đỡ, nên bỏ code đó. Hậu quả nhẹ: lần gọi sau tự spawn lại server. Chạy Claude Code interactive trên chính máy Windows (kh qua ssh) thì server được kỳ vọng sống tiếp, **not verified**.

---

## 12. Tích hợp omp / oh-my-pi (2026-11-01, `3b1f3ce`)

Owner dùng cả omp v18.13.3, nên thêm 3 thứ để omp tự gọi stylesync:

- **MCP:**
  - Entry `stylesync` trong `~/.omp/agent/mcp.json` (backup `mcp.json.bak-stylesync`): stdio, `/opt/homebrew/bin/uv run --directory <repo> stylesync-mcp`, env `STYLESYNC_LLAMA_SERVER`.
  - Trong omp tool tên là `mcp__stylesync_rewrite_style`, gọi qua `xd://`.
- **Skill:** `integrations/omp/stylesync/SKILL.md`, symlink vào `~/.agents/skills/stylesync`. Quy trình 5 bước:
  1. Viết bản trung tính (anh/em, backtick).
  2. Gọi tool.
  3. Dùng nguyên văn kết quả; `guard_passed: false` thì dùng bản trung tính và báo lý do.
  4. Commit bằng `git commit -F` để giữ xuống dòng.
  5. Kh tự gửi tin cho người thật.
- **Rule:** `integrations/omp/AGENTS.md`, nối vào `~/.omp/agent/AGENTS.md`.
- **Kh đụng** `APPEND_SYSTEM.md` và `~/.omp/skills`, vì 2 chỗ đó do `8sync harness` quản lý.

**E2E omp** (`omp -p`, model mặc định gemini-3.8-flash, prompt **kh nhắc** tới stylesync):

| Case | Kết quả |
|---|---|
| Commit trong repo test | Đọc `skill://stylesync` → gọi tool → commit dùng nguyên văn, `guard_passed: true` |
| Comment ghi ra file | Như trên |
| Tin nhắn | Gọi tool, đưa bản viết lại cho owner copy, **kh tự gửi** |

Lưu ý: `omp -p` phải có `< /dev/null`, kh thì treo ở `readPipedInput`.

---

## 13. Code và test

### 13.1. Cấu trúc

```
src/stylesync/
  ingest/      messenger.py, discord.py            Phase 1
  clean/       burst, filters, pii, drift, pipeline Phase 1
  neutralize/  prompt, validate, run               Phase 2
  dataset/     build.py (SYSTEM prompt)            Phase 3
  train/       sft.py (Unsloth + TRL)              Phase 4
  eval/        generate, llm, metrics, owner_set, run   Phase 5
  export/      parity.py                           Phase 6
  llama.py     backend llama-server dùng chung      Phase 7
  mcp_server.py  rewrite_style, health, guard, mask   Phase 0 → 7
configs/       filter, neutralize, dataset, train, eval, export (.yaml); threads.example.yaml
spike/phase0/  bench GGUF/ONNX, mcp smoke, 40 câu test giả
integrations/omp/  SKILL.md + AGENTS.md
reports/       phase0, data (+drift.svg), neutralize, dataset, eval, eval_owner, export, project_report
tests/         8 file, 96 test
.github/workflows/ci.yml   matrix ubuntu/macos/windows: uv sync + pytest
```

### 13.2. CLI

| Lệnh | Mục đích |
|---|---|
| `stylesync-data` | Ingest, clean |
| `stylesync-neutralize` | Neutralize |
| `stylesync-dataset` | Build dataset |
| `stylesync-train` | Train LoRA |
| `stylesync-eval` | Eval |
| `stylesync-export` | Export GGUF, parity |
| `stylesync-mcp` | MCP server |

### 13.3. Test

- **96 test** (`uv run pytest -q`), pass trên macOS, Windows và Linux (WSL).
- Gồm:
  - parser Messenger/Discord với fixture JSON giả;
  - mojibake, burst, PII;
  - validator neutralize;
  - split, embargo;
  - render template train;
  - metric eval, gate;
  - parity;
  - guard (prefix, trailer, backtick, mask, coverage, PII);
  - backend llama giả lập (spawn, fast-fail, truncated, adapter check).
- **Kh test nào dùng data thật.** Prompt và ví dụ trong code đều là câu giả.

---

## 14. Chi phí

| Hạng mục | Chi phí |
|---|---:|
| Phase 2 pilot v1 + v2 | $0.04 |
| Phase 2 full run | $1.06 |
| Phase 5 Baseline B + judge (cả 2 bộ) | $0.54 |
| **Tổng OpenAI** | **≈ $1.62** |
| GPU train/gen | $0 (RTX 3060 ở nhà) |

Giá Batch gpt-6-luna lúc chạy: $0.05 input / $0.25 output mỗi 1M token (xem 2026-10-27).

---

## 15. Riêng tư và bảo mật

- **`.gitignore` có từ commit đầu:** kh lên git
  - `data/`
  - export gốc (`your_facebook_activity/`, `package/`)
  - `configs/threads.yaml`
  - `runs/`, `models/`
  - output thô của spike
- **Kh public model/adapter**, vì train từ hội thoại riêng tư có cả tin của người khác.
- **Gửi ra ngoài:** chỉ text đã che PII, gửi OpenAI Batch với `store: false`, xoá file batch sau khi collect.
- **Report chỉ có số tổng hợp.** Ví dụ đẹp/xấu và sheet blind test nằm trong `data/eval/` (private).
- **Sec gate trước mỗi commit:**
  - regex secret (khoá API AWS/GitHub/Anthropic/OpenAI/Slack, PEM, password literal);
  - lọc file nhạy cảm trong `git ls-files` + toàn bộ history.
  - Lần gần nhất sạch.
  - `pip-audit` kh có trên máy → **skipped**.
- **Runtime:**
  - llama-server chỉ bind `127.0.0.1`;
  - chỉ tự start server khi URL là localhost;
  - guard chặn model tự thêm PII.

---

## 16. Quyết định owner đã chốt

| Quyết định | Chốt |
|---|---|
| Data GitHub | Kh dùng (commit/comment cũ do AI viết) |
| Tone | 1 giọng duy nhất, kh tag tone |
| Cutoff Discord | ≥ 2023-10-27; trước 2025-04-01 weight 0.5 |
| Neutralizer | OpenAI gpt-6-luna Batch (thay Claude Haiku, vì có credit) |
| Phase 2 | Chạy full v2 kh chấm lại spot-check |
| Judge + Baseline B | gpt-6-luna Batch; chờ batch kẹt thay vì gọi API thường |
| Ngưỡng "added" | Tỉ lệ của tin thật + 3 điểm % |
| Xưng hô bản trung tính | anh/em (kh tôi/bạn) |
| Model chạy thật | ft-gemma4-e2b-r16-e1 + che code span |
| Runtime | GGUF base Q4_K_M + LoRA f16 riêng qua llama-server |
| Public | Kh public model/adapter/data |
| Tích hợp | Claude Code (scope user + CLAUDE.md) và omp (mcp.json + skill + AGENTS.md) |

---

## 17. Chưa xong / not verified

- **CI 3 OS chưa chạy lần nào:** repo chưa có remote. Đã chạy pytest tay trên cả 3 OS, nhưng GitHub Actions thì chưa.
- **Blind test 68%:** chưa đạt ~50%. Hướng tiếp: thêm người chấm; thêm data owner tự viết (commit/comment); thử `gemma4-e2b-r16-e2` hoặc r32.
- **Claude Code interactive trên Windows:** chưa test. Mới test qua ssh, mà ở đó server chết khi phiên ssh đóng.
- **omp:** mới test với model mặc định gemini-3.8-flash. Model khác có tự đọc skill hay kh thì not verified.
- **Output GGUF chưa được judge chấm giữ ý.** Parity chỉ đo chrF/exact/guard.
- **Bản merge LoRA thành 1 file:** chưa làm (để khi chốt bản cuối).
- **Latency CPU x64:** đo qua e2e (~1.3s/lần trên PC có GPU), chưa đo riêng CPU-only trên Windows/Linux.
- **Commit `e0fdb52` có message tiếng Anh, chưa qua stylesync:** lúc đó tool chưa nạp trong session. Commit `3b1f3ce` thì đã qua stylesync.
- **CLI/git hook cho lúc commit tay** (plan mục 13.7, tuỳ chọn): chưa làm.
- **`pip-audit`:** chưa chạy (thiếu tool).

---

## 18. Cách dùng nhanh

```bash
# 1. copy model vào cache (cần models/gguf/ từ Phase 6, kh có trong git)
uv run stylesync-export install

# 2. Claude Code
claude mcp add --scope user stylesync -e STYLESYNC_LLAMA_SERVER=/opt/homebrew/bin/llama-server \
  -- /opt/homebrew/bin/uv run --directory /path/to/fine-tune-chat stylesync-mcp
# + thêm rule `## stylesync` vào ~/.claude/CLAUDE.md (README.md có mẫu)

# 3. omp
#    thêm entry stylesync vào ~/.omp/agent/mcp.json (README.md có mẫu), rồi:
ln -s "$PWD/integrations/omp/stylesync" ~/.agents/skills/stylesync
cat integrations/omp/AGENTS.md >> ~/.omp/agent/AGENTS.md

# 4. tắt backend
pkill -f "llama-server.*stylesync-lora"
```

Log của backend: `llama-server.log`, nằm cạnh model trong thư mục cache.
