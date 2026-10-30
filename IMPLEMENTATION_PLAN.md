# Implementation Plan: Style Rewriter (local, MCP server)

> Plan triển khai cho ý tưởng trong [style-rewriter-plan.md](style-rewriter-plan.md).
> Ngày viết: 2026-10-27. Mọi số liệu về model/tooling lấy từ nguồn ở mục 14; số liệu về data đo trực tiếp trên export trong thư mục này. Cái gì chưa tự chạy thì ghi **not verified**.

---

## 0. TL;DR

- **Ý tưởng khả thi.** Tooling 2026 đã support đủ chuỗi: fine-tune LoRA → export **GGUF chạy llama.cpp** (Metal GPU trên Mac, CPU/CUDA trên Windows/Linux). ONNX int4 CPU là phương án phụ.
- **Cách dùng chính: MCP server.** Claude Code (LLM mạnh) tự viết bản trung tính rồi gọi tool `rewrite_style`, kh cần gọi Claude API riêng lúc chạy. CLI/git hook là phương án phụ cho lúc commit tay.
- **Base model đề xuất:** `Qwen3.5-2B` (chính), `Gemma 4 E2B` (đối chứng), `Qwen3.5-0.8B` (bản nhẹ nếu CPU chậm), `Qwen3.5-4B` (nếu 2B chưa đủ giọng).
- **4 chỗ phải sửa so với plan gốc:**
  1. Qwen3.5 **kh nên dùng QLoRA** (Unsloth khuyến cáo vì sai số quantize cao) → train **LoRA bf16**. 2B chỉ cần ~5GB VRAM nên 3060 dư sức.
  2. **Chỉ dùng data chat (Messenger/Discord).** Commit/comment GitHub cũ đều do AI viết nên kh phải giọng m → kh dùng. Train **1 giọng duy nhất** (kh tag casual/work, owner chốt 2026-10-27). Commit/comment dùng chung giọng đó, còn format cứng (prefix, trailer, backtick) do code giữ. Eval 2 task này bằng ~30–50 mẫu m tự gõ (5.4).
  3. Rewriter phải **giữ nguyên phần "cứng"**: prefix conventional commit (`feat(x):`), trailer `Co-Authored-By`, identifier/path/code trong backtick. Cần guard kiểm tra output, fail thì trả về bản trung tính.
  4. Lọc theo thời gian **bằng số đo drift**, kh đoán: tính metric văn phong theo quý rồi mới chốt mốc cắt (mặc định Discord lấy **36 tháng** gần nhất, Messenger lấy hết vì export chỉ có ~1 năm).
- **Rủi ro lớn nhất:** (a) model nhỏ bịa thêm hoặc làm mất ý, (b) agent quên gọi tool hoặc tự "sửa lại" output về văn chuẩn. (a) chặn bằng eval gate + guard, (b) chặn bằng mô tả tool + `CLAUDE.md` + hook (Phase 7).

---

## 1. Đánh giá khả thi

### 1.1. Data hiện có (đo trên export trong thư mục này)

| Nguồn | Tổng | Tin của mình | Sau khi gộp burst (≤120s, ≥20 ký tự) | Ghi chú |
|---|---|---|---|---|
| Messenger (`your_facebook_activity/messages/`) | ~164k tin, 2025–2026 | ~56k tin có text | **~8.3k mẫu** | 42 thread (inbox + archived + e2ee_cutover). Sender của mình = người có mặt ở 41/42 thread và gửi nhiều nhất |
| Discord (`package/Messages/`) | 82.7k tin, 2019–2026 | toàn bộ (file kh có field author) | **~3.0k mẫu** (từ 2023-10-27) | 16.5k tin trong 3 năm gần nhất: DM 11.4k, guild 4.6k, group DM 0.5k. (Nếu chỉ lấy 2 năm: 10.4k tin → ~1.8k mẫu) |

Phân bố Discord theo năm: 2019: 5.2k · 2020: 19.5k · 2021: 34.0k · 2022: 1.8k · 2023: 8.8k · 2024: 4.1k · 2025: 3.1k · 2026: 6.2k. Phần lớn tin cũ là 2020–2021, nên lọc thời gian là bắt buộc, đúng như m nghĩ.

Tin nhắn rất ngắn (Discord p50 = 15 ký tự, p90 = 38) → **gộp burst** (các tin liên tiếp cùng thread cách nhau ≤120s) là bước quan trọng nhất để có mẫu đủ dài.

→ **~11.3k mẫu thô** trước khi lọc chất lượng. Plan gốc nói vài trăm đến 1–2k mẫu sạch là thấy style, nên lượng data **đủ**. Not verified cho đến khi train thật.

> Discord: README của package nói folder `Messages/` chứa "các channel m đã gửi tin"; file `messages.json` kh có field author nên coi như toàn tin của m. Not verified 100%, Phase 1 spot-check 50 tin.

### 1.2. Model nhỏ chạy CPU, có tiếng Việt

Tính tới 2026-10: dòng Qwen mới hơn (3.6, 3.8) chỉ ra size lớn (27B trở lên), nên tier nhỏ vẫn là **Qwen3.5 Small** (3/2026) và **Gemma 4 E2B/E4B** (4/2026).

| Model | Size | Tiếng Việt | Fine-tune | ONNX (onnxruntime-genai) | GGUF (llama.cpp) | License |
|---|---|---|---|---|---|---|
| **Qwen3.5-2B** ⭐ | 2B, hybrid Gated DeltaNet + Attention, vocab 248k | "201 languages and dialects", kh nêu tên tiếng Việt. Qwen3 (đời trước) có bảng benchmark tiếng Việt riêng trong tech report | Unsloth bf16 LoRA ~5GB VRAM. MLX-LM có `qwen3_5.py` | Builder có class `Qwen3_5ForConditionalGeneration` | Converter có `Qwen3_5ForCausalLM` | Apache-2.0 |
| Qwen3.5-0.8B | 0.8B | như trên, yếu hơn | ~3GB VRAM | như trên | như trên | Apache-2.0 |
| Qwen3.5-4B | 4B | như trên, mạnh hơn | ~10GB VRAM (sát 3060 12GB) | như trên | như trên | Apache-2.0 |
| **Gemma 4 E2B** | 2.3B effective, 5.1B tính cả per-layer embeddings | "pre-trained on 140+ languages, out-of-the-box 35+", kh nêu tên tiếng Việt | Unsloth LoRA ~8–10GB VRAM. Có note bug KV-sharing (đã fix) | Builder có `Gemma4ForConditionalGeneration` | Có `Gemma4ForCausalLM` | Apache-2.0 |
| Qwen3-1.7B | 1.7B, transformer thuần | có benchmark tiếng Việt trong Qwen3 tech report | ổn định | `Qwen3ForCausalLM` (support lâu nhất) | có | Apache-2.0 |

**Chọn `Qwen3.5-2B` làm chính** vì: VRAM train nhỏ, CPU inference nhẹ hơn Gemma 4 E2B (E2B nặng ~5B tham số khi tính embeddings), cả 3 runtime đều support. **Gemma 4 E2B làm đối chứng**, vì dù chưa ai đo tiếng Việt head-to-head thì đây vẫn là đối thủ trực tiếp. **Qwen3-1.7B là phương án dự phòng** nếu export ONNX cho kiến trúc hybrid của Qwen3.5 bị lỗi.

> Chưa tìm được benchmark tiếng Việt nào so sánh trực tiếp Qwen3.5 với Gemma 4 ở size nhỏ → **Phase 0 tự đo** bằng bộ 30–50 câu của chính m. Đây là việc plan gốc đã đề xuất, giữ nguyên.

Các model "thuần Việt" (Vistral-7B, Arcee-VyLinh 3B, các bản fine-tune Qwen3.5/Gemma 4 tiếng Việt của cộng đồng trên HF) đều là 2024 hoặc fine-tune nhỏ lẻ, chưa có eval tin cậy → kh chọn làm base, có thể thêm vào bảng test Phase 0 nếu muốn.

### 1.3. Runtime: GGUF (chính) hay ONNX (phụ)?

| | llama.cpp / GGUF ⭐ | ONNX Runtime GenAI (v0.17.0, 2026-09-28) |
|---|---|---|
| Export | `convert_hf_to_gguf.py` + `llama-quantize`, hoặc `save_pretrained_gguf` của Unsloth | `onnxruntime_genai.models.builder -p int4 -e cpu`, nhận LoRA qua `adapter_path` |
| GPU trên Mac | **Metal**, bật mặc định | t chưa thấy hỗ trợ GPU Metal cho LLM, khả năng cao chỉ chạy CPU (not verified) |
| Windows/Linux | CPU, CUDA | CPU, CUDA, DirectML |
| LoRA | `--lora` hot-load adapter GGUF, khỏi merge lại mỗi lần train | Multi-LoRA ở runtime |
| Server dùng chung | `llama-server` (HTTP, OpenAI-compatible) | phải tự viết |
| Rủi ro | Trưởng thành nhất cho quant | Support Qwen3.5/Gemma 4 mới, dễ bug với kiến trúc hybrid |

**Quyết định:** runtime chính là **GGUF + `llama-server`**. Trên Mac chạy Metal: model 2B Q4 chiếm ~1.5–2GB unified memory, mỗi lần rewrite chỉ chạy GPU vài giây nên kh làm lag lúc code (not verified, Phase 0 đo). Phase 0 vẫn export ONNX để so, nếu ONNX CPU nhanh hơn rõ thì tính lại. Code viết theo interface `Rewriter` để đổi backend kh phải sửa MCP server/CLI.

MLX chạy GPU Mac nhanh, train LoRA được, nma chỉ có trên Mac → chỉ dùng để thử nghiệm, kh làm runtime chính.

### 1.4. Phần cứng

- Máy này: Apple M4, 24GB RAM. Desktop: RTX 3060 12GB (theo plan gốc).
- **Train trên 3060 (Unsloth) hoặc Colab**, kh train trên laptop (laptop là máy code, train là việc nặng). MLX-LM trên M4 chỉ dùng cho thử nghiệm siêu nhỏ, chạy `nice -n 19`.
- Inference đo ở Phase 0 trên M4 (Metal và CPU-only) và ít nhất 1 máy x64 chạy CPU.

### 1.5. Kết luận

**Ổn, làm được.** Điểm cần nhớ: giá trị thật của prj nằm ở **data pipeline + eval**, kh nằm ở bước train. Nếu cuối cùng Baseline B (Claude + few-shot tin thật của m) thắng model fine-tune, đó vẫn là kết quả hợp lệ và nên ghi vào README.

---

## 2. Kiến trúc

```
                       ┌─────────────── offline (1 lần) ────────────────┐
 Messenger JSON ─┐     │                                                 │
 Discord JSON ───┴─► ingest ─► filter/burst/scrub ─► neutralize (Claude) │
                       │        (local)                (batch API)       │
                       │                    │                            │
                       │                    ▼                            │
                       │          pairs.jsonl (neutral → mine)           │
                       │                    │                            │
                       │                    ▼                            │
                       │      LoRA SFT (3060/Colab) ─► eval gate         │
                       │                    │                            │
                       │                    ▼                            │
                       │      GGUF Q4_K_M (+ adapter)  (phụ: ONNX int4)  │
                       └─────────────────────────────────────────────────┘

 runtime (chính, qua agent):
   Claude Code tự viết bản trung tính
        │  gọi tool rewrite_style(text, task)
        ▼
   MCP server (stdio, lớp mỏng) ──HTTP localhost──► llama-server (1 process dùng chung, Metal/CPU)
        │
        ▼
      guard ── pass ─► text đúng giọng (agent dùng verbatim)
        └──── fail ─► trả bản trung tính + warning

 runtime (phụ, commit tay): git diff ─► Claude API (bản trung tính) ─► cùng core Rewriter ─► guard
```

Nguyên tắc: **cùng họ model lúc neutralize và lúc inference** (Claude). Khi chạy qua Claude Code thì chính Claude viết bản trung tính, nên input lúc chạy thật khớp phân phối input lúc train mà kh tốn thêm lần gọi API nào.

Vì sao tách `llama-server` riêng: mỗi session Claude Code spawn một MCP server stdio riêng. Nếu mỗi server tự load model thì mở 4 session là 4 bản model (~1.5–2GB mỗi bản). Một `llama-server` dùng chung chỉ load 1 lần.

---

## 3. Cấu trúc repo

```
fine-tune-chat/
├── data/                      # .gitignore toàn bộ
│   ├── raw/                   # export gốc (zip, your_facebook_activity/, package/)
│   ├── interim/               # messages.parquet sau ingest
│   └── processed/             # pairs.jsonl, splits/
├── configs/
│   ├── threads.yaml           # allowlist thread (include: true/false)
│   ├── filter.yaml            # cutoff date, burst gap, min/max len
│   └── train/*.yaml           # mỗi experiment một file
├── src/stylesync/
│   ├── ingest/{messenger,discord}.py
│   ├── clean/{filters,burst,pii,drift}.py
│   ├── neutralize/{prompt,batch,validate}.py
│   ├── train/sft.py
│   ├── eval/{style_clf,content,stylometry,blind}.py
│   ├── export/{onnx,gguf,parity}.py
│   ├── runtime/{rewriter,llama_server_backend,onnx_backend,guard}.py
│   ├── mcp_server.py          # FastMCP, tool rewrite_style + health
│   └── cli.py
├── hooks/prepare-commit-msg.py
├── tests/
├── IMPLEMENTATION_PLAN.md
├── style-rewriter-plan.md
├── CHANGELOG.md
└── pyproject.toml             # uv, Python 3.11+
```

Việc đầu tiên: `git init` và `.gitignore` (`data/`, `*.zip`, `your_facebook_activity/`, `package/`, `models/`, `.env`) **trước** khi commit bất cứ gì. Export Facebook 1.6GB và chat cá nhân tuyệt đối kh lên git.

---

## 4. Phase 0: Spike (chốt model + runtime)

Mục tiêu: trả lời 3 câu trước khi đầu tư data pipeline.

1. **Export chạy được kh?** Với `Qwen3.5-2B`, `Gemma 4 E2B`, `Qwen3.5-0.8B`, `Qwen3-1.7B`: export GGUF Q4_K_M (chính) + ONNX int4 CPU (để so).
2. **Nhanh cỡ nào?** Đo `llama-server` trên M4 Metal, M4 CPU-only và 1 máy x64 CPU (thêm ONNX CPU để so): thời gian load, prefill 150 token, decode 80 token, RAM peak.
3. **Tiếng Việt zero-shot ra sao?** 30–50 câu trung tính tiếng Việt (có lẫn thuật ngữ Anh) → prompt "viết lại thân mật, viết tắt" → chấm tay: đúng dấu, kh lẫn tiếng Trung, giữ ý.

4. **MCP chạy được kh?** Một MCP server FastMCP tối thiểu gọi `llama-server` với base model (chưa fine-tune), đăng ký vào Claude Code, xem Claude có gọi tool và dùng verbatim kết quả kh.

**Deliverable:** `reports/phase0.md` có bảng số đo.

**Gate:** chọn 1 model + 1 runtime. Gợi ý ngưỡng (owner chỉnh): rewrite 1 commit message ≤ 1.5s trên Metal và ≤ 3s trên CPU, RAM ≤ 3GB, 0 lỗi dấu/lẫn ngôn ngữ trên bộ test.

---

## 5. Phase 1: Data ingestion & cleaning

### 5.1. Ingest về schema chung

```python
# data/interim/messages.parquet
{source, thread_id, thread_type, ts (UTC), text, has_attachment}
```

- **Messenger:** đọc `inbox/`, `archived_threads/`, `e2ee_cutover/` (`message_*.json`). Fix mojibake bằng `s.encode("latin1").decode("utf-8")`, bọc try/except vì kh phải chuỗi nào cũng bị. Xác định "me" tự động: participant có mặt ở nhiều thread nhất **và** là top sender. Assert cả 2 điều kiện trùng nhau. Bỏ `ai_conversations.json`.
- **Discord:** `package/Messages/c*/{channel.json,messages.json}`. `Timestamp` dạng `YYYY-MM-DD HH:MM:SS`. Lấy `channel.type` làm `thread_type` (DM / GROUP_DM / GUILD_TEXT…).
- **GitHub: kh dùng** (xem 5.4).

### 5.2. Lọc

| Bước | Rule |
|---|---|
| Sender | chỉ tin của mình |
| Thời gian | `ts >= cutoff`. Discord mặc định **2023-10-27** (36 tháng). Messenger lấy hết (export chỉ 2025–2026). Cutoff để trong `configs/filter.yaml` theo từng source, chốt lại bằng 5.3 |
| Rác | bỏ tin chỉ có attachment/sticker, tin hệ thống ("đã gửi một file đính kèm", "reacted", "unsent"…), tin chỉ có link, tin chỉ có emoji |
| Burst | gộp tin liên tiếp cùng thread, cách nhau ≤ 120s, nối bằng `\n`. Giữ xuống dòng vì nó cũng là văn phong |
| Độ dài | burst ≥ 20 ký tự và ≤ 400 ký tự |
| Dedup | exact + near-dup (MinHash hoặc chuẩn hoá lowercase/khoảng trắng) |
| Thread | `configs/threads.yaml`: allowlist + tag giọng. Mặc định **loại** thread với thầy cô/khách hàng/gia đình nếu kh muốn học giọng đó |
| PII | regex (SĐT VN `0\d{9}`, `+84`, email, URL, STK ≥ 9 số, token `sk-`/`ghp_`/`AKIA`) + **tên người** lấy từ danh sách participant → thay bằng placeholder `<NAME>`, `<PHONE>`… |

> Thay PII bằng placeholder thay vì bỏ cả mẫu, để giữ được giọng mà model kh học thuộc tên/SĐT người khác. Placeholder kh bao giờ xuất hiện ở input lúc chạy thật, nên guard ở runtime chặn output có placeholder.

### 5.3. Đo style drift để chốt mốc thời gian

Tính theo **quý** trên tin của mình:

- độ dài burst trung bình,
- tỉ lệ từ viết tắt (`kh, k, ko, đc, dc, z, v, bt, nma, ng, mn, vd, thoi, dk…`),
- tỉ lệ có dấu tiếng Việt (gõ kh dấu hay có dấu),
- tỉ lệ emoji / `=))` / `:v`,
- tỉ lệ code-mixing Anh-Việt.

Vẽ 1 chart theo quý → chọn mốc từ chỗ các đường bắt đầu ổn định. Mặc định giữ 36 tháng cho Discord. Nếu quý cuối 2023 – đầu 2024 lệch hẳn so với 2025–2026 thì lùi mốc lên 2024-10-27 (~1.8k mẫu Discord) hoặc 2025-01-01 (~1.4k). Cách khác là giữ data cũ nhưng gán `sample_weight` thấp hơn để model vẫn ưu tiên giọng gần đây.

### 5.4. Commit / comment: kh có data train riêng

Commit và comment GitHub cũ của m đều do AI viết, kh phải giọng m → **bỏ khỏi data** (cả train lẫn test).

Hệ quả và cách xử lý:
- **Train chỉ có chat, 1 giọng.** Mọi mẫu đều là `task=message`, kh tag tone. Data work gần như kh có (8/8,769 mẫu), và owner kh cần giọng work riêng.
- **Runtime:** `rewrite_style(task=commit|comment)` dùng chung model/prompt với `message`. Phần cứng (prefix `type(scope):`, trailer, backtick) do server tách ra và guard kiểm, kh trông vào model (đã có từ Phase 0).
- **Eval riêng cho commit/comment:** m tự gõ ~30–50 mẫu theo đúng giọng m muốn (vd lấy 15 commit + 15 comment trung tính trong `spike/phase0/testset.jsonl` rồi viết lại theo giọng m) → `data/eval/owner_commit_comment.jsonl`. **Chỉ dùng để eval, kh train** (ít quá, train vào thì eval vô nghĩa).
- Nếu Phase 5 thấy giọng chat lên commit quá "chat" (dài, nhiều emoji, kh hợp commit) → mới tính tới: thêm vài trăm cặp commit/comment do m tự viết, hoặc rule hậu xử lý (bỏ emoji, giới hạn độ dài cho commit).

### 5.5. Review tay

Đọc ngẫu nhiên 200 mẫu sau lọc (120 Messenger, 80 Discord). Ghi tỉ lệ mẫu "rác/kh đúng giọng" vào `reports/data.md`. Tỉ lệ > 10% → siết rule rồi lọc lại.

**Deliverable:** `data/processed/mine.jsonl`, `reports/data.md` (số lượng từng bước, chart drift, kết quả review).

> **Trạng thái (2026-10-27): xong.** Có 8,769 mẫu (Messenger 6,951 + Discord 1,818). Review lượt cuối: 0.5% lỗi. Chi tiết ở `reports/data.md`. Code nằm ở `src/stylesync/{ingest,clean}/`, chạy bằng `uv run stylesync-data`.
>
> Khác plan:
> - interim là jsonl, kh dùng parquet;
> - thêm lọc nhiễu, lọc tiếng Anh, lọc burst kh dấu;
> - Discord trước 2025-04-01 có `weight = 0.5` vì emoticon cao hơn 3–9×;
> - tone `work` gần như kh có (8 mẫu) → owner chốt **train 1 giọng**, bỏ tag tone khỏi pipeline (xem `reports/data.md` mục 4).

---

## 6. Phase 2: Neutralize (back-translation)

- **Model:** OpenAI **gpt-6-luna** qua **Batch API** (`/v1/responses`, cửa sổ 24h), `reasoning.effort: low`. Owner đổi từ Claude Haiku 5.5 vì có sẵn credit OpenAI. Giá Batch lúc chạy: $0.05 input / $0.25 output mỗi 1M token ([pricing](https://developers.openai.com/api/docs/pricing), xem 2026-10-27); con số này nằm trong `configs/neutralize.yaml` và `stats.json` tính tiền thật từ usage.
- **Privacy:** chỉ gửi text đã scrub PII (5.2). Request để `store: false`; file input/output của batch bị xoá trên OpenAI sau khi collect ([data handling](https://developers.openai.com/api/docs/guides/your-data)). Key ở env var `OPENAI_API_KEY` (macOS: Keychain + `~/.zshrc`), kh ghi vào file trong repo.
- **Lệnh:** `uv run stylesync-neutralize run [--pilot]` (= submit + collect + build), `review --n 100` ra phiếu spot-check `data/neutral/review.md`.
- **Prompt** (`neutralize/prompt.py`, `prompt_version` nằm trong cache key):
  - Giữ nguyên code, identifier, path, tên lệnh, nội dung trong backtick, mọi con số, placeholder `<NAME>`...
  - Output JSON theo schema strict `{"formal": "...", "concise": "..."}`.
  - **v2** (sau spot-check v1: 7/100 cặp lệch nghĩa, chỉ ~45% từ tiếng Anh trong bản gốc còn giữ): thuật ngữ IT/công việc giữ tiếng Anh (push code, deploy, review...) vì lúc inference Claude cũng viết vậy; lóng thì viết lại đúng nghĩa, kh chắc thì giữ nguyên từ; t/m → tôi/bạn nhưng a/e/c giữ là anh/em/chị; kh tự thêm chủ ngữ, kh thêm câu hỏi đuôi; câu cảm thán kh thành mệnh lệnh; viết lại từng dòng, kh gộp dòng rời rạc. Test bắt mọi ví dụ trong prompt phải qua validator.
- **2 biến thể neutral cho mỗi mẫu** (1 trang trọng, 1 trung tính ngắn gọn). Data x2, và model kh bị overfit vào 1 kiểu câu của neutralizer.
- **Validate tự động** (`neutralize/validate.py`), bỏ mẫu nếu:
  - neutral rỗng hoặc == gốc (sau chuẩn hoá),
  - tỉ lệ độ dài neutral/gốc ngoài [0.5, 3],
  - mất token "cứng" (code, số, placeholder) so với gốc; số 0–10 viết thành chữ ("2 cái" → "hai cái") vẫn tính là giữ,
  - còn viết tắt trong danh sách 5.3, còn emoticon, hoặc mất dấu tiếng Việt,
  - model từ chối hoặc bị cắt vì hết token.
- **Spot-check tay 100 cặp.** Nếu > 5% cặp bị thêm/mất ý → sửa prompt, tăng `prompt_version`, pilot lại.
- Cache theo hash (model, prompt_version, text) để chạy lại kh tốn tiền; batch hết hạn thì lần submit sau gửi lại phần còn thiếu.

**Deliverable:** `data/processed/pairs.jsonl`:

```json
{"id": "...", "task": "message", "source": "messenger", "ts": "2026-05-01T10:00:00Z", "input": "<bản trung tính>", "output": "<bản gốc của mình>"}
```

---

## 7. Phase 3: Dataset build

- **Format chat** (dùng chat template của chính base model, **non-thinking**):
  - system: `Bạn là bộ viết lại văn phong. Giữ nguyên ý, code, identifier. Chỉ trả về bản viết lại.` (kh có tag task/tone vì train 1 giọng, chỉ có message, xem 5.4)
  - user: `{input}`
  - assistant: `{output}`
- **Train on completions only** (mask loss phần system/user).
- **Split theo thời gian, riêng từng nguồn**, kh random theo mẫu: 10% burst mới nhất của mỗi nguồn làm **test**, 10% kế tiếp làm **val**. Burst cùng 1 cuộc hội thoại rất giống nhau, split random sẽ leak và làm eval đẹp giả. Thêm **embargo 24h** trước mỗi ranh giới; 2 variant của 1 burst luôn cùng split.
- **Sample weight** (Discord cũ 0.5) làm bằng số variant giữ lại trong train: weight 0.5 → 1/2 variant. Kh cần trainer hỗ trợ weight.
- **Lệnh:** `uv run stylesync-dataset` → `data/processed/splits/{train,val,test}.jsonl` (TRL prompt-completion), `stats.json`. Kết quả: `reports/dataset.md` (train 12,707 / val 1,649 / test 1,708; max ~390 token < 512).

---

## 8. Phase 4: Fine-tune

- **Framework:** Unsloth + TRL `SFTTrainer` trên 3060 (hoặc Colab). Theo docs Unsloth, Qwen3.5 cần **transformers v5**.
- **LoRA bf16, kh QLoRA** (Unsloth khuyến cáo kh QLoRA cho Qwen3.5). So sánh LoRA vs QLoRA như plan gốc vẫn làm được với Gemma 4 E2B/Qwen3-1.7B nếu m muốn có bài so sánh trade-off.
- **Text-only:** Qwen3.5 là VLM, set `finetune_vision_layers=False`, chỉ train language layers.
- **Hyperparameter khởi điểm** (not verified cho data này):

| Tham số | Giá trị |
|---|---|
| `r` / `lora_alpha` | 16 / 16 (thử 8, 32) |
| target modules | attention + MLP (mặc định Unsloth) |
| LR | 2e-4, cosine, warmup 3% |
| Epochs | 2 (thử 1, 3) |
| Max seq len | 512 |
| Effective batch | 16 (batch nhỏ + grad accumulation) |
| Seed | cố định, log vào config |

- **Ma trận experiment tối thiểu:** {Qwen3.5-2B, Gemma 4 E2B} × {r=16} × {1, 2 epoch}, tức 4 run. Mở rộng sau khi thấy kết quả.
- **Tracking:** log loss train/val mỗi N step vào file CSV/JSON trong `runs/<exp>/` (kh cần W&B).
- **Dấu hiệu dừng:** val loss tăng 2 lần eval liên tiếp → early stop. Output trùng nguyên văn câu trong train → giảm epoch.

**Deliverable:** adapter `runs/<exp>/adapter/`, `runs/<exp>/metrics.json`.

**Đã làm (2026-10-29):**
- **Máy train:** desktop RTX 3060 12GB, Windows 11 + WSL2 Ubuntu 24.04 (distro nằm ở ổ E). Từ laptop vào qua Tailscale + OpenSSH (`ssh pc`), job chạy trong `tmux` nên tắt laptop kh sao.
  - WSL tắt distro khi kh còn `wsl.exe` nào mở, kéo theo tmux. Scheduled task `wsl-keepalive` trên Windows (chạy lúc khởi động, S4U, `wsl.exe -d Ubuntu-24.04 --exec sleep infinity`) giữ distro sống. Gỡ: `Unregister-ScheduledTask wsl-keepalive`.
- **Smoke test (20 step, 400 example):** pass. Loss 2.46 → 1.70, val loss 2.17 → 2.05, adapter + checkpoint + metrics.json ghi đủ. VRAM khoảng 5.4 GB ở batch 8 × grad accum 2.
- **Run `qwen35-2b-r16-e1`:** 795 step (1 epoch) trong 19.9 phút, kh early stop. Val loss 1.369 (step 100) → 1.089 (step 795), best ở step cuối và vẫn còn giảm → đáng chạy `qwen35-2b-r16-e2`.
- **Run `qwen35-2b-r16-e2`:** early stop ở step 1000/1590 (23.4 phút). Best val loss 1.085 ở step 800 (khoảng hết epoch 1), sang epoch 2 tăng lại 1.111 → 1.114, tức bắt đầu overfit. Chỉ hơn e1 0.004 → 1 epoch là đủ, chọn adapter nào để Phase 5 quyết theo eval thật.
- **Gemma 4 E2B:** kh gated, chỉ cần `hf auth login` (`~/stylesync/.venv/bin/hf`). Có 2 chỗ phải sửa mới chạy được:
  - Unsloth tự đổi repo sang `unsloth/gemma-4-E2B-it` → load với `use_exact_model_name=True`.
  - torch.compile của Unsloth làm backward đầu tiên lỗi (checkpoint recompute lệch graph, pytorch/pytorch#166926) → `no_compile: [gemma4-e2b]` trong config.
- **Run `gemma4-e2b-r16-e1`:** 795 step trong 31.3 phút, kh early stop. Val loss 1.164 (step 100) → 0.941 (step 795), best ở step cuối. Kh so thẳng được với val loss của Qwen vì 2 model khác tokenizer; so ở Phase 5. Chưa chạy `gemma4-e2b-r16-e2`.
- **Phase 6 cần để ý:** Unsloth báo template Gemma tạo double BOS và tự bỏ 1 cái lúc train. Parity test GGUF phải kiểm BOS.
- **Môi trường:** dependency group `train` chỉ cài trên Linux (`uv sync --group train`). Đã pin unsloth 2026.10.10, kéo theo torch 2.14.1 (CUDA 13.0), transformers 5.17.0, trl 1.15.0, peft 0.21.2.
- **Lệnh:**
  - `uv run --group train stylesync-train --exp qwen35-2b-r16-e1`
  - Thêm `--resume` để chạy tiếp từ checkpoint.
  - Thêm `--max-steps 20 --limit 400` để smoke test.
  - Experiment khai báo trong `configs/train.yaml`.
- **Format:** mỗi example được render bằng chat template của base model, non-thinking (`enable_thinking=False`), giống hệt cái MCP server gửi cho llama-server. Trainer nhận cặp prompt-completion dạng text và chỉ tính loss trên completion.
  - Qwen3.5: prompt kết thúc bằng `<|im_start|>assistant\n<think>\n\n</think>\n\n`, completion kết thúc bằng `<|im_end|>`.
  - Template prefix-stable trên cả 14,356 example train+val. Dài nhất 478 token (đã tính template), vẫn dưới 512.
- **Chọn best:** eval val mỗi 100 step, giữ checkpoint có val loss thấp nhất, early stop khi 2 lần eval liên tiếp kh tốt hơn.

---

## 9. Phase 5: Eval (gate trước khi làm MCP server)

Chạy trên **test set** (kh đụng lúc train), cho 4 hệ thống: Baseline A (base model zero-shot), Baseline B (Claude + 10 few-shot retrieve từ tin thật của m theo similarity), model fine-tune, và bản trung tính (đối chứng sàn).

Task commit/comment kh có trong test set chat → chạy thêm trên `data/eval/owner_commit_comment.jsonl` (bộ m tự gõ, 5.4) với cùng các metric, báo số riêng.

| Metric | Cách đo | Ngưỡng gợi ý |
|---|---|---|
| **Đúng giọng (tự động)** | Train 1 classifier nhỏ (char n-gram + logistic regression) phân biệt "tin thật của m" vs "bản trung tính" trên train set. Đo % output bị classify là "của m" | ≥ 70% |
| **Giữ ý** | LLM-judge (Claude) chấm 1–5 giữa input và output + cờ "thêm ý/mất ý" | trung bình ≥ 4.3, cờ bịa ≤ 3% |
| **Token cứng** | exact-match code/identifier/prefix conventional commit/trailer | 100% (fail thì guard chặn) |
| **Stylometry** | độ lệch độ dài, tỉ lệ viết tắt, emoji, code-mix so với phân phối thật (5.3) | càng gần càng tốt, báo cáo số |
| **Blind test** | 50 cặp (thật vs model) trộn ngẫu nhiên, m + 1–2 bạn đoán | tỉ lệ đoán đúng gần 50% là tốt |
| **Latency** | đo trên bản GGUF qua `llama-server` (Metal + CPU) | theo gate Phase 0 |

**Gate:** model fine-tune phải thắng Baseline A ở "đúng giọng" và kh thua Baseline B quá xa ở "giữ ý". Nếu thua B ở cả hai → ghi kết quả vào README, cân nhắc chỉ dùng B (kh cần model local).

**Deliverable:** `reports/eval.md` có bảng so sánh + 20 ví dụ đẹp/xấu.

---

## 10. Phase 6–7: Export & MCP server

### 10.1. Export

1. **GGUF (chính):** `save_pretrained_gguf(..., quantization_method="q4_k_m")` của Unsloth, hoặc `convert_hf_to_gguf.py` + `llama-quantize`. Có 2 cách đóng gói:
   - merge LoRA vào base rồi quantize (đơn giản, 1 file),
   - giữ base GGUF + adapter GGUF riêng (`llama-server --lora`), train lại chỉ cần thay adapter. Mặc định chọn cách này khi đang iterate, merge khi chốt bản.
2. **ONNX (phụ, để so ở Phase 0/5):** `python -m onnxruntime_genai.models.builder -m <merged_dir> -o models/onnx-int4 -p int4 -e cpu` (kiểm tra lại flag theo version đang dùng).
3. **Parity test:** 50 mẫu test, greedy decode, so output PyTorch vs GGUF (và ONNX nếu có) bằng chrF hoặc embedding similarity. Lệch nhiều → nghi chat template/EOS sai (theo docs Unsloth đây là lỗi hay gặp nhất).

### 10.2. Model server dùng chung: `llama-server`

- 1 process duy nhất, bind `127.0.0.1` (kh mở ra mạng), port cố định trong config.
- Mac: Metal mặc định. Windows/Linux: CPU, CUDA nếu có.
- Chat template + sampling (temperature thấp, ~0.3) cố định theo lúc train/eval.
- Khởi động: MCP server tự spawn `llama-server` nếu port chưa có ai nghe, và các session sau dùng lại. Kh viết launchd/systemd/service Windows cho bản đầu.
- Model file để ở `platformdirs.user_cache_dir("stylesync")`, kh hardcode `~` hay `/tmp`.

### 10.3. MCP server `stylesync` (cách dùng chính)

Python MCP SDK (FastMCP), transport **stdio**, là lớp mỏng gọi `llama-server` qua HTTP localhost.

**Tools:**

| Tool | Input | Output |
|---|---|---|
| `rewrite_style` | `text` (bản trung tính), `task`: `message` \| `commit` \| `comment` | `{text, guard_passed, warnings[]}` |
| `health` | – | model đang load, backend, latency lần gọi gần nhất |

**Mô tả tool** (agent đọc cái này để biết khi nào gọi và dùng ra sao):
- "Gọi trước khi viết commit message, PR/issue comment, hoặc tin nhắn thay owner. Input là bản trung tính, đủ ý."
- "Dùng **nguyên văn** `text` trả về, kh sửa chính tả, viết tắt hay giọng điệu. Nếu `guard_passed=false` thì dùng `text` (lúc này là bản trung tính) và báo owner."

**Guard** (`runtime/guard.py`, chạy trong server), fail thì trả bản trung tính kèm warning:
- giữ nguyên prefix `type(scope):` và mọi trailer (`Co-Authored-By`, `Signed-off-by`). Cách làm an toàn hơn: server tự tách prefix + trailer ra, chỉ đưa phần mô tả cho model rồi ráp lại,
- giữ nguyên mọi token trong backtick, path, identifier có trong input,
- kh có placeholder `<NAME>`/`<PHONE>` trong output,
- kh có PII pattern mới xuất hiện,
- tỉ lệ độ dài trong khoảng cho phép,
- `llama-server` timeout hoặc lỗi → trả bản trung tính, kh làm agent bị kẹt.

**Đăng ký vào Claude Code:** `claude mcp add stylesync -- uv run --directory <repo> stylesync-mcp` (scope user để mọi repo đều dùng được). Kiểm tra lại cú pháp theo version Claude Code đang dùng.

**Ép agent dùng tool**, làm theo thứ tự, bước sau chỉ làm khi bước trước chưa đủ:
1. Mô tả tool rõ ràng (như trên).
2. Thêm vào `~/.claude/CLAUDE.md`: "Trước `git commit`, `gh pr comment`, `gh issue comment` hoặc gửi tin thay t: gọi `stylesync.rewrite_style` và dùng verbatim kết quả."
3. Hook `PreToolUse` cho Bash: thấy `git commit`/`gh … comment` mà trong lượt đó chưa gọi `rewrite_style` thì chặn và nhắc. Cơ chế hook đọc lịch sử tool call **not verified**, chỉ làm nếu đo thấy agent hay quên.

**Human-in-the-loop:** server chỉ trả text, kh tự commit, kh tự post comment, kh tự gửi tin. Ghi vào `CLAUDE.md` để owner xem trước khi gửi tin nhắn cho ng thật.

### 10.4. CLI + git hook (phương án phụ)

Cho lúc m tự commit, kh có agent. Dùng chung core `Rewriter` + guard với MCP server.

```bash
stylesync rewrite --task message < neutral.txt
stylesync commit                 # đọc git diff --staged → Claude API viết bản trung tính → rewrite → in ra
```

- Chỉ đường này cần `ANTHROPIC_API_KEY` (env var, kh ghi vào file trong repo).
- Git hook `prepare-commit-msg` viết bằng **Python** (kh bash) để chạy được trên Windows/macOS/Linux. Có `STYLESYNC_SKIP=1` để tắt nhanh.
- Làm sau MCP server, có thể bỏ nếu thấy kh cần.

### 10.5. Test & CI

- Unit test: parser Messenger/Discord (fixture JSON giả, **kh dùng data thật**), mojibake fix, burst merge, PII scrub, guard (tách/ráp prefix + trailer).
- Integration test: backend giả (mock `llama-server`) để test MCP tools, CLI và hook kh cần model. Test MCP bằng client trong SDK (gọi `list_tools`, `call_tool`).
- GitHub Actions matrix `ubuntu-latest`, `macos-latest`, `windows-latest`: lint + unit + integration. Smoke test inference với Qwen3.5-0.8B GGUF chỉ chạy khi trigger tay (`workflow_dispatch`) vì tốn phút.
- `CHANGELOG.md` + conventional commits từ commit đầu.

---

## 11. Rủi ro & cách chặn

| Rủi ro | Khả năng | Cách chặn |
|---|---|---|
| Model nhỏ bịa/mất ý | Cao | Validate cặp ở Phase 2, LLM-judge ở Phase 5, guard + fallback ở runtime |
| GGUF/ONNX của Qwen3.5 (hybrid DeltaNet) lỗi hoặc lệch output | Trung bình | Phase 0 spike + parity test, fallback Qwen3-1.7B |
| Agent quên gọi `rewrite_style` | Trung bình | Mô tả tool + `CLAUDE.md`, thêm hook `PreToolUse` nếu đo thấy hay quên |
| Agent "sửa lại" output về văn chuẩn | Trung bình | Mô tả tool ghi rõ dùng verbatim, eval bằng cách so text trong commit với output tool |
| Nhiều session load nhiều bản model | Cao nếu mỗi MCP server tự load | 1 `llama-server` dùng chung, MCP server chỉ là lớp mỏng |
| Giọng chat ≠ giọng commit | Cao | Kh có data commit thật (đều do AI viết). Dùng chung 1 giọng chat, format cứng do code giữ, eval bằng bộ commit/comment m tự gõ (5.4), nếu kém thì thêm data tự viết |
| Văn phong drift theo năm | Trung bình | Đo drift (5.3) rồi chốt cutoff, split test theo thời gian |
| Lộ PII / memorize tên người | Trung bình | Scrub placeholder trước neutralize và train. **Kh public adapter/model** (train từ chat riêng tư của người khác) |
| Leak giữa train/test | Trung bình | Split theo thời gian + thread |
| Data riêng tư lọt lên git | Thấp nếu làm đúng | `.gitignore` trước commit đầu, chạy `/sec` trước mỗi commit |
| Chậm trên máy kh có GPU | Thấp–TB | `llama-server` giữ model warm, hạ xuống Qwen3.5-0.8B nếu cần |

---

## 12. Quyết định mở (đã chọn mặc định, owner đổi được)

| Quyết định | Mặc định | Lý do |
|---|---|---|
| Mốc thời gian | Discord 36 tháng (≥ 2023-10-27), Messenger lấy hết. Chốt lại sau đo drift | Owner chọn 3 năm cho Discord: data gần gấp đôi so với 2 năm (~3.0k vs ~1.8k mẫu) |
| Base model | Qwen3.5-2B, đối chứng Gemma 4 E2B | Xem 1.2 |
| Runtime | GGUF + `llama-server` (Metal trên Mac), ONNX để so | Có GPU trên Mac, trưởng thành nhất, server dùng chung sẵn |
| Cách dùng | MCP server (chính), CLI/git hook (phụ) | Claude tự viết bản trung tính, kh tốn thêm API call |
| Neutralizer (lúc tạo data) | OpenAI gpt-6-luna Batch API | Owner có sẵn credit OpenAI; full ~8.8k burst dự phóng < $1 |
| Thread nào học | Bạn bè + nhóm làm việc, loại thầy cô/khách/gia đình | **Chốt: 1 giọng, kh tag tone** (2026-10-27) |
| Public model | Kh | Data chứa hội thoại riêng tư |
| Data GitHub | Kh dùng | Commit/comment cũ đều do AI viết, kh phải giọng m |

---

## 13. Thứ tự làm

1. Phase 0 spike → `reports/phase0.md` → chốt model + runtime.
2. Phase 1 ingest + clean + drift → `reports/data.md` → chốt cutoff + thread.
3. Phase 2 neutralize (chạy thử 200 mẫu, spot-check, rồi mới chạy full).
4. Phase 3–4 build dataset + 4 run train.
5. Phase 5 eval → **gate**.
6. Phase 6–7 export GGUF + `llama-server` + MCP server + CI.
7. (Tuỳ chọn) CLI + git hook cho lúc commit tay.

---

## 14. Nguồn

- Qwen3.5-2B model card (kiến trúc hybrid Gated DeltaNet, 201 ngôn ngữ, Apache-2.0, non-thinking mặc định): <https://huggingface.co/Qwen/Qwen3.5-2B>
- Qwen3.5 Small release (3/2026): <https://artificialanalysis.ai/articles/qwen3-5-small-models>, <https://gigazine.net/gsc_news/en/20260303-qwen-3-5-small>
- Qwen3 Technical Report (bảng benchmark tiếng Việt, Table 33): <https://arxiv.org/pdf/2505.09388>
- Gemma 4 E2B model card (2.3B effective / 5.1B, 140+ ngôn ngữ, 128K): <https://huggingface.co/google/gemma-4-E2B-it>
- Gemma 4 release 2026-04-02, Apache 2.0: <https://rits.shanghai.nyu.edu/ai/google-releases-gemma-4-frontier-open-models-under-apache-2-0/>
- Unsloth Qwen3.5 fine-tune (VRAM, kh QLoRA, transformers v5, GGUF export): <https://unsloth.ai/docs/models/qwen3.5/fine-tune>
- Unsloth Gemma 4 train (E2B 8–10GB): <https://unsloth.ai/docs/jp/moderu/gemma-4/train>
- ONNX Runtime GenAI (v0.17.0, 2026-09-28; builder có `Qwen3_5ForConditionalGeneration`, `Gemma4ForConditionalGeneration`, `adapter_path`): <https://github.com/microsoft/onnxruntime-genai>, <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md>
- llama.cpp converter (`conversion/qwen.py`, `conversion/gemma.py` có Qwen3.5 / Gemma4): <https://github.com/ggml-org/llama.cpp/tree/master/conversion>
- Claude Code MCP: <https://docs.claude.com/en/docs/claude-code/mcp>
- MCP Python SDK (FastMCP): <https://github.com/modelcontextprotocol/python-sdk>
- llama.cpp `llama-server`: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>
- MLX-LM (v0.32.2, có `qwen3_5.py`, `gemma4_text.py`): <https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models>
- Ví dụ cộng đồng LoRA Qwen3.5-0.8B tiếng Việt bằng MLX: <https://huggingface.co/baeGil/Qwen3.5-0.8B-mlx-LoRA>
- Discord data package: `package/README.txt` trong export, <https://support.discord.com/hc/articles/360004957991>
