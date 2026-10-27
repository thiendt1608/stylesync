# Style Rewriter: Fine-tune LLM local để viết đúng văn phong của mình

> Prj luyện tập AI: fine-tune một model nhỏ chạy local, nhận một đoạn văn trung tính và viết lại đúng giọng của mình.

---

## 1. Tổng quan

### Mục tiêu
- Model nhỏ (1.5B-3B) chạy local, **chỉ làm 1 việc: rewrite style**.
- Kh đọc code, kh đọc diff. Phần "ý" do LLM mạnh (API) lo, model nhỏ chỉ chỉnh giọng.
- Có thể đóng gói thành CLI / git hook để dùng thật.

### Kiến trúc chạy thật (two-stage pipeline)

```
[Ý / diff / ngữ cảnh]
        │
        ▼
 LLM mạnh (API)  ──►  bản trung tính (chuẩn, đủ ý)
        │
        ▼
 Model nhỏ local (đã fine-tune)  ──►  bản đúng giọng mình
```

### Gọi tên cho đúng (dùng khi viết README/CV)
| Thành phần | Tên kỹ thuật |
|---|---|
| Cách train | **SFT** (supervised fine-tuning) |
| Cách thực thi | **LoRA / QLoRA** (PEFT) |
| Bài toán | **Style transfer / text rewriting** |
| Tạo data | **Back-translation / synthetic data generation** |
| Kiến trúc chạy | **Two-stage pipeline** (LLM → local rewriter) |

> Lưu ý: đây là **pipeline**, kh phải distillation. Distillation là student học thay thế teacher; ở đây model nhỏ học **style của mình**, và model mạnh vẫn phải gọi mỗi lần dùng.

---

## 2. Kiến thức nền cần nắm

- **SFT** = *train cái gì*: học từ cặp (input → output mẫu).
- **LoRA** = *train bằng cách nào*: đóng băng weights gốc `W`, thêm 2 ma trận nhỏ `A`, `B` (rank `r`), output là `W·x + B·A·x`. Chỉ train A, B (~0.1-1% params).
- **QLoRA** = LoRA + load model gốc ở **4-bit (NF4)**. VRAM giảm khoảng 3-4 lần, chất lượng gần ngang LoRA thường.
- SFT và LoRA/QLoRA kh phải 2 lựa chọn ngang hàng. Làm **SFT, thực thi bằng LoRA hoặc QLoRA** (chọn 1).

---

## 3. Môi trường & công cụ

| Thứ | Chọn |
|---|---|
| GPU | RTX 3060 12GB (desktop) |
| Framework train | Unsloth (tiết kiệm VRAM, đổi LoRA ↔ QLoRA bằng `load_in_4bit`) |
| Base model ứng viên | Qwen2.5 / Qwen3 (1.5B-3B), Gemma 2B. Llama 3.2 tiếng Việt thường yếu hơn |
| LLM mạnh (neutralize) | Tuỳ chọn qua API |
| Deploy | GGUF + Ollama / llama.cpp |

**Chọn base model:** test nhanh vài chục câu tiếng Việt trên 2-3 model trước khi chốt, vì model nhỏ thường yếu tiếng Việt hơn tiếng Anh. Kết quả test này là *chưa verified*, phải tự chạy thử.

---

## 4. Thu thập dữ liệu

Nguyên tắc: chỉ lấy tin do **chính mình** viết. Chất lượng quan trọng hơn số lượng, vài trăm đến 1-2 nghìn mẫu **sạch** là bắt đầu thấy style rõ.

### 4.1. Nguồn data

| Nguồn | Cách lấy | Ghi chú |
|---|---|---|
| **Messenger** | Accounts Center → Your information and permissions → Download your information → Messages → format **JSON** | Chờ vài giờ đến vài ngày. Lỗi encoding tiếng Việt, xem 4.2 |
| **Discord** | User Settings → Privacy & Safety → Request all of my Data | Nhận file zip, có folder `messages/`. **Kh dùng self-bot/token** vì vi phạm ToS |

### 4.2. Fix encoding Messenger

Export JSON của Facebook hay bị mojibake (`\u00c3\u00a1...`). Decode lại:

```python
import json

def fix_fb(s: str) -> str:
    return s.encode("latin1").decode("utf-8")

with open("message_1.json", encoding="utf-8") as f:
    data = json.load(f)

ME = "Tên của bạn trên Facebook"
my_msgs = [
    fix_fb(m["content"])
    for m in data["messages"]
    if fix_fb(m.get("sender_name", "")) == ME and "content" in m
]
```

### 4.3. Lọc & làm sạch

- Chỉ giữ message có sender là mình.
- Bỏ tin quá ngắn ("ok", "ừ", "haha", "?") hoặc gộp các tin liên tiếp thành 1 đoạn.
- Bỏ link, sticker, file, tin hệ thống.
- Lọc thông tin nhạy cảm bằng regex (SĐT, email, số dài như STK, URL, token/API key, địa chỉ). Lý do: model có thể **memorize và nhả lại** lúc generate, nhất là khi định public adapter.
- Giọng nhắn tin theo từng đối tượng (bạn thân, người yêu, thầy cô) rất khác nhau. Chỉ chọn các cuộc trò chuyện đúng giọng muốn học, hoặc tag riêng.

```python
import re

PATTERNS = [
    r"\b\d{9,}\b",                       # số dài (SĐT, STK...)
    r"[\w\.-]+@[\w\.-]+\.\w+",            # email
    r"https?://\S+",                      # URL
    r"\b(sk|ghp|AKIA)[-_A-Za-z0-9]{10,}", # token dạng phổ biến
]

def is_clean(text: str) -> bool:
    return not any(re.search(p, text) for p in PATTERNS)
```

> Regex kh bắt hết đc. Nên đọc lướt mẫu ngẫu nhiên của tập cuối trước khi train.

---

## 5. Tạo cặp train (back-translation)

Chỉ có vế "văn của mình", thiếu vế "văn trung tính". Dùng LLM mạnh để **neutralize** ngược lại rồi ghép cặp.

**Ví dụ**

- Gốc (của mình): `fix lỗi login đc rồi, hồi nãy quên check null =))`
- Neutralize: `Đã sửa lỗi đăng nhập do thiếu kiểm tra giá trị null.`
- Cặp train: **input = bản trung tính**, **output = bản gốc của mình**.

**Prompt gợi ý cho LLM mạnh**

```
Viết lại câu sau thành phiên bản trung tính, chuẩn mực, rõ ràng.
Yêu cầu:
- Giữ NGUYÊN ý, KHÔNG thêm thông tin mới, KHÔNG bỏ ý.
- Cùng ngôn ngữ với câu gốc. Thuật ngữ kỹ thuật giữ tiếng Anh.
- Bỏ viết tắt, từ đệm, emoji, giọng điệu cá nhân.
Chỉ trả về câu đã viết lại.

Câu gốc: {text}
```

**Lưu ý**
- Ngôn ngữ neutralize nên **cùng ngôn ngữ với ngôn ngữ thật sự viết**. Dịch qua Anh rồi ngược lại sẽ mất sắc thái.
- Nếu model mạnh thêm ý, model nhỏ sẽ học bịa thêm. Nên spot-check mẫu.
- Lúc inference, input là bản trung tính do LLM mạnh viết, khớp phân phối lúc train.

### Format mẫu (JSONL)

```json
{"task": "message", "input": "Đã sửa lỗi đăng nhập do thiếu kiểm tra giá trị null.", "output": "fix lỗi login đc rồi, hồi nãy quên check null =))"}
```

Prompt template khi train:

```
<task: message>
Viết lại đoạn sau theo giọng của mình:
{input}
---
{output}
```

Chia **train / validation / test** (khoảng 80/10/10), test set kh đụng tới lúc train.

---

## 6. Fine-tune

### Bắt đầu với QLoRA

- 3B trên 3060 12GB khá sát VRAM, QLoRA an toàn hơn. 1.5B thì LoRA thường vẫn có thể chạy.
- Làm **cả 2 rồi so sánh** (LoRA vs QLoRA) là bài tập hay để thấy trade-off.

| | LoRA | QLoRA |
|---|---|---|
| VRAM | nhiều hơn | ít hơn |
| Tốc độ | nhanh hơn chút | chậm hơn chút |
| Chất lượng | tốt nhất | gần ngang |

### Hyperparameter khởi điểm (cần tune, chưa verified cho data của mình)

| Tham số | Giá trị thử |
|---|---|
| LoRA rank `r` | 16 (thử 8 / 32) |
| `lora_alpha` | 16-32 |
| Learning rate | ~2e-4 |
| Epochs | 2-3 |
| Max seq length | 256-512 (tin ngắn nên kh cần dài) |
| Batch size | nhỏ + gradient accumulation |

Hướng đi code (Unsloth + TRL `SFTTrainer`): load model với `load_in_4bit=True`, gắn LoRA bằng `FastLanguageModel.get_peft_model`, train bằng `SFTTrainer`. Làm theo notebook mẫu trong docs của Unsloth cho đúng API bản hiện tại, vì API có thể đổi.

### Dấu hiệu cần để ý
- Loss train giảm nhưng validation tăng → **overfit**, giảm epoch hoặc thêm data.
- Output y chang câu trong tập train → memorize, giảm epoch / thêm data đa dạng.

---

## 7. Đánh giá

Style khó đo tự động, nên kết hợp:

1. **Check "đúng ý"**: output có giữ nguyên nội dung input kh, có bỏ sót / bịa thêm kh.
2. **Check "đúng giọng"**: đọc bằng mắt so với văn thật.
3. **Blind test**: trộn output model với tin thật của mình, nhờ mình (hoặc bạn bè) đoán cái nào là thật.
4. **So sánh baseline**:
   - Baseline A: model gốc, kh fine-tune, prompt zero-shot.
   - Baseline B: LLM mạnh + few-shot vài ví dụ của mình.
   - Model fine-tune có thắng B kh? Nếu kh thì vẫn là kết quả đáng ghi vào README.
5. Metric phụ (tuỳ chọn): độ dài trung bình, tỉ lệ dùng từ viết tắt, tỉ lệ code-mixing Anh-Việt so với văn thật.

---

## 8. Deploy local

1. Merge adapter vào model gốc (hoặc giữ adapter riêng).
2. Convert sang **GGUF** (llama.cpp / export của Unsloth).
3. Chạy qua **Ollama** hoặc llama.cpp.
4. Bọc thành CLI:

```
$ git diff --staged | stylecommit
→ LLM mạnh viết bản trung tính → model local rewrite → in ra commit message
```

5. Tuỳ chọn: git hook `prepare-commit-msg` để tự gợi ý message.