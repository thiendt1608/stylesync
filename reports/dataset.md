# Phase 3: Dataset report

Ngày chạy: 2026-10-28. Lệnh: `uv run stylesync-dataset` (config `configs/dataset.yaml`).
File này chỉ có số tổng hợp, kh có nội dung tin nhắn hay tên người. Data thật nằm trong `data/processed/splits/` (gitignored).

## 1. Cách chia

- **Theo thời gian, riêng từng nguồn:** 10% burst mới nhất của Messenger và 10% mới nhất của Discord làm test, 10% kế tiếp làm val, còn lại làm train. Chia riêng từng nguồn để test có cả 2 nguồn. Nếu chia chung thì test gần như toàn Messenger, vì Messenger có data mới hơn.
- **Embargo 24h:** bỏ các burst nằm trong 24h trước mỗi ranh giới split, để một cuộc hội thoại kh bị cắt ngang nằm ở 2 split.
- **2 variant (formal, concise) của cùng một burst luôn ở cùng một split.**
- **Sample weight → số variant (chỉ train):** burst Discord trước 2025-04-01 có weight 0.5 nên chỉ giữ 1 trong 2 variant (chọn ngẫu nhiên, seed 7). Nhờ vậy kh cần trainer hỗ trợ sample weight. Val/test giữ đủ cả 2 variant.

## 2. Số lượng

| Split | Example | Burst | Messenger | Discord | Khoảng thời gian (Messenger / Discord) |
|---|---:|---:|---:|---:|---|
| train | 12,707 | 6,823 | 10,775 | 1,932 | 2025-10-27 → 2026-06-18 / 2023-10-28 → 2026-05-08 |
| val | 1,649 | 826 | 1,315 | 334 | 2026-06-19 → 2026-07-29 / 2026-05-09 → 2026-06-13 |
| test | 1,708 | 857 | 1,356 | 352 | 2026-07-30 → 2026-10-26 / 2026-06-16 → 2026-10-07 |

- Đầu vào: 17,038 pair từ 8,567 burst (Phase 2).
- Bị bỏ: 852 variant do weight 0.5, 122 example do embargo.
- Formal và concise gần bằng nhau ở mọi split (train 6,375 / 6,332).

## 3. Kiểm tra

- **Leak:** 0 bản gốc trùng nhau (sau chuẩn hoá) giữa train và val, cũng 0 giữa train và test.
- **Độ dài:**
  - Input (bản trung tính): p99 430–453 ký tự, max 560.
  - Output (bản gốc): p99 366–376 ký tự, max 400.
- **Token:** đếm bằng `llama-tokenize` trên 20 example dài nhất (system + input + output):
  - Qwen3.5-2B: max 381 token.
  - Gemma 4 E2B: max 388 token.
  - Cộng phần chat template vẫn dưới `max_seq_len` 512 của Phase 4, nên kh example nào bị cắt.

## 4. Format

Mỗi dòng là một example dạng conversational prompt-completion của TRL. Trainer tự áp chat template của base model (non-thinking) và chỉ tính loss trên phần completion:

```json
{"id": "...", "source": "messenger", "variant": "concise", "ts": "...",
 "prompt": [{"role": "system", "content": "<SYSTEM>"}, {"role": "user", "content": "<bản trung tính>"}],
 "completion": [{"role": "assistant", "content": "<bản gốc>"}]}
```

`SYSTEM` nằm ở `stylesync.dataset.build.SYSTEM`. Lúc inference (MCP server, Phase 7) phải gửi đúng system prompt này. MCP server hiện vẫn dùng prompt zero-shot của Phase 0.
