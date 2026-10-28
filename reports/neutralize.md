# Phase 2: Neutralize report

Ngày chạy: 2026-10-28. Lệnh: `uv run stylesync-neutralize run` (config `configs/neutralize.yaml`, prompt v2).
File này chỉ có số tổng hợp, kh có nội dung tin nhắn hay tên người. Data thật nằm trong `data/` (gitignored).

## 1. Provider

- OpenAI Batch API, model `gpt-6-luna`, `reasoning.effort: low`, output JSON schema strict (`formal`, `concise`).
- Request `store: false`; file input/output/error của batch bị xoá trên OpenAI sau khi collect.
- Giá Batch: $0.05 input / $0.005 cached input / $0.25 output mỗi 1M token ([pricing](https://developers.openai.com/api/docs/pricing), xem 2026-10-27).

## 2. Pilot 200 mẫu (120 Messenger + 80 Discord, seed 7)

| | v1 | v2 |
|---|---:|---:|
| Pair qua validator (trên 400) | 383 | 383 |
| Spot-check tay: cặp lệch nghĩa / 100 | **7** (6 burst) | kh chấm (owner chốt chạy full) |
| Từ tiếng Anh trong bản gốc còn giữ (formal, heuristic) | 43% | 59% |
| Chi phí | $0.0187 | $0.0242 |

- Lỗi v1: đổi a/e thành tôi/bạn rồi nhầm người nói, tự thêm chủ ngữ, đoán sai lóng (câu cảm thán thành mệnh lệnh), gộp các dòng rời rạc. Prompt v1 còn dịch thuật ngữ ("push code" → "đẩy mã nguồn"), lệch với input lúc inference (Claude viết thuật ngữ bằng tiếng Anh).
- v2 sửa đúng 4/6 burst bị đánh dấu ở v1; còn sót: lóng kh chắc nghĩa được giữ nguyên từ, và câu chửi kèm cảm thán vẫn bị viết thành mệnh lệnh.
- Tỉ lệ "giữ tiếng Anh" đo bằng script tạm: đếm từ ASCII ≥3 chữ kh thuộc danh sách viết tắt. Script còn đếm nhầm vài viết tắt tiếng Việt và lóng nên bị thiếu, chỉ dùng để so v1 với v2.
- Validator v1 ban đầu loại 52 cặp vì số nhỏ viết bằng chữ ("2" → "hai"); sau khi cho phép 0–10 viết bằng chữ thì còn 13.

## 3. Full run (prompt v2)

| | Số |
|---|---:|
| Burst đầu vào | 8,769 (Messenger 6,951, Discord 1,818) |
| Request gửi mới (200 burst pilot v2 lấy từ cache) | 8,569 |
| Request completed / lỗi / refusal / bị cắt | 8,769 / 0 / 0 / 0 |
| **Pair qua validator** | **17,038** (formal 8,550, concise 8,488) |
| Pair theo nguồn | Messenger 13,520, Discord 3,518 |
| Burst còn ≥1 pair | 8,567 / 8,769 (97.7%) |
| Variant bị loại | 500 (2.9%) |

Lý do loại (một variant có thể bị nhiều lý do):

| Lý do | Số |
|---|---:|
| number (mất số) | 206 |
| unchanged (gần như y bản gốc, thường là bản gốc đã trung tính sẵn) | 188 |
| abbrev (còn viết tắt) | 60 |
| len_ratio | 21 |
| placeholder | 12 |
| code | 6 |
| emoticon | 6 |
| empty | 4 |

Token: input 8,740,948 (cached 2,080), output 2,499,634 (trong đó reasoning 1,650,167).
**Chi phí thật full v2: $1.06.** Tổng cả Phase 2 (pilot v1 + pilot v2 + full): $1.08.

Thời gian: batch full tạo lúc 11:15, xong trước 15:36 (giờ VN), tức ~4.3 giờ. Riêng 113 request cuối đứng yên ít nhất ~1.5 giờ (theo log poll 5 phút).

## 4. Deliverable

`data/processed/pairs.jsonl`: mỗi dòng `{id, task, source, thread_id, ts, weight, variant, input: <bản trung tính>, output: <bản gốc>}`.
