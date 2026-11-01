# Phase 1: Data report

Ngày chạy: 2026-10-27. Lệnh: `uv run stylesync-data` (config `configs/filter.yaml`), chạy ~6s trên M4.
File này chỉ có số tổng hợp, kh có nội dung tin nhắn hay tên người. Data thật nằm trong `data/` (gitignored).

## 1. Số lượng qua từng bước

| Bước | Messenger | Discord | Ghi chú |
|---|---:|---:|---|
| Tin của mình (ingest) | 57,526 | 82,748 | Messenger: tự nhận "me" (top sender + có mặt nhiều thread nhất, 2 điều kiện khớp). Discord: package chỉ chứa tin mình gửi |
| Bỏ rác | 55,381 | 77,107 | rỗng/chỉ attachment, tin hệ thống, chỉ link, chỉ emoji |
| Gộp burst (≤120s) | 10,737 | 23,021 | |
| PII → placeholder, độ dài 20–400 | 7,025 | 14,955 | |
| Bỏ nhiễu | 7,008 | 14,734 | code dán vào (Lua/SQL…), spam lặp từ, burst ≥40% placeholder |
| Dedup | 7,002 | 14,615 | chuẩn hoá lowercase/dấu câu/khoảng trắng, "kkkkk" = "kk" |
| Bỏ burst tiếng Anh | 6,986 | 13,406 | chủ yếu Discord 2024: chat với khách bằng tiếng Anh |
| Bỏ burst kh dấu (≥6 từ) | 6,951 | 10,901 | gõ kh dấu kiểu cũ; guard runtime cũng chặn output kiểu này |
| Cutoff thời gian | 6,951 | **1,818** | Discord ≥ 2023-10-27, Messenger lấy hết |
| Thread allowlist | **6,951** | **1,818** | chưa loại thread nào (xem mục 4) |

**Tổng: 8,769 mẫu** → `data/processed/mine.jsonl`. Độ dài trung bình 92.8 ký tự.

Ước lượng ở plan là ~11.3k (8.3k Messenger + 3.0k Discord). Thực tế ít hơn vì có thêm 3 bộ lọc (nhiễu, tiếng Anh, kh dấu) do review tay phát hiện.

Placeholder trong output: `<NAME>` 701, `<PHONE>` 15, `<EMAIL>` 15, `<NUMBER>` 10, `<URL>` 9, `<SECRET>` 3.

## 2. Style drift theo quý (chốt mốc Discord)

Đo trên burst đã lọc, trước khi cắt thời gian. Tỉ lệ viết tắt / có dấu / code-mixing tính trên số từ; emoticon tính trên số burst. Chart: [drift.svg](drift.svg) (mỗi đường scale theo max riêng).

**Messenger** (giọng hiện tại, đây là mốc so sánh):

| Quý | n | Dài TB | Viết tắt | Có dấu | Emoticon | Code-mix |
|---|---:|---:|---:|---:|---:|---:|
| 2025Q4 | 2028 | 89.9 | 0.105 | 0.660 | 0.027 | 0.050 |
| 2026Q1 | 1767 | 92.1 | 0.111 | 0.663 | 0.014 | 0.050 |
| 2026Q2 | 1932 | 97.2 | 0.102 | 0.656 | 0.013 | 0.066 |
| 2026Q3 | 1011 | 95.3 | 0.103 | 0.654 | 0.015 | 0.065 |
| 2026Q4 | 213 | 76.5 | 0.111 | 0.663 | 0.028 | 0.044 |

**Discord** (từ 2023Q3; quý có n < 30 kh đáng tin):

| Quý | n | Dài TB | Viết tắt | Có dấu | Emoticon | Code-mix |
|---|---:|---:|---:|---:|---:|---:|
| 2023Q3 | 889 | 83.5 | 0.091 | 0.566 | 0.171 | 0.100 |
| 2023Q4 | 485 | 93.4 | 0.099 | 0.608 | 0.136 | 0.075 |
| 2024Q1 | 14 | 109.5 | 0.068 | 0.575 | 0.143 | 0.123 |
| 2024Q2 | 8 | 103.8 | 0.126 | 0.498 | 0.000 | 0.227 |
| 2024Q3 | 48 | 66.2 | 0.094 | 0.499 | 0.083 | 0.136 |
| 2024Q4 | 273 | 79.0 | 0.093 | 0.605 | 0.044 | 0.117 |
| 2025Q1 | 114 | 66.8 | 0.110 | 0.571 | 0.044 | 0.094 |
| 2025Q2 | 99 | 75.0 | 0.108 | 0.607 | 0.010 | 0.091 |
| 2025Q3 | 56 | 108.6 | 0.100 | 0.610 | 0.071 | 0.100 |
| 2025Q4 | 68 | 86.1 | 0.105 | 0.571 | 0.015 | 0.083 |
| 2026Q1 | 270 | 109.1 | 0.110 | 0.595 | 0.015 | 0.106 |
| 2026Q2 | 324 | 104.0 | 0.109 | 0.616 | 0.015 | 0.080 |
| 2026Q3 | 106 | 91.4 | 0.093 | 0.574 | 0.009 | 0.117 |

Discord 2019–2022 (8,107 burst, kh dùng; quý n ≥ 30): viết tắt 0.04–0.08 (thấp hơn hẳn 0.10–0.11 bây giờ), có dấu 0.28–0.67, code-mix 0.10–0.16, emoticon 0.01–0.14.

**Kết luận:**
- Viết tắt và tỉ lệ có dấu của Discord từ 2023Q4 đã gần giọng hiện tại (0.09–0.11 và 0.57–0.62, so với Messenger 0.10–0.11 và 0.65–0.66). Giữ cutoff **2023-10-27**.
- Lệch rõ nhất là emoticon: 2023Q3–2024Q1 khoảng 0.14, gấp ~9 lần bây giờ (≤0.02). 2024Q3–2025Q1 vẫn cao gấp ~3 lần. Nếu train đều thì model dễ chêm emoji nhiều hơn m bây giờ.
- Xử lý: burst Discord trước **2025-04-01** có `weight = 0.5` (883 mẫu), từ 2025-04-01 `weight = 1.0` (935 mẫu). Messenger luôn 1.0. Cấu hình ở `configs/filter.yaml` (`full_weight_from`, `old_weight`).
- Lưu ý trước khi đọc bảng: trước khi có bộ lọc tiếng Anh và kh dấu, 2024Q3 có tỉ lệ có dấu chỉ 0.10 và code-mix 0.49. Cái lệch đó đến từ chat tiếng Anh/kh dấu, kh phải giọng thay đổi, nên đã xử lý bằng lọc thay vì dời mốc.

## 3. Review tay

Đã đọc 3 lượt × 200 mẫu (120 Messenger + 80 Discord, seed 7 / 11 / 23). Sau mỗi lượt, lỗi tìm thấy được biến thành rule và có test.

| Lượt | Mẫu lỗi | Lỗi tìm thấy → rule thêm vào |
|---|---|---|
| 1 | 5/200 = 2.5% (+ lỗi tìm qua audit riêng) | JWT/token dài lọt qua (→ `<SECRET>`); credential dạng `pw: …` (→ rule credential); code Lua; spam lặp; burst toàn `<NAME>`; câu tiếng Anh dài; scrubber che nhầm "AI", "Google", "FPT", "tiền", "gia đình"… (lấy từ tên page/shop) → chỉ dùng tên người, Title-case |
| 2 | 9/200 = 4.5% | SQL dán vào; 7 burst gõ kh dấu; 1 email kh có TLD (`…@gmail`) |
| 3 (cuối) | **1/200 = 0.5%** | 1 burst còn đuôi "(edited)" (đã strip) |

Ngưỡng plan là 10% → **đạt**.

**Discord authorship spot-check:** 80 mẫu Discord ở lượt 3 (và 80 ở mỗi lượt trước) đều đúng giọng m (xưng t/e/tui, kh có output bot hay tin người khác bị quote). Plan yêu cầu 50 mẫu.

**Hạn chế đã biết (kh phải rác, nhưng cần biết):**
- **Tên gọi viết thường còn sót.** Tên bạn bè gõ thường trong câu ("ê <tên> ơi") vẫn còn, khoảng 1/10 mẫu. Lý do: participant list có cả page/shop, nên rule bắt tên viết thường che nhầm từ thường rất nhiều. Đã thử và bỏ (vd "tiền" 915 lần, "gia đình" 17 lần). Rủi ro thấp:
  - model và adapter là private;
  - ở Phase 2 bản neutral sinh ra từ chính câu gốc nên cũng giữ tên đó, nên model học **copy** tên chứ kh tự bịa tên.
- **Burst toàn emoticon/tiếng lóng ngắn** dưới 20 ký tự đã bị loại theo plan. Giọng "kk", "dr", "oke" xuất hiện bên trong burst dài chứ kh đứng riêng.
- **Credential:** rule chỉ bắt dạng `pw:/mk là/pass =` + giá trị có chữ số. Mật khẩu toàn chữ cái sẽ kh bị bắt (not verified: kh còn trường hợp nào).

## 4. Tone: train 1 giọng

Thử gán tone casual/work theo từ khoá tên thread thì work chỉ có **8 / 8,769 mẫu (0.09%)**, toàn ở Discord. Oversample 8 mẫu thì vô nghĩa.

**Owner chốt (2026-10-27): train 1 giọng duy nhất, kh tag tone.**
- Pipeline kh còn `tone` / `work_keywords`. `configs/threads.yaml` chỉ còn `include` để loại thread; tag `tone` cũ tự bị xoá khi chạy lại.
- Commit/comment ở runtime dùng chung giọng này. Format cứng (prefix `type(scope):`, trailer, backtick) do server tách và guard kiểm. Đánh giá bằng bộ m tự gõ `data/eval/owner_commit_comment.jsonl` (plan 5.4).

Việc cho owner (kh chặn Phase 2): mở `configs/threads.yaml`, đặt `include: false` cho thread muốn bỏ (thầy cô/khách hàng/gia đình), rồi chạy lại `uv run stylesync-data`. Mặc định chưa loại thread nào.

## 5. Khác với plan

- Interim lưu `data/interim/messages.jsonl` + `bursts.jsonl` thay vì parquet, để khỏi thêm pyarrow/pandas. ~140k dòng đọc/ghi trong vài giây.
- Export đang để ở repo root (`your_facebook_activity/`, `package/`), kh phải `data/raw/`. Đường dẫn nằm trong `configs/filter.yaml`.
- Thêm 3 bộ lọc ngoài plan: nhiễu, tiếng Anh, kh dấu. Thêm `weight` theo thời gian. PII thêm JWT, token dài, credential, số 6–8 chữ số (OTP/MSSV).
- `thread_id` được hash (`m:<sha1[:10]>`, `d:<channel id>`) để file data kh mang tên thread.

## 6. Bảo mật

- Credential và token dán trong chat được che bằng rule (`<SECRET>`, rule credential) trước khi gửi đi neutralize hay đưa vào train.
- `data/`, `configs/threads.yaml` và export gốc đều gitignored. Report này và `drift.svg` chỉ có số tổng hợp.
