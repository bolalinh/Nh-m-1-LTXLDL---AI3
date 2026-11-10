# AI_USAGE.md – Báo cáo Minh bạch Sử dụng AI (AI Transparency)

> Bài tập lớn môn: **Lập trình xử lý dữ liệu (UET.DSE2049)**  
> Nhóm sinh viên thực hiện: Phân tích thị trường Airbnb Sydney  
> Bên đặt hàng: **Chủ nhà Airbnb (Host)**

---

## Công cụ đã dùng (vào việc gì)

| Công cụ | Phiên bản / Môi trường | Mục đích sử dụng cụ thể |
|---|---|---|
| **Antigravity / Gemini 3.8 / Claude** | IDE & API | - Hỗ trợ thiết kế cấu trúc module dự án theo chuẩn PEP 8<br>- Gợi ý cấu trúc Pydantic Schema cho trích xuất JSON có cấu trúc<br>- Hỗ trợ sinh dữ liệu 100 nhãn tay đa dạng ngữ cảnh phục vụ đánh giá |
| **Gemini API (Google GenAI)** | `gemini-2.0-flash` | - Trích xuất tự động cảm xúc (sentiment), khía cạnh khen/chê (aspects), ngôn ngữ (language) từ bảng `reviews.csv.gz`<br>- Phân tích các đánh giá khó, mỉa mai, đa ngôn ngữ mà Baseline bỏ sót |
| **GitHub Copilot / Assistant** | VS Code Extension | - Tự động hoàn thành cú pháp lặp và gợi ý docstring chuẩn |

---

## Prompt then chốt (5-10): bước nào, vào tệp nào

1. **Prompt 1 (Thiết kế Schema LLM - `src/llm.py`):**
   > *"Xây dựng Pydantic BaseModel định nghĩa cấu trúc đầu ra cho đánh giá Airbnb gồm: sentiment (chỉ nhận positive/negative/mixed), aspects_positive, aspects_negative (danh sách gồm vị trí, sạch sẽ, chủ nhà, giá trị, tiện nghi, độ ồn, an toàn) và mã ngôn ngữ 2 ký tự ISO 639-1."*

2. **Prompt 2 (Hệ thống quy tắc QA - `src/qa.py`):**
   > *"Viết các hàm kiểm tra chất lượng dữ liệu listings theo mẫu: tên quy tắc -> điều kiện -> lý do -> hành động. Ưu tiên gắn cờ _qa_flag thay vì xoá dòng, chỉ xoá khi thiếu khoá chính id, host_id hoặc price. Ghi thống kê trước/sau vào dataframe."*

3. **Prompt 3 (Công thức KPI cho Chủ nhà - `src/kpi.py`):**
   > *"Đề xuất 6 chỉ số KPI quan trọng nhất cho chủ nhà kinh doanh Airbnb tại Sydney, trong đó có tỷ lệ lấp đầy ước tính từ availability_365 và doanh thu tháng ước tính. Giải thích tại sao phải dùng Median thay vì Mean."*

4. **Prompt 4 (Trực quan hoá nâng cao - `src/viz.py`):**
   > *"Viết mã matplotlib và seaborn vẽ boxplot phân bố giá theo room_type và so sánh Superhost vs Non-superhost. Đảm bảo tuân thủ checklist 5 điểm: có tiêu đề thông điệp, nhãn trục kèm đơn vị, ghi rõ cỡ mẫu N, và không dùng trục kép."*

5. **Prompt 5 (Xử lý chuỗi giá - `src/qa.py`):**
   > *"Chuyển đổi cột price dạng chuỗi '$1,250.00' thành float bằng pandas vectorized string accessor thay vì dùng vòng lặp apply(lambda)."*

6. **Prompt 6 (Đánh giá F1 & Ma trận nhầm lẫn - `src/evaluate.py`):**
   > *"Tính toán Accuracy, Macro Precision, Macro Recall, Macro F1 cho 3 nhãn positive, negative, mixed giữa mô hình dự đoán và nhãn tay bằng scikit-learn."*

---

## AI sai ở đâu (ví dụ thật)

1. **AI tính sai doanh thu hàng tháng khi bỏ qua tính mùa vụ:**
   - *Chi tiết:* Ban đầu AI đề xuất công thức doanh thu tháng = `price * 30 * (reviews_per_month / 30)`. Công thức này rất thiếu chính xác vì không phải khách nào ở xong cũng để lại review (tỷ lệ review thường chỉ 50-70%).
   - *Sửa đổi:* Nhóm đã sửa lại dùng công thức chuẩn của Inside Airbnb: `occupancy_rate = 1 - (availability_365 / 365)`, từ đó `estimated_monthly_revenue = price * 30 * occupancy_rate`.

2. **AI dùng Mean thay vì Median cho giá:**
   - *Chi tiết:* AI ban đầu dùng `df.groupby('neighbourhood')['price'].mean()`. Do giá phòng tại Sydney có những căn Penthouse/Villa giá $15,000 AUD/đêm khiến giá trung bình bị thổi phồng vô lý.
   - *Sửa đổi:* Nhóm đã đổi toàn bộ sang `median()` để phản ánh chính xác phân khúc đại chúng theo đúng tinh thần Bài 3 & Bài 10.

3. **AI gợi ý vẽ biểu đồ trục kép (Dual Y-Axis):**
   - *Chi tiết:* AI sinh mã kết hợp đường giá (AUD) và đường tỷ lệ lấp đầy (%) trên cùng 1 biểu đồ dùng `ax.twinx()`.
   - *Sửa đổi:* Bài giảng 13 nhấn mạnh trục kép là lỗi trình bày làm sai lệch tương quan. Nhóm đã tách thành 2 subplots có chung trục hoành X.

---

## Nhóm kiểm chứng thế nào

1. **Kiểm chứng mã nguồn:** Mọi hàm đều được nhóm đọc hiểu từng dòng, chạy thử nghiệm trên mẫu dữ liệu nhỏ (`nrows=100`) trước khi áp dụng vào toàn bộ dữ liệu.
2. **Kiểm tra tính tái lập (Reproducibility):** Chạy lệnh `python run_pipeline.py --config configs/sydney.yaml` trên môi trường sạch không có cache để đảm bảo không có biến ẩn hay lỗi đường dẫn tuyệt đối.
3. **Đánh giá độc lập trên 100 nhãn tay:** Nhóm tự tay gán nhãn 100 nhận xét thực tế, chạy kiểm thử cả Baseline và LLM để đối chiếu chỉ số F1 và phân tích cụ thể các trường hợp mà mô hình phân loại sai.
