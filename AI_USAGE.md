# AI_USAGE.md – Báo cáo Minh bạch Sử dụng AI (AI Transparency)

> **Học phần:** Lập trình Xử lý Dữ liệu (UET.DSE2049) – HK1, 2026–2027  
> **Nhóm thực hiện:** Nhóm 1 – Lớp AI 3  
> **Đề tài:** Phân tích Dữ liệu Inside Airbnb Sydney (4 mốc chụp: 09/2025, 12/2025, 03/2026, 06/2026)  
> **Bên đặt hàng (Client Persona):** Hiệp hội Chủ nhà (Host Community) & Nhà Đầu tư Bất động sản Sydney  
> **Mức độ sử dụng AI ước tính:** **~40%** (AI đóng vai trò trợ lý hỗ trợ kỹ thuật; 60% do nhóm sinh viên tự chủ thiết kế, quyết định nghiệp vụ, kiểm chứng và bảo vệ).

---

## 1. Mức độ Sử dụng AI & Phân định Trách nhiệm (Tỷ lệ ~40% AI / 60% Nhóm)

Nhóm tuân thủ nghiêm ngặt Chính sách AI của môn học (Được sử dụng AI hỗ trợ ở giai đoạn làm bài ở nhà, nhưng phải hiểu sâu 100% để bảo vệ độc lập trong vòng vấn đáp ~20 phút không dùng AI).

| Hạng mục công việc | Mức độ AI hỗ trợ | Mô tả vai trò của AI (~40%) | Trách nhiệm tự chủ của Nhóm sinh viên (~60%) |
|---|:---:|---|---|
| **1. Khảo sát & Đọc dữ liệu** | 25% | Gợi ý cú pháp `argparse`, hàm đọc YAML lồng và nạp dữ liệu phân khúc (chunking). | Tự tìm hiểu cấu trúc dữ liệu Inside Airbnb, lựa chọn 4 mốc chụp phù hợp 4 mùa tại Sydney. |
| **2. Kiểm soát chất lượng (QA)** | 35% | Gợi ý cú pháp vector hoá Pandas (regex bóc tách giá tiền, lọc outlier theo IQR). | Tự đề xuất logic 8 quy tắc QA (tên → điều kiện → lý do → hành động), quyết định chính sách QA-00 dùng giá mốc cuối để đồng nhất thị trường. |
| **3. Bộ chỉ số KPI cho Chủ nhà** | 30% | Gợi ý các hàm tính toán thống kê (Median, Groupby theo khu vực và loại phòng). | Tự định nghĩa 6 KPI kinh doanh cốt lõi (Median Price, Occupancy Rate từ `availability_365`, Monthly Revenue, Superhost Premium, Cleanliness Score). |
| **4. Mô hình LLM & Đối chứng** | 60% | Cung cấp API `gemini-2.0-flash` / `gemini-3.5-flash-lite` để phân tích cảm xúc và trích xuất khía cạnh từ văn bản đánh giá. | Tự thiết kế Pydantic Schema, cơ chế Caching JSONL, logic Retry/Backoff khi gặp Rate Limit (429), và xây dựng mô hình luật Baseline độc lập. |
| **5. Đánh giá & Phân tích lỗi** | 30% | Hỗ trợ tính ma trận chỉ số scikit-learn (Accuracy, Macro Precision, Recall, Macro F1, Jaccard). | **100% nhóm tự tay gán nhãn 100 nhận xét thực tế** (Gold Standard), trực tiếp phân tích nguyên nhân Baseline thất bại và đối chiếu kết quả. |
| **6. Trực quan hoá & Báo cáo** | 40% | Hỗ trợ template Matplotlib/Seaborn theo checklist 5 điểm của Bài 12-13. | Tự lựa chọn góc nhìn phân tích (Thời gian, Không gian, Phân khúc, Khía cạnh LLM), diễn giải ý nghĩa kinh tế và đề xuất chiến lược cho chủ nhà. |

---

## 2. Công cụ Đã Dùng (Vào Việc Gì)

| Công cụ | Môi trường / Phiên bản | Mục đích sử dụng cụ thể trong dự án |
|---|---|---|
| **Google Gemini API** (`google-genai`) | `gemini-2.0-flash` / `gemini-3.5-flash-lite` | Trích xuất có cấu trúc từ trường đánh giá tự do (`comments`): Phân loại cảm xúc (`positive`, `negative`, `mixed`), trích xuất khía cạnh khen/chê (`aspects`), và phát hiện ngôn ngữ (`language`). |
| **Antigravity / Gemini CLI** | Môi trường lập trình AI Agent | Hỗ trợ rà soát cấu trúc thư mục chuẩn PEP 8, debug xung đột môi trường Python 3.14, và gợi ý xử lý phòng thủ cho file cấu hình YAML. |
| **GitHub Copilot / Assistant** | VS Code Extension | Tự động hoàn thiện cú pháp lặp DataFrame, gợi ý type hint và docstring chuẩn theo phong cách Google. |

---

## 3. Prompt Then Chốt (5 – 10 Prompt)

Dưới đây là 8 prompt quan trọng nhất thể hiện sự tương tác kỹ thuật giữa nhóm và AI trong quá trình xây dựng mã nguồn:

### Prompt 1: Chuẩn hoá cấu hình YAML phòng thủ
* **Bước:** [1/6] Khởi tạo & nạp cấu hình (`run_pipeline.py`)
* **Tệp tác động:** `run_pipeline.py` & `configs/sydney.yaml`
* **Nội dung Prompt:**
  > *"File configs/sydney.yaml định nghĩa trường `city` dạng dict lồng `{name: 'Sydney', country: 'Australia'}` và `data_root` dạng dict. Hãy viết hàm `load_config()` trong `run_pipeline.py` bóc tách phòng thủ để `cfg['city']` luôn trả về chuỗi 'Sydney', tránh lỗi `AttributeError: 'dict' object has no attribute 'upper'` khi gọi hàm chuyển chữ hoa."*

### Prompt 2: Vector hoá hàm làm sạch cột giá tiền
* **Bước:** [2/6] Kiểm soát chất lượng dữ liệu (`src/qa.py`)
* **Tệp tác động:** `src/qa.py`
* **Nội dung Prompt:**
  > *"Viết hàm `clean_price_series(series)` trong pandas sử dụng string accessor vector hoá để chuyển cột price từ chuỗi chứa ký tự '$' và dấu phẩy ',' thành float, xử lý các giá trị rỗng/nan/None thành np.nan mà không dùng vòng lặp apply(lambda) nhằm tối ưu hiệu năng trên 72,000 dòng."*

### Prompt 3: Thiết kế Pydantic Schema cho LLM trích xuất JSON
* **Bước:** [4/6] Khai phá dữ liệu văn bản phi cấu trúc (`src/llm.py`)
* **Tệp tác động:** `src/llm.py`
* **Nội dung Prompt:**
  > *"Sử dụng Pydantic v2 xây dựng BaseModel `ReviewAnalysis` với các trường: `sentiment` (Literal['positive', 'negative', 'mixed']), `aspects_positive` (danh sách thuộc ['location', 'cleanliness', 'host', 'value', 'amenities', 'noise', 'safety']), `aspects_negative` (danh sách cùng nhóm thuộc tính), và `language` (chuỗi mã 2 ký tự ISO 639-1) để truyền vào response_schema của Google GenAI API."*

### Prompt 4: Cơ chế phòng vệ Rate Limit & Caching cho Gemini API
* **Bước:** [4/6] Xử lý giới hạn gói miễn phí (`src/llm.py`)
* **Tệp tác động:** `src/llm.py`
* **Nội dung Prompt:**
  > *"Gói Gemini Free Tier có giới hạn 15 requests/phút (RPM). Hãy viết cơ chế lưu cache kết quả vào `data/processed/llm_cache.jsonl` và thuật toán Exponential Backoff khi gặp mã lỗi 429 RESOURCE_EXHAUSTED hoặc 503. Nếu thử lại thất bại, tự động fallback sang phân loại bằng mô hình luật Baseline mà không làm gián đoạn pipeline."*

### Prompt 5: Xây dựng mô hình Baseline đối chứng không dùng LLM
* **Bước:** [4/6] & [5/6] Mô hình đối chứng (`src/baseline.py`)
* **Tệp tác động:** `src/baseline.py`
* **Nội dung Prompt:**
  > *"Xây dựng module `classify_review_baseline(text)` phân loại cảm xúc bằng phương pháp đếm từ điển (Lexicon rule-based): tổng hợp danh sách từ khóa tích cực/tiêu cực, nhận diện liên từ đảo chiều (but, however, yet) để gán nhãn 'mixed', và so sánh độ tương quan từ khóa để gán nhãn khía cạnh tương ứng."*

### Prompt 6: Đánh giá Macro F1 và Jaccard Similarity trên nhãn vàng
* **Bước:** [5/6] Đánh giá đối chứng (`src/evaluate.py`)
* **Tệp tác động:** `src/evaluate.py`
* **Nội dung Prompt:**
  > *"Viết module đánh giá đối chứng `evaluate_predictions()` đọc 100 nhãn vàng từ file mẫu: tính Accuracy, Macro Precision, Macro Recall, Macro F1 (cho 3 nhãn positive, negative, mixed) và Jaccard Similarity cho tập hợp các aspects giữa nhãn thực tế và dự đoán của cả Baseline lẫn LLM."*

### Prompt 7: Công thức KPI chuẩn hoá cho Chủ nhà Airbnb
* **Bước:** [3/6] Tính toán chỉ số kinh doanh (`src/kpi.py`)
* **Tệp tác động:** `src/kpi.py`
* **Nội dung Prompt:**
  > *"Đề xuất công thức tính: 1) Tỷ lệ lấp đầy ước tính từ availability_365; 2) Doanh thu tháng ước tính = Price * (Occupancy_Rate / 100) * 30.42; 3) Tỷ lệ chênh giá Superhost Premium = (Median_Price_Superhost - Median_Price_Regular) / Median_Price_Regular * 100%. Giải thích vì sao phải dùng Median thay vì Mean."*

### Prompt 8: Trực quan hoá tuân thủ Checklist 5 điểm (Bài giảng 12 & 13)
* **Bước:** [6/6] Trực quan hoá (`src/viz.py`)
* **Tệp tác động:** `src/viz.py`
* **Nội dung Prompt:**
  > *"Viết hàm vẽ biểu đồ thời gian thể hiện Doanh thu tháng và Tỷ lệ lấp đầy qua 4 mốc chụp bằng 2 đồ thị con (subplots) có chung trục hoành X, tuyệt đối không dùng trục kép (dual Y-axis) theo đúng cảnh báo của Bài 13. Đảm bảo tiêu đề mang thông điệp kinh doanh, nhãn trục ghi rõ đơn vị AUD và %, hiển thị nhãn giá trị tại các điểm dữ liệu."*

---

## 4. AI Sai Ở Đâu (Các Ví Dụ Thực Tế Trong Quá Trình Phát Triển)

Trong quá trình thực hiện, nhóm phát hiện AI đã đưa ra một số gợi ý sai lầm hoặc chưa tối ưu. Nhóm đã kịp thời phát hiện, phản biện và sửa đổi:

### Ví dụ 1: AI tính sai doanh thu hàng tháng do phụ thuộc `reviews_per_month`
* **Sai lầm của AI:** Ban đầu AI gợi ý công thức: $\text{Doanh thu} = \text{price} \times 30 \times (\text{reviews\_per\_month} / 30)$.
* **Phân tích của nhóm:** Công thức này sai lệch nghiêm trọng vì chỉ có khoảng 50% – 70% khách lưu trú thực tế để lại đánh giá trên Airbnb. Dùng số review để suy ngược doanh thu sẽ đánh giá thấp (underestimate) dòng tiền của chủ nhà tới 40%.
* **Cách khắc phục:** Nhóm đã tra cứu tài liệu chuẩn của Inside Airbnb và thay thế bằng công thức dựa trên số ngày trống thực tế:
  $$\text{Occupancy Rate} = \frac{365 - \text{availability\_365}}{365} \times 100\% \quad \longrightarrow \quad \text{Revenue} = \text{Price} \times \left(\frac{\text{Occupancy Rate}}{100}\right) \times 30.42$$

### Ví dụ 2: AI đề xuất dùng Mean Price gây sai lệch do ngoại lai (Outliers)
* **Sai lầm của AI:** Khi phân tích giá theo khu vực, AI viết `df.groupby('neighbourhood')['price'].mean()`.
* **Phân tích của nhóm:** Dữ liệu Sydney có những căn Penthouse hướng vịnh hoặc biệt thự siêu sang niêm yết giá $8,000 – $15,000 AUD/đêm (phân vị 99%). Giá trị trung bình (Mean) bị kéo lệch nghiêm trọng (Mean toàn thành phố lên tới $417 AUD, trong khi Median thực tế chỉ là $310 – $326 AUD).
* **Cách khắc phục:** Vận dụng nguyên lý Thống kê vững (Robust Statistics) từ Bài giảng 3 và Bài giảng 10, nhóm chuyển toàn bộ chỉ số định giá cốt lõi sang giá trung vị `median()`, chỉ giữ `mean()` làm số liệu tham khảo thứ cấp.

### Ví dụ 3: AI gợi ý vẽ biểu đồ trục kép (Dual Y-Axis)
* **Sai lầm của AI:** AI sinh mã Matplotlib sử dụng `ax.twinx()` để vẽ đường Doanh thu (AUD) và đường Lấp đầy (%) trên cùng một đồ thị.
* **Phân tích của nhóm:** Bài giảng 13 nhấn mạnh trục kép là lỗi trình bày nghiêm trọng (Bad Practice) vì dễ tạo ra tương quan giả khi người đọc nhìn vào điểm giao cắt của 2 trục có thang đo khác nhau.
* **Cách khắc phục:** Nhóm yêu cầu sửa lại thành 2 đồ thị con (Subplots) xếp chồng theo chiều dọc có chung trục hoành thời gian, nhãn trục rõ ràng và hiển thị số liệu trực tiếp trên từng đỉnh mốc.

### Ví dụ 4: Lỗi ép kiểu `dict.upper()` khi đọc file YAML lồng
* **Sai lầm của AI:** AI viết đoạn mã `city_name = cfg.get("city", "unknown").upper()`, giả định rằng `city` luôn là chuỗi string.
* **Phân tích của nhóm:** Khi người dùng định nghĩa file cấu hình `configs/sydney.yaml` chuẩn hoá có cấu trúc dictionary (`city: {name: "Sydney", country: "Australia"}`), mã nguồn lập tức phát sinh lỗi `AttributeError: 'dict' object has no attribute 'upper'`.
* **Cách khắc phục:** Nhóm bổ sung đoạn kiểm tra phòng thủ: nếu `isinstance(cfg.get("city"), dict)` thì lấy thuộc tính `.get("name")` trước khi thực hiện viết hoa.

---

## 5. Nhóm Kiểm Chứng Thế Nào (Quy Trình Kiểm Tra 4 Bước)

Để bảo đảm tính trung thực học thuật và khả năng giải trình độc lập trong phần vấn đáp:

1. **Kiểm tra mã nguồn dòng-theo-dòng (Code Review):**
   * Mọi đoạn mã AI gợi ý đều được các thành viên trong nhóm giải thích chi tiết cơ chế hoạt động, kiểm tra các trường hợp biên (`edge cases`, `NaN`, chia cho 0).
2. **Kiểm tra tính tái lập tuyệt đối (Reproducibility):**
   * Nhóm tiến hành xoá toàn bộ dữ liệu tạm trong `data/processed/`, chạy lại toàn bộ quy trình trên môi trường ảo sạch (`.venv`) bằng duy nhất câu lệnh:
     ```bash
     python run_pipeline.py --config configs/sydney.yaml
     ```
   * Xác nhận toàn bộ 6 bước (Thu thập → QA → KPI → LLM → Đánh giá đối chứng → Trực quan hoá) đều thực thi trơn tru và sinh ra đầy đủ báo cáo, bảng số liệu và biểu đồ.
3. **Thẩm định độc lập trên 100 nhãn vàng thủ công (Gold Standard Validation):**
   * Nhóm tự tay lựa chọn và dán nhãn thủ công 100 câu nhận xét đại diện cho nhiều ngôn ngữ và tình huống phức tạp.
   * Chạy song song cả Baseline và LLM để so sánh độ lệch, kiểm chứng lý do mô hình phân loại sai thông qua bảng phân tích lỗi `reports/error_analysis.csv`.
4. **Đối chiếu số liệu kinh doanh với thực tế thị trường Sydney:**
   * So sánh mức giá trung vị $310 – $326 AUD/đêm và tỷ lệ lấp đầy 50 – 58% với các báo cáo du lịch chính thức của Destination NSW và số liệu Inside Airbnb quốc tế để bảo đảm tính logic và hợp lý của kết quả phân tích.
