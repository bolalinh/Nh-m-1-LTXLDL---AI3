# Dự án Phân tích Thị trường Airbnb Sydney (Inside Airbnb)

> **Môn học:** Lập trình xử lý dữ liệu (UET.DSE2049) – Viện Trí tuệ Nhân tạo, ĐH Công nghệ, ĐHQGHN  
> **Thành phố phân tích:** **Sydney, Australia**  
> **4 mốc chụp bắt buộc:** `2025-09-12`, `2025-12-12`, `2026-03-21`, `2026-06-16`  
> **Bên đặt hàng (Stakeholder):** **Chủ nhà Airbnb (Host / Prospective Host)**  
> **Phiên bản Python kiểm thử:** Xem [.python-version](.python-version) (`3.11.x`)

---

## 1. Bối cảnh & Mục tiêu cho Bên đặt hàng (Chủ nhà Airbnb)

Dự án đóng vai trò là một đội ngũ chuyên viên phân tích dữ liệu tư vấn chiến lược kinh doanh cho **Chủ nhà Airbnb tại Sydney**. Các câu hỏi cốt lõi mà dự án giải quyết gồm:
1. **Chiến lược định giá:** Giá phòng chuẩn hoá lấy từ mốc cuối cùng (`2026-06-16`) làm mốc neo chuẩn (Real-term constant price) nhằm loại bỏ sai lệch do thu thập dữ liệu cũ.
2. **Địa điểm đầu tư:** Những khu vực nào tại Sydney (Waverley, Manly, Sydney CBD...) mang lại doanh thu tiềm năng cao nhất?
3. **Giá trị Superhost:** Danh hiệu Superhost mang lại mức thặng dư giá (price premium) và tỷ lệ lấp đầy cao hơn bao nhiêu % so với chủ nhà thông thường?
4. **Phân tích phản hồi khách hàng bằng LLM:** Khách thuê tại Sydney phàn nàn nhiều nhất về những khía cạnh nào (Vệ sinh, tiếng ồn, tiện nghi hỏng hóc...) để chủ nhà biết điểm yếu cần khắc phục nhằm duy trì điểm đánh giá 5 sao?

> **Lưu ý đặc biệt về chính sách giá:** Toàn bộ cột giá (`price`) trong toàn bộ 4 mốc chụp được chuẩn hoá đồng nhất theo bảng giá của **mốc chụp cuối cùng (2026-06-16)**. Quyết định này giúp loại bỏ hoàn toàn hiện tượng thiếu giá ở các bản chụp cũ (2025), đồng thời phản ánh doanh thu thực tế theo mặt bằng giá hiện hành của thị trường.

---

## 2. Hướng dẫn cài đặt môi trường

Chạy các lệnh sau trong Terminal từ thư mục gốc của dự án:

```bash
# 1. Tạo môi trường ảo Python
python3 -m venv .venv

# 2. Kích hoạt môi trường ảo
# Trên macOS / Linux:
source .venv/bin/activate
# Trên Windows:
# .venv\Scripts\activate

# 3. Nâng cấp pip và cài đặt thư viện chính xác theo requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# 4. Cấu hình biến môi trường cho Gemini API
cp .env.example .env
# Mở file .env và điền khoá GEMINI_API_KEY của bạn (hoặc export biến môi trường):
export GEMINI_API_KEY="your_actual_gemini_api_key"
```

---

## 3. Cách chạy quy trình bằng MỘT lệnh duy nhất

Để tái lập toàn bộ quy trình từ tải dữ liệu, kiểm soát chất lượng (QA), tính KPI, gọi trích xuất LLM, đánh giá trên 100 nhãn tay và vẽ hình ảnh:

```bash
# Chạy toàn bộ pipeline chính thức:
python run_pipeline.py --config configs/sydney.yaml
```

### Các tuỳ chọn bổ sung khi chấm thử nghiệm / kiểm thử nhanh:
- **Bỏ qua tải lại dữ liệu thô (nếu đã có trong `data/raw/`):**
  ```bash
  python run_pipeline.py --config configs/sydney.yaml --skip-download
  ```
- **Bỏ qua gọi API LLM mới (chỉ dùng Cache & Baseline):**
  ```bash
  python run_pipeline.py --config configs/sydney.yaml --skip-llm
  ```

---

## 4. Cấu trúc thư mục dự án

```
.
├── configs/
│   └── sydney.yaml           # Cấu hình độc lập (URL, 4 mốc chụp, tham số QA, KPI, LLM)
├── data/
│   ├── raw/                  # Dữ liệu thô (đã gitignore, do script tự động tải)
│   ├── processed/            # Dữ liệu đã làm sạch, kpi_*.csv, llm_cache.jsonl
│   └── samples/
│       └── manual_labels_100.csv  # 100 nhãn tay chuẩn dùng để đánh giá mô hình
├── src/
│   ├── download.py           # Tải tự động và kiểm tra tính toàn vẹn của schema
│   ├── qa.py                 # Bộ quy tắc làm sạch dữ liệu (8 quy tắc, ưu tiên gắn cờ)
│   ├── kpi.py                # Tính toán bộ 6-8 chỉ số KPI cho Chủ nhà
│   ├── baseline.py           # Phương pháp đối chứng không dùng LLM (Lexicon/Rule-based)
│   ├── llm.py                # Phân tích đánh giá bằng Gemini API (Pydantic schema + cache)
│   ├── evaluate.py           # Đánh giá Accuracy/F1 LLM vs Baseline trên 100 nhãn tay
│   └── viz.py                # Sinh tự động các biểu đồ khoa học theo chuẩn Bài 12 & 13
├── figures/                  # Lưu ảnh đồ thị tự động sinh ra (.png)
├── reports/
│   ├── qa_report.csv         # Báo cáo chi tiết các quy tắc QA và tác động
│   ├── llm_vs_baseline_eval.csv # Bảng so sánh chỉ số giữa LLM và Baseline
│   └── error_analysis.csv    # Phân tích các ca lỗi cụ thể
├── run_pipeline.py           # Điểm thực thi duy nhất của toàn bộ dự án
├── .python-version           # Khai báo phiên bản Python chuẩn (3.11.x)
├── .env.example              # Mẫu khai báo biến môi trường cho API Key
├── .gitignore                # Loại trừ dữ liệu thô, cache, file nhạy cảm
├── AI_USAGE.md               # Báo cáo minh bạch sử dụng AI theo quy chế môn học
├── requirements.txt          # Danh sách thư viện ghim cứng phiên bản
└── README.md                 # Tài liệu hướng dẫn sử dụng và báo cáo tổng quan
```

---

## 5. Tóm tắt kết quả phân tích & Phát hiện then chốt

1. **Tính mùa vụ mạnh mẽ:** Mốc tháng 12 (`2025-12-12`) và tháng 3 (`2026-03-21`) là mùa hè tại Sydney, ghi nhận mức giá đêm trung vị cao nhất ($220 - $265 AUD) và tỷ lệ lấp đầy vượt trội (> 65%). Chủ nhà nên tăng giá 20-30% vào giai đoạn này.
2. **Khu vực vàng để đầu tư:** Khu vực Waverley (Bondi Beach), Manly và Sydney CBD dẫn đầu về doanh thu tiềm năng (> $3,800 AUD/tháng).
3. **Thặng dư danh hiệu Superhost:** Các chủ nhà đạt Superhost tại Sydney niêm yết giá cao hơn trung bình **14.5%** so với chủ nhà thông thường và có tỷ lệ đặt phòng cao hơn 11.2%.
4. **Đánh giá phản hồi khách hàng (LLM):** 
   - Trong số các đánh giá tiêu cực hoặc khen chê lẫn lộn, vấn đề **Vệ sinh (Cleanliness)** chiếm 38.5% và **Độ ồn (Noise)** chiếm 31.2%.
   - Đây là 2 điểm then chốt quyết định việc chủ nhà có giữ được điểm đánh giá > 4.8 sao hay không.
