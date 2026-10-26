# AI Usage

## 1. Mục đích

LLM được sử dụng trong dự án để phân tích các review Airbnb theo các tiêu chí sau:

- xác định sentiment của review: `positive`, `mixed`, `negative`
- trích xuất các khía cạnh được khách hàng nhắc đến: ví dụ `cleanliness`, `location`, `price`, `host`, `noise`, `wifi`, `amenities`, `check-in`
- xác định ngôn ngữ chính của review: `English`, `Vietnamese`, `Chinese`, `Unknown`, ...
- hỗ trợ so sánh hiệu suất giữa phương pháp baseline và phương pháp LLM trên cùng một tập dữ liệu đánh giá thủ công

Mục tiêu của việc dùng AI không phải thay thế hoàn toàn quy tắc thủ công, mà là tạo ra một phương án phân loại có thể so sánh trực tiếp với baseline nhằm đánh giá độ chính xác, độ tin cậy và tính có thể mở rộng của mô hình.

---

## 2. Model

Dự án sử dụng mô hình LLM theo chuẩn OpenAI-compatible API, tức là có thể gọi qua endpoint tương thích với OpenAI API.

Cấu hình mặc định hiện tại:

- model: `gpt-4o-mini`
- biến môi trường bắt buộc:
  - `OPENAI_API_KEY` hoặc `LLM_API_KEY`
- biến môi trường tùy chọn:
  - `OPENAI_MODEL` hoặc `LLM_MODEL`
  - `OPENAI_BASE_URL` nếu endpoint là self-hosted hoặc provider khác
- file `.env` được hỗ trợ để lưu khóa API mà không hardcode trực tiếp vào mã nguồn

Ví dụ cấu hình .env:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

Nếu không có API key, pipeline sẽ bỏ qua bước LLM thay vì crash toàn bộ dự án.

---

## 3. Prompt

Prompt được thiết kế theo kiểu structured JSON để LLM trả về các nhãn rõ ràng, dễ parse và dễ đánh giá.

Cấu trúc JSON yêu cầu tối thiểu:

```json
{
  "sentiment": "positive|mixed|negative",
  "aspects": ["location", "cleanliness"],
  "language": "English"
}
```

Prompt mẫu được tạo trong `llm.py` như sau:

- hệ thống (system prompt): "Bạn là một trợ lý phân tích review. Chỉ trả về JSON hợp lệ theo schema yêu cầu."
- người dùng (user prompt): chứa nội dung review cần phân tích và yêu cầu định dạng đầu ra

Yêu cầu đối với prompt:

- chỉ trả về JSON thuần, không có markdown code fence
- không giải thích dài dòng
- nếu review rỗng hoặc không rõ nghĩa thì trả về:
  - `sentiment = "mixed"`
  - `aspects = []`
  - `language = "Unknown"`
- ưu tiên xử lý dạng structured output để dễ lưu trữ và so sánh với human labels

---

## 4. Human Evaluation

Nhóm sẽ xây dựng một human-labeled dataset từ một tập con review được lấy từ dữ liệu Airbnb đã qua QA.

Tiêu chí đánh giá người dùng:

- ít nhất 100 review được gán nhãn thủ công
- dữ liệu nên bao gồm nhiều mức sentiment khác nhau: tích cực, trung tính, tiêu cực
- label cần được kiểm tra lại để đảm bảo tính nhất quán
- mỗi review nên có ít nhất các nhãn sau:
  - `sentiment`
  - `aspects` (nếu có)
  - `language`

Mục tiêu của human evaluation:

- làm ground truth để đánh giá mô hình AI
- so sánh kết quả giữa LLM và baseline
- xác định lỗi phân loại và các trường hợp mô hình dễ nhầm lẫn

---

## 5. Evaluation

Kết quả LLM sẽ được đối chiếu với human labels và với baseline bằng các metric chuẩn sau:

- Accuracy
- Precision
- Recall
- Macro-F1
- Confusion matrix
- Error analysis

Điểm khác biệt quan trọng so với chỉ báo cáo Accuracy là:

- Accuracy đơn lẻ không đủ khi dữ liệu có class imbalance
- Precision/Recall/F1 cho thấy hiệu suất theo từng nhãn
- Confusion matrix giúp xác định nhãn nào bị lẫn nhau
- Error analysis giúp nhìn rõ các review bị sai, từ đó cải thiện prompt hoặc quy tắc baseline

Đối tượng đánh giá:

- `baseline` → phương pháp phân loại dựa trên từ khóa và luật
- `llm` → phương pháp dựa trên mô hình AI
- `human` → label gốc dùng làm ground truth

Các báo cáo đánh giá được lưu trong thư mục `reports/` như JSON và markdown.

---

## 6. Cost

Việc dùng LLM có chi phí phụ thuộc vào số lượng review và số token được gửi đi.

Trong dự án, chi phí được ước tính theo công thức:

- chi phí = chi phí input + chi phí output
- với gpt-4o-mini, ước tính tham khảo:
  - input: $0.15 / 1M tokens
  - output: $0.60 / 1M tokens

Công thức ước tính trong `llm.py`:

```python
estimated_cost = (
    total_input_tokens / 1_000_000 * input_price
    + total_output_tokens / 1_000_000 * output_price
)
```

Để giảm chi phí, dự án có áp dụng:

- sampling theo nhóm (uniform-group sampling)
- giới hạn `max_total`
- chỉ phân tích một tập mẫu đại diện thay vì toàn bộ review của thành phố
- ưu tiên model nhỏ hơn như `gpt-4o-mini` khi cần tối ưu chi phí

---

## 7. Quy trình sử dụng thực tế

Pipeline tổng quát của dự án:

1. `download.py` tải dữ liệu raw từ Airbnb
2. `qa.py` kiểm tra chất lượng dữ liệu và sinh dữ liệu đã xử lý
3. `kpi.py` tính các KPI theo thời gian/khu vực/loại phòng
4. `baseline.py` phân loại review bằng luật và từ khóa
5. `llm.py` phân loại review bằng LLM
6. `evaluate.py` so sánh baseline và LLM với human labels
7. `viz.py` trực quan hóa KPI và kết quả đánh giá

---

## 8. Lưu ý thực thi

- Không ghi trực tiếp API key vào source code
- Sử dụng `python-dotenv` để load `.env`
- Cần đảm bảo file config YAML hợp lệ trước khi chạy pipeline
- Nếu thiếu API key, phần LLM bị skip an toàn thay vì làm hỏng toàn bộ quy trình

---

## 9. Kết luận

AI trong dự án được sử dụng như một công cụ hỗ trợ phân tích review Airbnb theo hướng có thể kiểm chứng, có thể so sánh và có thể định lượng. Việc kết hợp giữa baseline, LLM và human evaluation giúp đảm bảo không chỉ có thể xây dựng model phân loại mà còn có thể đánh giá một cách khách quan về hiệu quả thực tế của nó.