"""
src/llm.py – Phân tích đánh giá bằng Large Language Model (Gemini API)
======================================================================

YÊU CẦU ĐỀ BÀI (MỤC 4: PHÂN TÍCH DÙNG LLM):
─────────────────────────────────────────────
1. Trích xuất mỗi đánh giá thành dữ liệu có cấu trúc:
   - sentiment: positive / mixed / negative
   - aspects_positive: danh sách khía cạnh khen
   - aspects_negative: danh sách khía cạnh chê
   - language: mã ngôn ngữ ISO 639-1
2. Cỡ mẫu và cách chọn: Giải thích rõ phương pháp lấy mẫu (Stratified sampling).
3. Cơ chế CACHE: Lưu vào data/processed/llm_cache.jsonl để tránh gọi lại API nhiều lần,
   tiết kiệm chi phí và đảm bảo tái lập (reproducible).
4. An toàn & bảo mật: Đọc khoá API từ biến môi trường GEMINI_API_KEY, hỗ trợ cờ --skip-llm
   để chấm test tự động mà không bị crash nếu hết quota.
5. Ước tính chi phí: Tính toán lượng token và tiền USD nếu xử lý toàn bộ review của Sydney.

KỸ THUẬT TỪ BÀI GIẢNG 11:
─────────────────────────
- google-genai Client với structured output / Pydantic schema
- Cache-first pattern: kiểm tra file cache trước khi gửi request
- Batching & Rate-limit handling với backoff retry
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from pydantic import BaseModel, Field

from src.logger import measure_execution_time

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# SCHEMA ĐẦU RA CÓ CẤU TRÚC (PYDANTIC MODEL - BÀI 11)
# ══════════════════════════════════════════════════════════════

class ReviewAnalysis(BaseModel):
    sentiment: str = Field(
        ...,
        description="Tổng thể cảm xúc của đánh giá: 'positive', 'negative', hoặc 'mixed'."
    )
    aspects_positive: List[str] = Field(
        default_factory=list,
        description="Các khía cạnh được khen ngợi trong các chủ đề: vị trí, sạch sẽ, chủ nhà, giá trị, tiện nghi, độ ồn, an toàn."
    )
    aspects_negative: List[str] = Field(
        default_factory=list,
        description="Các khía cạnh bị phàn nàn/chê trong các chủ đề: vị trí, sạch sẽ, chủ nhà, giá trị, tiện nghi, độ ồn, an toàn."
    )
    language: str = Field(
        default="en",
        description="Mã ngôn ngữ 2 ký tự ISO 639-1 (ví dụ: en, vi, zh, fr, es) hoặc 'und' nếu không xác định."
    )


# ══════════════════════════════════════════════════════════════
# HÀM BỔ TRỢ CACHE
# ══════════════════════════════════════════════════════════════

def load_cache(cache_file: Path) -> Dict[str, dict]:
    """Tải toàn bộ kết quả đã lưu trong file cache JSONL."""
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        rid = str(record.get("review_id"))
                        cache[rid] = record
                    except Exception:
                        continue
        logger.info("  ✓ Đã nạp %d bản ghi từ cache %s", len(cache), cache_file.name)
    return cache


def append_to_cache(cache_file: Path, record: dict) -> None:
    """Ghi bổ sung một bản ghi vào file JSONL."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════
# GỌI API GEMINI
# ══════════════════════════════════════════════════════════════

def _call_gemini_single(client: Any, text: str, model_name: str) -> Optional[dict]:
    """Gọi Gemini API trích xuất có cấu trúc cho 1 đoạn review."""
    prompt = f"""Bạn là chuyên gia phân tích dữ liệu cho chủ nhà Airbnb.
Hãy phân tích đánh giá sau đây và trả về kết quả JSON đúng định dạng:
1. sentiment: chỉ được là 'positive', 'negative', hoặc 'mixed'.
2. aspects_positive: danh sách khía cạnh khen ngợi (chỉ chọn từ: 'vị trí', 'sạch sẽ', 'chủ nhà', 'giá trị', 'tiện nghi', 'độ ồn', 'an toàn').
3. aspects_negative: danh sách khía cạnh phàn nàn/chê (chỉ chọn từ: 'vị trí', 'sạch sẽ', 'chủ nhà', 'giá trị', 'tiện nghi', 'độ ồn', 'an toàn').
4. language: mã ngôn ngữ 2 ký tự (en, vi, zh, es, fr, und).

Nội dung đánh giá:
\"\"\"{text}\"\"\"
"""
    try:
        # Cách 1: Sử dụng SDK google-genai mới (Interactions API / GenerateContentConfig)
        if hasattr(client, "interactions") and hasattr(client.interactions, "create"):
            response = client.interactions.create(
                model=model_name,
                input=prompt,
                response_format={"type": "json_object"}
            )
            raw_text = response.output_text
        elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            raw_text = response.text
        else:
            # Cách 2: google-generativeai cũ
            response = client.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            raw_text = response.text

        clean_json = raw_text.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        return json.loads(clean_json.strip())
    except Exception as e:
        logger.warning("  Lỗi gọi API Gemini: %s", e)
        return None


def get_gemini_client():
    """Khởi tạo Client kết nối với Gemini API từ biến môi trường."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("⚠ Không tìm thấy biến môi trường GEMINI_API_KEY!")
        return None

    # Thử import google.genai (chuẩn mới theo Bài 11)
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        pass

    # Thử import google.generativeai (phiên bản phổ biến)
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        return genai_legacy.GenerativeModel("gemini-1.5-flash")
    except ImportError:
        logger.warning("⚠ Chưa cài thư viện google-genai hoặc google-generativeai.")
        return None


# ══════════════════════════════════════════════════════════════
# LẤY MẪU & XỬ LÝ TOÀN BỘ MẪU
# ══════════════════════════════════════════════════════════════

@measure_execution_time(label="sample_reviews_stratified")
def sample_reviews_stratified(reviews_df: pd.DataFrame, listings_df: pd.DataFrame, sample_size: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Lấy mẫu phân tầng (Stratified Sampling) theo khu vực (neighbourhood).
    
    TẠI SAO LÀM THẾ NÀY?
    - Nếu lấy ngẫu nhiên đơn thuần (Simple Random Sampling), các khu vực trung tâm
      đông đúc (CBD, Bondi) sẽ chiếm hết mẫu, các khu ngoại ô ít phòng sẽ không có đại diện.
    - Lấy mẫu phân tầng theo khu vực giúp bức tranh phản hồi khách hàng công bằng
      cho tất cả các chủ nhà ở mọi khu vực tại Sydney.
    """
    logger.info("  Thực hiện lấy mẫu phân tầng %d đánh giá theo khu vực...", sample_size)
    if "neighbourhood_cleansed" in listings_df.columns and "listing_id" in reviews_df.columns:
        merged = reviews_df.merge(listings_df[["id", "neighbourhood_cleansed"]], left_on="listing_id", right_on="id", how="inner")
        if len(merged) > sample_size:
            # Lấy mẫu phân tầng
            sampled = merged.groupby("neighbourhood_cleansed", group_keys=False).apply(
                lambda g: g.sample(max(1, int(len(g) / len(merged) * sample_size)), random_state=seed)
            )
            # Nếu còn lệch một ít do làm tròn, bù thêm
            if len(sampled) < sample_size:
                extra = merged[~merged["id_x"].isin(sampled["id_x"])].sample(sample_size - len(sampled), random_state=seed)
                sampled = pd.concat([sampled, extra], ignore_index=True)
            return sampled.head(sample_size).copy()
    
    # Fallback nếu không có cột khu vực
    n_sample = min(sample_size, len(reviews_df))
    return reviews_df.sample(n=n_sample, random_state=seed).copy()


def estimate_total_sydney_cost(total_reviews_count: int, avg_words_per_review: float = 40.0) -> dict:
    """Ước tính chi phí (Tokens & USD) nếu xử lý toàn bộ đánh giá của Sydney.
    
    Yêu cầu bắt buộc của đề bài (Mục 4).
    Giá tham chiếu Gemini 2.0 Flash / 1.5 Flash:
    - Input: ~ $0.075 / 1 triệu tokens (hoặc $0.10)
    - Output: ~ $0.30 / 1 triệu tokens
    """
    # 1 từ tiếng Anh ~ 1.3 token + prompt cố định ~ 150 token input
    tokens_input_per_review = (avg_words_per_review * 1.3) + 150
    tokens_output_per_review = 60  # JSON output có cấu trúc
    
    total_input_tokens = total_reviews_count * tokens_input_per_review
    total_output_tokens = total_reviews_count * tokens_output_per_review
    
    cost_input_usd = (total_input_tokens / 1_000_000) * 0.10
    cost_output_usd = (total_output_tokens / 1_000_000) * 0.40
    total_cost_usd = cost_input_usd + cost_output_usd

    return {
        "total_reviews": total_reviews_count,
        "avg_words_per_review": round(avg_words_per_review, 1),
        "est_total_input_tokens": int(total_input_tokens),
        "est_total_output_tokens": int(total_output_tokens),
        "est_cost_input_usd": round(cost_input_usd, 2),
        "est_cost_output_usd": round(cost_output_usd, 2),
        "est_total_cost_usd": round(total_cost_usd, 2),
    }


@measure_execution_time(label="process_reviews_with_llm")
def process_reviews_with_llm(
    reviews_sample: pd.DataFrame,
    cfg: dict,
    skip_llm: bool = False
) -> pd.DataFrame:
    """Hàm chính xử lý trích xuất LLM trên tập mẫu đánh giá."""
    llm_cfg = cfg.get("llm", {})
    cache_path = Path(llm_cfg.get("cache", "data/processed/llm_cache.jsonl"))
    model_name = llm_cfg.get("model", "gemini-2.0-flash")
    
    cache = load_cache(cache_path)
    client = None if skip_llm else get_gemini_client()
    
    results = []
    new_api_calls = 0
    max_api_calls = 50  # Giới hạn an toàn để không vượt quota trong 1 lần chạy mẫu
    
    logger.info("─── BẮT ĐẦU XỬ LÝ LLM CHO %d ĐÁNH GIÁ ───", len(reviews_sample))
    if skip_llm:
        logger.info("  [CỜ --skip-llm BẬT]: Chỉ dùng cache và fallback đối chứng!")
    elif client is None:
        logger.info("  [KHÔNG CÓ API KEY]: Sử dụng kết hợp cache và mô hình heuristic đối chứng!")

    for idx, row in reviews_sample.iterrows():
        rid = str(row.get("id", idx))
        comment = str(row.get("comments", "")).strip()

        # 1. Kiểm tra cache trước (Cache-first)
        if rid in cache:
            results.append(cache[rid])
            continue

        # 2. Nếu có client và chưa quá giới hạn -> gọi API
        parsed_record = None
        if client and new_api_calls < max_api_calls and len(comment) > 10:
            parsed_record = _call_gemini_single(client, comment[:500], model_name)
            if parsed_record:
                new_api_calls += 1
                time.sleep(0.4)  # Tránh chạm rate limit

        # 3. Nếu không gọi được API hoặc skip-llm -> Fallback mô hình quy tắc
        if not parsed_record:
            from src.baseline import classify_review_baseline
            parsed_record = classify_review_baseline(comment)

        record = {
            "review_id": rid,
            "listing_id": row.get("listing_id"),
            "date": row.get("date"),
            "sentiment": parsed_record.get("sentiment", "positive"),
            "aspects_positive": parsed_record.get("aspects_positive", []),
            "aspects_negative": parsed_record.get("aspects_negative", []),
            "language": parsed_record.get("language", "en"),
            "source": "api" if client and new_api_calls <= max_api_calls else "baseline_cache"
        }
        
        append_to_cache(cache_path, record)
        cache[rid] = record
        results.append(record)

    df_results = pd.DataFrame(results)
    logger.info("✓ Hoàn tất xử lý LLM: %d bản ghi (gọi API mới: %d)", len(df_results), new_api_calls)
    return df_results
