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
4. An toàn & bảo mật: Đọc khoá API từ biến môi trường GEMINI_API_KEY (tự nạp từ .env),
   hỗ trợ cờ --skip-llm để chấm test tự động mà không bị crash nếu hết quota.
5. Ước tính chi phí: Tính toán lượng token và tiền USD nếu xử lý toàn bộ review của Sydney.

KỸ THUẬT TỪ BÀI GIẢNG 11:
─────────────────────────
- google-genai Client với structured output / Pydantic schema
- Cache-first pattern: kiểm tra file cache trước khi gửi request
- Rate-limit handling với exponential backoff retry
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from src.logger import measure_execution_time

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ASPECTS = [
    "vị trí",
    "sạch sẽ",
    "chủ nhà",
    "giá trị",
    "tiện nghi",
    "độ ồn",
    "an toàn",
    "giao thông",
]
VALID_SENTIMENTS = {"positive", "negative", "mixed"}
LANGUAGE_ALIASES = {
    "english": "en",
    "vietnamese": "vi",
    "chinese": "zh",
    "mandarin": "zh",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "japanese": "ja",
    "korean": "ko",
    "unknown": "und",
}


def _load_env(env_path: Optional[Path] = None) -> None:
    """Nạp biến môi trường từ file .env của project nếu tồn tại."""
    target = env_path or (PROJECT_ROOT / ".env")
    if not target.exists():
        return
    if load_dotenv is not None:
        load_dotenv(dotenv_path=target, override=False)
        return
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# ══════════════════════════════════════════════════════════════
# SCHEMA ĐẦU RA CÓ CẤU TRÚC (PYDANTIC MODEL - BÀI 11)
# ══════════════════════════════════════════════════════════════

class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "mixed"] = Field(
        ...,
        description="Tổng thể cảm xúc của đánh giá: 'positive', 'negative', hoặc 'mixed'."
    )
    aspects_positive: List[str] = Field(
        default_factory=list,
        description="Các khía cạnh được khen ngợi."
    )
    aspects_negative: List[str] = Field(
        default_factory=list,
        description="Các khía cạnh bị phàn nàn/chê."
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
    cache: Dict[str, dict] = {}
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

def _build_prompt(text: str, aspects: List[str]) -> str:
    """Tạo prompt chuẩn hóa theo danh sách khía cạnh từ cấu hình."""
    aspects_str = ", ".join(f"'{a}'" for a in aspects)
    return f"""Bạn là chuyên gia phân tích dữ liệu cho chủ nhà Airbnb.
Hãy phân tích đánh giá sau đây và trả về JSON đúng cấu trúc:
1. sentiment: chỉ chọn 1 trong 3 giá trị: 'positive', 'negative', hoặc 'mixed'.
2. aspects_positive: danh sách khía cạnh được khen ngợi (chỉ chọn trong: {aspects_str}).
3. aspects_negative: danh sách khía cạnh bị phàn nàn/chê (chỉ chọn trong: {aspects_str}).
4. language: mã ngôn ngữ ISO 639-1 gồm 2 ký tự viết thường (ví dụ: en, vi, zh, es, fr) hoặc 'und' nếu không xác định.

Nội dung đánh giá:
\"\"\"{text}\"\"\"
"""


def _normalize_analysis(payload: dict, allowed_aspects: Optional[List[str]] = None) -> dict:
    """Chuẩn hóa và kiểm tra hợp lệ đầu ra từ LLM."""
    sentiment = str(payload.get("sentiment", "mixed")).strip().lower()
    if sentiment == "neutral":
        sentiment = "mixed"
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "mixed"

    allowed_set = {a.strip().lower() for a in allowed_aspects} if allowed_aspects else None

    def _clean_aspects(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        cleaned: List[str] = []
        for item in raw:
            val = str(item).strip().lower()
            if not val:
                continue
            if allowed_set is not None and val not in allowed_set:
                continue
            if val not in cleaned:
                cleaned.append(val)
        return cleaned

    raw_lang = str(payload.get("language", "und")).strip().lower()
    lang = LANGUAGE_ALIASES.get(raw_lang, raw_lang)
    if lang != "und" and (len(lang) != 2 or not lang.isalpha()):
        lang = "und"

    return {
        "sentiment": sentiment,
        "aspects_positive": _clean_aspects(payload.get("aspects_positive")),
        "aspects_negative": _clean_aspects(payload.get("aspects_negative")),
        "language": lang,
    }


def _parse_response_payload(response: Any) -> dict:
    """Trích xuất dict JSON từ response của SDK google-genai hoặc legacy."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return parsed

    raw_text = getattr(response, "output_text", None) or getattr(response, "text", None) or ""
    clean_json = raw_text.strip()
    if clean_json.startswith("```json"):
        clean_json = clean_json[7:]
    elif clean_json.startswith("```"):
        clean_json = clean_json[3:]
    if clean_json.endswith("```"):
        clean_json = clean_json[:-3]

    data = json.loads(clean_json.strip())
    if not isinstance(data, dict):
        raise ValueError("LLM trả về JSON không phải object.")
    return data


def _is_non_retryable_error(exc: Exception) -> bool:
    """Phân biệt lỗi vĩnh viễn (auth/permission/model không tồn tại) để dừng retry sớm."""
    msg = str(exc)
    non_retryable_tokens = (
        "400 INVALID_ARGUMENT",
        "401 UNAUTHENTICATED",
        "403 PERMISSION_DENIED",
        "404 NOT_FOUND",
        "API_KEY_INVALID",
    )
    return any(token in msg for token in non_retryable_tokens)


def _call_gemini_single(
    client: Any,
    text: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    aspects: Optional[List[str]] = None,
    max_retries: int = 3,
) -> Optional[dict]:
    """Gọi Gemini API trích xuất có cấu trúc cho 1 đoạn review."""
    aspect_list = aspects or DEFAULT_ASPECTS
    prompt = _build_prompt(text, aspect_list)
    attempts = max(1, int(max_retries))

    for attempt in range(attempts):
        try:
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                try:
                    from google.genai import types
                    config: Any = types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                        response_schema=ReviewAnalysis,
                    )
                except Exception:
                    config = {
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                    }
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            elif hasattr(client, "interactions") and hasattr(client.interactions, "create"):
                response = client.interactions.create(
                    model=model_name,
                    input=prompt,
                    response_format={"type": "json_object"},
                )
            else:
                response = client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                    },
                )

            payload = _parse_response_payload(response)
            return _normalize_analysis(payload, allowed_aspects=aspect_list)
        except Exception as e:
            if _is_non_retryable_error(e) or attempt >= attempts - 1:
                logger.warning("  Lỗi gọi API Gemini (%s): %s", model_name, e)
                return None
            wait_sec = min(8.0, 1.0 * (2 ** attempt))
            logger.warning(
                "  Lỗi tạm thời khi gọi Gemini (lần %d/%d), thử lại sau %.1fs: %s",
                attempt + 1, attempts, wait_sec, e,
            )
            time.sleep(wait_sec)
    return None


def get_gemini_client(model_name: str = DEFAULT_MODEL):
    """Khởi tạo Client kết nối với Gemini API từ biến môi trường / file .env."""
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("⚠ Không tìm thấy biến môi trường GEMINI_API_KEY hợp lệ!")
        return None

    # Ưu tiên SDK google-genai mới (chuẩn Bài 11)
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        pass

    # Fallback google-generativeai (legacy)
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        return genai_legacy.GenerativeModel(model_name)
    except ImportError:
        logger.warning("⚠ Chưa cài thư viện google-genai hoặc google-generativeai.")
        return None


# ══════════════════════════════════════════════════════════════
# LẤY MẪU & XỬ LÝ TOÀN BỘ MẪU
# ══════════════════════════════════════════════════════════════

@measure_execution_time(label="sample_reviews_stratified")
def sample_reviews_stratified(
    reviews_df: pd.DataFrame,
    listings_df: pd.DataFrame,
    sample_size: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """Lấy mẫu phân tầng (Stratified Sampling) theo khu vực (neighbourhood).

    TẠI SAO LÀM THẾ NÀY?
    - Nếu lấy ngẫu nhiên đơn thuần (Simple Random Sampling), các khu vực trung tâm
      đông đúc (CBD, Bondi) sẽ chiếm hết mẫu, các khu ngoại ô ít phòng sẽ không có đại diện.
    - Lấy mẫu phân tầng theo khu vực giúp bức tranh phản hồi khách hàng công bằng
      cho tất cả các chủ nhà ở mọi khu vực tại Sydney.
    """
    logger.info("  Thực hiện lấy mẫu phân tầng %d đánh giá theo khu vực...", sample_size)
    if "neighbourhood_cleansed" in listings_df.columns and "listing_id" in reviews_df.columns:
        listings_area = (
            listings_df[["id", "neighbourhood_cleansed"]]
            .drop_duplicates(subset=["id"])
            .rename(columns={"id": "listing_id"})
        )
        merged = reviews_df.merge(listings_area, on="listing_id", how="inner")
        if len(merged) <= sample_size:
            return merged.copy()

        total_rows = len(merged)
        parts = []
        for _, group in merged.groupby("neighbourhood_cleansed", sort=False):
            n_take = min(len(group), max(1, int(len(group) / total_rows * sample_size)))
            parts.append(group.sample(n=n_take, random_state=seed))

        sampled = pd.concat(parts, ignore_index=True)
        if len(sampled) < sample_size:
            if "id" in merged.columns:
                remaining = merged[~merged["id"].isin(sampled["id"])]
            else:
                remaining = merged.drop(index=sampled.index, errors="ignore")
            n_extra = min(len(remaining), sample_size - len(sampled))
            if n_extra > 0:
                extra = remaining.sample(n=n_extra, random_state=seed)
                sampled = pd.concat([sampled, extra], ignore_index=True)
        elif len(sampled) > sample_size:
            sampled = sampled.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        return sampled.copy()

    # Fallback nếu không có cột khu vực
    n_sample = min(sample_size, len(reviews_df))
    return reviews_df.sample(n=n_sample, random_state=seed).copy()


def estimate_total_sydney_cost(total_reviews_count: int, avg_words_per_review: float = 40.0) -> dict:
    """Ước tính chi phí (Tokens & USD) nếu xử lý toàn bộ đánh giá của Sydney.

    Yêu cầu bắt buộc của đề bài (Mục 4).
    Giá tham chiếu Gemini 3.5 Flash Lite:
    - Input: ~ $0.15 / 1 triệu tokens
    - Output: ~ $0.60 / 1 triệu tokens
    """
    # 1 từ tiếng Anh ~ 1.3 token + prompt cố định ~ 150 token input
    tokens_input_per_review = (avg_words_per_review * 1.3) + 150
    tokens_output_per_review = 60  # JSON output có cấu trúc

    total_input_tokens = total_reviews_count * tokens_input_per_review
    total_output_tokens = total_reviews_count * tokens_output_per_review

    cost_input_usd = (total_input_tokens / 1_000_000) * 0.15
    cost_output_usd = (total_output_tokens / 1_000_000) * 0.60
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
    skip_llm: bool = False,
) -> pd.DataFrame:
    """Hàm chính xử lý trích xuất LLM trên tập mẫu đánh giá."""
    llm_cfg = cfg.get("llm", {})
    raw_cache_path = Path(llm_cfg.get("cache", "data/processed/llm_cache.jsonl"))
    cache_path = raw_cache_path if raw_cache_path.is_absolute() else (PROJECT_ROOT / raw_cache_path)

    model_name = llm_cfg.get("model", DEFAULT_MODEL)
    temperature = float(llm_cfg.get("temperature", 0.0))
    max_retries = int(llm_cfg.get("max_retries", 3))
    aspects = llm_cfg.get("aspects") or DEFAULT_ASPECTS
    max_api_calls = int(llm_cfg.get("max_api_calls", 15))  # Giới hạn an toàn tránh vượt quota/chi phí

    cache = load_cache(cache_path)
    client = None if skip_llm else get_gemini_client(model_name=model_name)

    results = []
    new_api_calls = 0
    cache_hits = 0

    logger.info("─── BẮT ĐẦU XỬ LÝ LLM (%s) CHO %d ĐÁNH GIÁ ───", model_name, len(reviews_sample))
    if skip_llm:
        logger.info("  [CỜ --skip-llm BẬT]: Chỉ dùng cache API và fallback đối chứng!")
    elif client is None:
        logger.info("  [KHÔNG CÓ API KEY]: Sử dụng kết hợp cache và mô hình heuristic đối chứng!")

    for idx, row in reviews_sample.iterrows():
        rid = str(row.get("id", row.get("review_id", idx)))
        raw_comment = row.get("comments", "")
        comment = "" if pd.isna(raw_comment) else str(raw_comment).strip()

        # 1. Kiểm tra cache trước (chỉ dùng bản ghi fallback khi không gọi API)
        cached_record = cache.get(rid)
        if cached_record and (cached_record.get("source") == "api" or skip_llm or client is None):
            results.append(cached_record)
            cache_hits += 1
            continue

        # 2. Nếu có client và chưa quá giới hạn -> gọi API
        parsed_record = None
        used_api = False
        if client and new_api_calls < max_api_calls and len(comment) > 10:
            parsed_record = _call_gemini_single(
                client,
                comment[:500],
                model_name=model_name,
                temperature=temperature,
                aspects=aspects,
                max_retries=max_retries,
            )
            if parsed_record is not None:
                used_api = True
                new_api_calls += 1
                time.sleep(0.4)  # Tránh chạm rate limit

        # 3. Nếu không gọi được API hoặc skip-llm -> Fallback mô hình quy tắc
        if parsed_record is None:
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
            "source": "api" if used_api else "baseline_fallback",
        }

        # Chỉ lưu cache cho kết quả thật từ API để tránh làm bẩn cache bằng baseline
        if used_api:
            append_to_cache(cache_path, record)
            cache[rid] = record
        results.append(record)

    df_results = pd.DataFrame(results)
    logger.info(
        "✓ Hoàn tất xử lý LLM: %d bản ghi (cache: %d, gọi API mới: %d, baseline fallback: %d)",
        len(df_results),
        cache_hits,
        new_api_calls,
        len(df_results) - cache_hits - new_api_calls,
    )
    return df_results


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Chạy module LLM phân tích review Airbnb.")
    parser.add_argument("--config", type=str, default="configs/sydney.yaml")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--max-api-calls", type=int, default=10)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("llm", {})["max_api_calls"] = args.max_api_calls
    snapshots = [str(s) for s in cfg.get("snapshots", [])]
    latest_snap = max(snapshots) if snapshots else "2026-06-16"

    reviews_path = PROJECT_ROOT / "data" / "raw" / latest_snap / "reviews.csv"
    listings_path = PROJECT_ROOT / "data" / "raw" / latest_snap / "listings.csv"
    reviews_df = pd.read_csv(reviews_path, usecols=["listing_id", "id", "date", "comments"])
    listings_df = pd.read_csv(listings_path, usecols=["id", "neighbourhood_cleansed"])

    sample_df = sample_reviews_stratified(
        reviews_df,
        listings_df,
        sample_size=args.sample_size,
        seed=int(cfg.get("seed", 42)),
    )
    result_df = process_reviews_with_llm(sample_df, cfg, skip_llm=args.skip_llm)
    print(result_df["source"].value_counts())
    print(result_df.head(5).to_string(index=False))

    if args.save_csv:
        out_path = PROJECT_ROOT / "data" / "processed" / "llm_review_annotations.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
        logger.info("✓ Đã lưu: %s", out_path)
