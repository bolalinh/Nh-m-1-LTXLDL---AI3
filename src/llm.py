"""Module đánh giá review bằng LLM.

Mục tiêu:
- Đọc review từ dataset đã xử lý.
- Chọn mẫu theo phương pháp nhóm thống nhất (uniform-group sampling).
- Gửi review đến API LLM dạng structured JSON.
- Kiểm tra nhãn, xử lý lỗi API và lưu kết quả.
- Ước tính chi phí nếu xử lý toàn bộ review của thành phố.

Yêu cầu khóa API:
- Sử dụng biến môi trường OPENAI_API_KEY hoặc LLM_API_KEY.
- Có thể đặt trong file .env được bỏ qua Git.
- Không ghi trực tiếp khóa API vào mã nguồn.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sydney.yaml"


def load_env_file() -> None:
    """Tải biến môi trường từ .env nếu có."""
    if load_dotenv is not None:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)


def read_config(config_path: str | Path) -> Dict[str, Any]:
    """Đọc file YAML cấu hình."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    if not config_file.exists():
        return {}
    with open(config_file, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f) or {}
    if isinstance(config, dict):
        return config
    raise ValueError(f"Cấu hình {config_file} phải là một mapping/dictionary.")


def get_model_and_api_key(config: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
    """Lấy mô hình LLM và API key từ env/config."""
    load_env_file()
    config = config or {}
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or config.get("llm_params", {}).get("api_key")
        or ""
    )
    if not api_key:
        raise ValueError(
            "Thiếu API key. Hãy đặt OPENAI_API_KEY hoặc LLM_API_KEY trong biến môi trường hoặc file .env."
        )

    model_name = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or config.get("llm_params", {}).get("model")
        or "gpt-4o-mini"
    )
    return model_name, api_key


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_review_dataframe(config_path: str | Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Đọc dữ liệu review đã xử lý từ thư mục processed.

    Hỗ trợ các vị trí phổ biến:
    - data/processed/<snapshot>/reviews.csv
    - data/processed/reviews.csv
    - data/processed/reviews_clean.csv
    """
    config = read_config(config_path)
    processed_root = config.get("processed_dir") or "data/processed"
    processed_root = Path(processed_root)
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root

    snapshots = [str(s) for s in config.get("snapshots", [])]
    candidates: List[Path] = []

    if snapshots:
        for snap in snapshots:
            candidates.extend(
                [
                    processed_root / str(snap) / "reviews.csv",
                    processed_root / str(snap) / "reviews_clean.csv",
                ]
            )
    candidates.extend(
        [
            processed_root / "reviews.csv",
            processed_root / "reviews_clean.csv",
        ]
    )

    seen: set[Path] = set()
    for path in candidates:
        norm = path.resolve() if path.exists() else path
        if norm in seen:
            continue
        seen.add(norm)
        if path.exists():
            df = pd.read_csv(path)
            if "comments" in df.columns:
                return df
            if "comment" in df.columns:
                df = df.rename(columns={"comment": "comments"})
                return df

    raise FileNotFoundError(
        "Không tìm thấy dữ liệu review đã xử lý. Cần có file reviews.csv hoặc reviews_clean.csv trong data/processed."
    )


def select_review_sample(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    sample_size_per_group: int = 25,
    max_total: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Chọn mẫu theo phương pháp nhóm thống nhất (uniform-group sampling).

    Ý tưởng:
    - Chia dataset theo các nhóm có ý nghĩa như snapshot, area, room_type hoặc listing_id.
    - Mỗi nhóm được lấy số lượng review gần như đồng đều.
    - Không cho phép mẫu quá lớn so với giới hạn `max_total`.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    work["comments"] = work.get("comments", work.get("comment", "")).map(_coerce_text)
    work = work[work["comments"].str.len() >= 5].copy()

    if work.empty:
        return work.copy()

    if group_cols is None:
        candidate_cols = []
        for col in ["snapshot", "analysis_area", "neighbourhood_cleansed", "room_type", "listing_id"]:
            if col in work.columns:
                candidate_cols.append(col)
        group_cols = candidate_cols[:1] if candidate_cols else []

    if not group_cols:
        sample_n = min(max_total, len(work))
        return work.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    selected_frames: List[pd.DataFrame] = []
    for _, group in work.groupby(group_cols, dropna=False):
        n_take = min(len(group), max(1, sample_size_per_group))
        selected_frames.append(group.sample(n=n_take, random_state=random_state))

    sample_df = pd.concat(selected_frames, ignore_index=True)
    if len(sample_df) > max_total:
        sample_df = sample_df.sample(n=max_total, random_state=random_state).reset_index(drop=True)
    return sample_df.reset_index(drop=True)


def build_analysis_prompt(review_text: str) -> str:
    """Tạo prompt gửi tới LLM để phân tích sentiment/aspect/language."""
    return f"""
Bạn là một nhà phân tích đánh giá khách sạn/nhà ở Airbnb.
Hãy phân tích review sau và trả về JSON hợp lệ với cấu trúc tối thiểu sau:

{{
  "sentiment": "positive|mixed|negative",
  "aspects": ["tên khía cạnh 1", "tên khía cạnh 2"],
  "language": "Tên ngôn ngữ (ví dụ: English, Vietnamese, Chinese, ... )"
}}

Yêu cầu:
- sentiment: chọn 1 trong 3 nhãn positive, mixed, negative.
- aspects: liệt kê những khía cạnh được khách hàng khen/chê (ví dụ: location, cleanliness, host, price, amenities, check-in, noise, wifi, ...).
- language: xác định ngôn ngữ chính của review.
- Nếu review rỗng hoặc không rõ nghĩa, trả về language = "Unknown" và aspects = []
- Chỉ trả về JSON thuần, không giải thích thêm, không markdown code fence.
- Review cần phân tích:
\n---\n{review_text}\n---\n
""".strip()


def _extract_json_from_content(content: str) -> Dict[str, Any]:
    """Trích JSON từ chuỗi trả về từ LLM."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Kết quả LLM không phải object JSON.")
    return parsed


def validate_label(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Kiểm tra cấu trúc nhãn từ LLM; nếu thiếu, bổ sung mặc định hợp lý."""
    sentiment = str(payload.get("sentiment", "mixed")).strip().lower()
    valid_sentiments = {"positive", "mixed", "negative"}
    if sentiment not in valid_sentiments:
        sentiment = "mixed"

    aspects = payload.get("aspects", [])
    if not isinstance(aspects, list):
        aspects = []
    aspects = [str(v).strip() for v in aspects if str(v).strip()]

    language = str(payload.get("language", "Unknown")).strip() or "Unknown"
    return {
        "sentiment": sentiment,
        "aspects": aspects,
        "language": language,
    }


def call_llm_for_review(
    review_text: str,
    model_name: str,
    api_key: str,
    api_base_url: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Gửi 1 review tới OpenAI-compatible API và parse JSON trả về."""
    if not review_text or not review_text.strip():
        return {
            "sentiment": "mixed",
            "aspects": [],
            "language": "Unknown",
            "error": "empty_review",
            "success": False,
        }

    endpoint = api_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "Bạn là một trợ lý phân tích review. Chỉ trả về JSON hợp lệ theo schema yêu cầu."},
            {"role": "user", "content": build_analysis_prompt(review_text)},
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            err_msg = response.text[:500]
            return {
                "sentiment": "mixed",
                "aspects": [],
                "language": "Unknown",
                "error": f"http_{response.status_code}: {err_msg}",
                "success": False,
            }

        data = response.json()
        if "choices" not in data or not data["choices"]:
            return {
                "sentiment": "mixed",
                "aspects": [],
                "language": "Unknown",
                "error": "empty_choices",
                "success": False,
            }

        content = data["choices"][0]["message"].get("content", "")
        parsed = _extract_json_from_content(content)
        validated = validate_label(parsed)
        validated["success"] = True
        validated["error"] = None
        return validated
    except requests.exceptions.Timeout:
        return {
            "sentiment": "mixed",
            "aspects": [],
            "language": "Unknown",
            "error": "timeout",
            "success": False,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "sentiment": "mixed",
            "aspects": [],
            "language": "Unknown",
            "error": f"request_error: {exc}",
            "success": False,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "sentiment": "mixed",
            "aspects": [],
            "language": "Unknown",
            "error": f"invalid_json: {exc}",
            "success": False,
        }


def estimate_total_cost(
    total_reviews: int,
    model_name: str,
    avg_input_tokens_per_review: int = 350,
    avg_output_tokens_per_review: int = 80,
) -> Dict[str, float]:
    """Ước tính chi phí nếu xử lý toàn bộ review của city.

    Mặc định dùng giá tham khảo của gpt-4o-mini:
    - input: $0.15 / 1M tokens
    - output: $0.60 / 1M tokens
    """
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (5.00, 15.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4-turbo": (10.00, 30.00),
    }
    input_price, output_price = pricing.get(model_name, (0.15, 0.60))

    total_input_tokens = total_reviews * avg_input_tokens_per_review
    total_output_tokens = total_reviews * avg_output_tokens_per_review

    est_cost = (
        total_input_tokens / 1_000_000 * input_price
        + total_output_tokens / 1_000_000 * output_price
    )

    return {
        "total_reviews": total_reviews,
        "avg_input_tokens_per_review": avg_input_tokens_per_review,
        "avg_output_tokens_per_review": avg_output_tokens_per_review,
        "estimated_cost_usd": round(est_cost, 6),
    }


def save_annotations(results: List[Dict[str, Any]], output_path: str | Path) -> Path:
    """Lưu danh sách kết quả phân tích dưới dạng CSV và JSONL."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = output_file.with_suffix(".csv")
    jsonl_path = output_file.with_suffix(".jsonl")

    if not df.empty:
        df = df.reset_index(drop=True)
        if "review_text" in df.columns:
            df["review_text"] = df["review_text"].astype(str)
        if "aspects" in df.columns:
            df["aspects"] = df["aspects"].map(lambda x: x if isinstance(x, list) else [])
        df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(columns=["review_id", "snapshot", "sentiment", "aspects", "language", "success", "error"]).to_csv(
            csv_path, index=False, encoding="utf-8"
        )

    return csv_path


def run_llm_annotation(
    config_path: str | Path = DEFAULT_CONFIG,
    sample_size_per_group: int = 25,
    max_total: int = 300,
    output_dir: Optional[str | Path] = None,
    group_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Chạy toàn bộ quy trình phân loại review bằng LLM."""
    config = read_config(config_path)
    model_name, api_key = get_model_and_api_key(config)

    df = get_review_dataframe(config_path)
    sample_df = select_review_sample(
        df,
        group_cols=group_cols,
        sample_size_per_group=sample_size_per_group,
        max_total=max_total,
        random_state=42,
    )

    payloads: List[Dict[str, Any]] = []
    for idx, row in sample_df.iterrows():
        review_text = _coerce_text(row.get("comments", row.get("comment", "")))
        review_id = row.get("id")
        snapshot = row.get("snapshot", "unknown")
        result = call_llm_for_review(review_text, model_name=model_name, api_key=api_key)
        payload = {
            "review_id": review_id,
            "snapshot": snapshot,
            "review_text": review_text,
            "sentiment": result.get("sentiment", "mixed"),
            "aspects": result.get("aspects", []),
            "language": result.get("language", "Unknown"),
            "success": result.get("success", False),
            "error": result.get("error"),
        }
        payloads.append(payload)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "llm_review_annotations"
    save_annotations(payloads, output_file)

    total_reviews = len(df)
    cost_estimate = estimate_total_cost(total_reviews, model_name)
    summary = {
        "model": model_name,
        "sample_size": len(sample_df),
        "total_reviews_in_city": total_reviews,
        "output_file": str(output_file.with_suffix(".csv")),
        "successful_labels": sum(1 for x in payloads if x.get("success") is True),
        "failed_labels": sum(1 for x in payloads if x.get("success") is not True),
        "cost_estimate_usd": cost_estimate["estimated_cost_usd"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân tích review Airbnb bằng LLM.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Đường dẫn file YAML cấu hình")
    parser.add_argument("--sample-size-per-group", type=int, default=25, help="Số review tối đa mỗi nhóm")
    parser.add_argument("--max-total", type=int, default=300, help="Số review mẫu tối đa gửi cho LLM")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Thư mục lưu file kết quả")
    args = parser.parse_args()

    run_llm_annotation(
        config_path=args.config,
        sample_size_per_group=args.sample_size_per_group,
        max_total=args.max_total,
        output_dir=args.output_dir,
    )
