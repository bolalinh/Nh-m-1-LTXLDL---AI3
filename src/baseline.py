"""Baseline rule-based sentiment classifier cho Airbnb reviews.

Mục tiêu:
- Phân loại review bằng từ khóa và luật thay vì LLM.
- Minh bạch, dễ giải thích, dễ debug.
- Xử lý biểu hiện mixed, phủ định, review ngắn, review không phải tiếng Anh.

Đầu ra chuẩn:
- sentiment: positive | mixed | negative
- aspects: danh sách khía cạnh được khen/chê
- language: English | Vietnamese | Chinese | Unknown
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sydney.yaml"


# -----------------------------------------------------------------------------
# Config / dữ liệu chuẩn
# -----------------------------------------------------------------------------

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
    raise ValueError(f"Cấu hình {config_file} phải là một dictionary.")


def get_review_dataframe(config_path: str | Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Đọc dữ liệu review đã xử lý từ thư mục processed."""
    config = read_config(config_path)
    processed_root = config.get("processed_dir") or "data/processed"
    processed_root = Path(processed_root)
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root

    snapshots = [str(s) for s in config.get("snapshots", [])]
    candidates: List[Path] = []
    if snapshots:
        for snap in snapshots:
            candidates.extend([
                processed_root / str(snap) / "reviews.csv",
                processed_root / str(snap) / "reviews_clean.csv",
            ])
    candidates.extend([
        processed_root / "reviews.csv",
        processed_root / "reviews_clean.csv",
    ])

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            df = pd.read_csv(path)
            if "comments" in df.columns:
                return df
            if "comment" in df.columns:
                return df.rename(columns={"comment": "comments"})
    raise FileNotFoundError("Không tìm thấy dữ liệu review đã xử lý trong data/processed.")


# -----------------------------------------------------------------------------
# Rule dictionaries
# -----------------------------------------------------------------------------

ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "location": [
        "location", "near", "close", "walk", "walkable", "district", "neighborhood", "neighbourhood",
        "area", "view", "beach", "downtown", "center", "centre", "transport", "station", "subway", "train"
    ],
    "cleanliness": [
        "clean", "dirty", "smell", "messy", "hygiene", "tidy", "sanitized", "spotless"
    ],
    "price": [
        "price", "cheap", "expensive", "value", "worth", "cost", "rate", "budget", "overpriced"
    ],
    "host": [
        "host", "owner", "communication", "responsive", "helpful", "friendly", "kind", "check-in", "check in"
    ],
    "amenities": [
        "amenity", "wifi", "wifi", "kitchen", "pool", "gym", "parking", "aircon", "ac", "tv", "balcony", "laundry"
    ],
    "comfort": [
        "comfortable", "cozy", "cosy", "quiet", "sleep", "bed", "pillow", "soft", "spacious", "roomy"
    ],
    "noise": [
        "noise", "loud", "noisy", "busy", "traffic", "construction"
    ],
    "check_in": [
        "check in", "check-in", "arrival", "checkout", "entry", "access", "lock", "key"
    ],
    "safety": [
        "safe", "security", "secure", "unsafe", "dangerous"
    ],
    "communication": [
        "respond", "reply", "message", "contact", "support"
    ],
}

POSITIVE_WORDS = {
    "location": ["great location", "excellent location", "perfect location", "central", "walkable", "close to", "beautiful view", "amazing view"],
    "cleanliness": ["clean", "very clean", "spotless", "tidy", "fresh"],
    "price": ["cheap", "affordable", "good value", "worth it", "reasonable"],
    "host": ["friendly", "helpful", "responsive", "kind", "great host", "wonderful host"],
    "amenities": ["wifi", "comfortable", "nice kitchen", "pool", "gym", "parking", "air conditioning", "balcony"],
    "comfort": ["comfortable", "cozy", "quiet", "spacious", "good sleep", "excellent bed"],
    "safety": ["safe", "secure"],
    "communication": ["responsive", "answered quickly", "helpful"],
    "check_in": ["easy check in", "smooth check in", "easy access"],
}

NEGATIVE_WORDS = {
    "location": ["bad location", "far from", "isolated", "unsafe area", "too far", "inconvenient location"],
    "cleanliness": ["dirty", "not clean", "smelly", "messy", "filthy", "unclean"],
    "price": ["expensive", "overpriced", "too expensive", "not worth it", "pricey"],
    "host": ["rude", "unresponsive", "bad host", "ignored me", "not helpful"],
    "amenities": ["broken wifi", "no wifi", "no parking", "poor amenities", "old facilities", "bathroom dirty"],
    "comfort": ["uncomfortable", "tiny", "cramped", "noisy", "bad sleep", "hard bed"],
    "noise": ["noisy", "loud", "too noisy"],
    "check_in": ["difficult check in", "hard to access", "late check in", "locked out"],
    "safety": ["unsafe", "dangerous"],
    "communication": ["slow response", "no response", "ignored"],
}

# Dùng cho phủ định đơn giản: nếu trước từ khóa xuất hiện not/never/no/didn't/wasn't/weren't
NEGATION_WORDS = {"not", "never", "no", "isn't", "isnt", "wasn't", "wasnt", "weren't", "werent", "don't", "dont", "didn't", "didnt", "can't", "cant", "hardly"}


# -----------------------------------------------------------------------------
# Tiền xử lý / phát hiện ngôn ngữ
# -----------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Chuẩn hóa text cho rule matching."""
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Phát hiện ngôn ngữ đơn giản dựa trên từ khóa và ký tự.

    Không phải model chính xác 100%, nhưng đủ để phân biệt tiếng Anh, tiếng Việt, tiếng Trung.
    """
    clean = normalize_text(text)
    if not clean:
        return "Unknown"

    # Tiếng Việt
    vi_markers = [
        "tốt", "xấu", "rất", "không", "sạch", "bẩn", "giá", "địa điểm", "chủ nhà",
        "vệ sinh", "hài lòng", "không hài lòng", "mình", "căn hộ", "nhiều", "nhà", "không gian"
    ]
    if any(m in clean for m in vi_markers):
        return "Vietnamese"

    # Tiếng Trung
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"

    # Tiếng Anh
    english_markers = [
        "good", "bad", "clean", "dirty", "host", "location", "price", "comfortable", "noisy",
        "wifi", "room", "stay", "great", "excellent", "terrible", "nice", "recommend"
    ]
    if any(m in clean for m in english_markers):
        return "English"

    return "Unknown"


# -----------------------------------------------------------------------------
# Matching rules
# -----------------------------------------------------------------------------

def find_aspects(text: str) -> List[str]:
    """Tìm khía cạnh liên quan theo keyword."""
    normalized = normalize_text(text)
    aspects_found: set[str] = set()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                aspects_found.add(aspect)
                break
    return sorted(aspects_found)


def has_negation_before(text: str, index: int, window: int = 3) -> bool:
    """Kiểm tra có phủ định gần từ khóa hay không.

    Ví dụ: 'not clean', 'was not comfortable', 'no wifi'.
    """
    window_start = max(0, index - window)
    left_tokens = text[window_start:index].split()
    return any(tok in NEGATION_WORDS for tok in left_tokens)


def compute_sentiment_and_aspects(text: str) -> Tuple[str, List[str]]:
    """Rule-based classifier cho 1 review.

    Luật:
    1. Tách positive và negative score theo từ khóa khía cạnh.
    2. Nếu có phủ định gần từ khóa => đảo polarity.
    3. Nếu pos > neg => positive, neg > pos => negative, ngược lại => mixed.
    4. Nếu không có từ khóa => mặc định mixed.
    """
    review = normalize_text(text)
    if not review:
        return "mixed", []

    aspects_found = find_aspects(review)
    score_pos = 0
    score_neg = 0
    matched_aspects: set[str] = set()

    for aspect in ASPECT_KEYWORDS:
        for phrase in POSITIVE_WORDS.get(aspect, []):
            if phrase in review:
                if has_negation_before(review, review.find(phrase)):
                    score_neg += 1
                    matched_aspects.add(aspect)
                else:
                    score_pos += 1
                    matched_aspects.add(aspect)

        for phrase in NEGATIVE_WORDS.get(aspect, []):
            if phrase in review:
                if has_negation_before(review, review.find(phrase)):
                    score_pos += 1
                    matched_aspects.add(aspect)
                else:
                    score_neg += 1
                    matched_aspects.add(aspect)

    # Xử lý keyword không nằm trong sentence-level dict nhưng vẫn trong ASPECT_KEYWORDS
    for aspect in aspects_found:
        matched_aspects.add(aspect)

    if score_pos == 0 and score_neg == 0:
        # review ngắn hoặc không nhận diện từ khóa
        return "mixed", sorted(matched_aspects)

    if score_pos > score_neg:
        return "positive", sorted(matched_aspects)
    if score_neg > score_pos:
        return "negative", sorted(matched_aspects)
    return "mixed", sorted(matched_aspects)


# -----------------------------------------------------------------------------
# Baseline annotation
# -----------------------------------------------------------------------------

def classify_review(review_text: str) -> Dict[str, Any]:
    """Phân loại 1 review theo baseline rule-based."""
    review = str(review_text or "")
    sentiment, aspects = compute_sentiment_and_aspects(review)
    language = detect_language(review)
    return {
        "sentiment": sentiment,
        "aspects": aspects,
        "language": language,
        "success": True,
        "error": None,
    }


def save_annotations(results: List[Dict[str, Any]], output_path: str | Path) -> Path:
    """Lưu kết quả ra CSV và JSONL."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = output_file.with_suffix(".csv")
    jsonl_path = output_file.with_suffix(".jsonl")

    if not df.empty:
        df = df.reset_index(drop=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(columns=["review_id", "snapshot", "sentiment", "aspects", "language", "success", "error"]).to_csv(
            csv_path, index=False, encoding="utf-8"
        )

    return csv_path


# -----------------------------------------------------------------------------
# Pipeline chính
# -----------------------------------------------------------------------------

def run_baseline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = "data/processed",
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Chạy baseline tự động trên toàn bộ review đã xử lý."""
    df = get_review_dataframe(config_path)
    if max_rows is not None:
        df = df.head(max_rows)

    annotations: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        review_text = str(row.get("comments", row.get("comment", "")) or "")
        result = classify_review(review_text)
        payload = {
            "review_id": row.get("id", idx),
            "snapshot": row.get("snapshot", "unknown"),
            "review_text": review_text,
            "sentiment": result["sentiment"],
            "aspects": result["aspects"],
            "language": result["language"],
            "success": result["success"],
            "error": result["error"],
        }
        annotations.append(payload)

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "baseline_review_annotations"
    save_annotations(annotations, output_file)

    summary = {
        "total_reviews": len(df),
        "positive": sum(1 for x in annotations if x["sentiment"] == "positive"),
        "negative": sum(1 for x in annotations if x["sentiment"] == "negative"),
        "mixed": sum(1 for x in annotations if x["sentiment"] == "mixed"),
        "output_csv": str(output_file.with_suffix(".csv")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline rule-based sentiment classifier cho review Airbnb.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="File YAML cấu hình")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Thư mục lưu kết quả")
    parser.add_argument("--max-rows", type=int, default=None, help="Giới hạn số review xử lý cho test nhanh")
    args = parser.parse_args()

    run_baseline(
        config_path=args.config,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
    )
