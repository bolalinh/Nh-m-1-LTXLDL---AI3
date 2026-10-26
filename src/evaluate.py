"""Đánh giá baseline và LLM trên tập review có nhãn thủ công.

Mỗi review phải có nhãn sentiment thủ công theo một trong:
- positive
- mixed
- negative

Các metric chính:
- Accuracy
- Precision (macro)
- Recall (macro)
- Macro-F1
- Confusion matrix
- Phân tích trường hợp sai

Mục tiêu: so sánh baseline rule-based với LLM trên cùng tập dữ liệu, tránh
chỉ tin vào accuracy khi dữ liệu mất cân bằng giữa các lớp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sydney.yaml"
LABELS = ["positive", "negative", "mixed"]


def read_config(config_path: str | Path) -> Dict[str, Any]:
    """Đọc file YAML cấu hình."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    if not config_file.exists():
        return {}
    with open(config_file, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    if isinstance(cfg, dict):
        return cfg
    raise ValueError(f"Cấu hình {config_file} phải là một dictionary.")


def normalize_label(value: Any) -> str:
    """Chuẩn hóa nhãn về positive | negative | mixed."""
    if value is None:
        return "mixed"
    label = str(value).strip().lower()
    if label in {"pos", "positive", "good"}:
        return "positive"
    if label in {"neg", "negative", "bad"}:
        return "negative"
    if label in {"mixed", "neutral", "neural"}:
        return "mixed"
    return "mixed"


def normalize_text(value: Any) -> str:
    """Chuẩn hóa chuỗi review để so khớp text."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def find_label_file(config: Dict[str, Any]) -> Path:
    """Tìm file nhãn thủ công mặc định.

    Hỗ trợ các tên file phổ biến:
    - data/processed/human_labeled_reviews.csv
    - data/processed/reviews_labeled.csv
    - data/processed/labeled_reviews.csv
    - data/processed/review_labels.csv
    - data/processed/human_labels.csv
    """
    processed_root = Path(config.get("processed_dir") or "data/processed")
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root

    candidates = [
        processed_root / "human_labeled_reviews.csv",
        processed_root / "reviews_labeled.csv",
        processed_root / "labeled_reviews.csv",
        processed_root / "review_labels.csv",
        processed_root / "human_labels.csv",
        processed_root / "ground_truth.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Không tìm thấy file đánh nhãn thủ công. Hãy tạo file như: "
        "data/processed/human_labeled_reviews.csv với cột review_text hoặc comments và sentiment."
    )


def load_human_labels(labels_path: str | Path) -> pd.DataFrame:
    """Đọc file nhãn thủ công và chuẩn hóa về cột cần thiết."""
    df = pd.read_csv(labels_path)

    if "sentiment" not in df.columns:
        for alt in ["label", "gold_label", "ground_truth", "target", "class"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "sentiment"})
                break
    if "sentiment" not in df.columns:
        raise ValueError(f"File {labels_path} thiếu cột 'sentiment' (hoặc alias tương đương).")

    if "review_text" in df.columns:
        df = df.rename(columns={"review_text": "review_text"})
    elif "comments" in df.columns:
        df = df.rename(columns={"comments": "review_text"})
    elif "comment" in df.columns:
        df = df.rename(columns={"comment": "review_text"})
    elif "text" in df.columns:
        df = df.rename(columns={"text": "review_text"})

    if "review_text" not in df.columns:
        raise ValueError(f"File {labels_path} thiếu cột review text như 'review_text' / 'comments'.")

    df = df[["review_text", "sentiment"] + [c for c in df.columns if c not in {"review_text", "sentiment"}]]
    df["sentiment"] = df["sentiment"].map(normalize_label)

    if "review_id" not in df.columns:
        df["review_id"] = df.index.astype(str)
    if "id" not in df.columns:
        df["id"] = df["review_id"]

    df["review_text_norm"] = df["review_text"].map(normalize_text)
    return df


def load_prediction_csv(path: str | Path) -> pd.DataFrame:
    """Đọc output của baseline/llm và chuẩn hóa.

    Cột mong đợi tối thiểu: sentiment và review_text.
    Nếu có review_id/id thì giữ lại.
    """
    df = pd.read_csv(path)
    if "sentiment" not in df.columns:
        for alt in ["label", "predicted_label", "prediction"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "sentiment"})
                break
    if "sentiment" not in df.columns:
        raise ValueError(f"File {path} thiếu cột 'sentiment'.")

    if "review_text" not in df.columns:
        for alt in ["comments", "comment", "text"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "review_text"})
                break
    if "review_text" not in df.columns:
        # Nếu không có review_text, xem như chỉ có review_id + label, vẫn có thể join bằng review_id.
        if "review_id" in df.columns or "id" in df.columns:
            df["review_text"] = ""
        else:
            raise ValueError(f"File {path} thiếu cột review_text/comment/text để so khớp dữ liệu.")

    df["sentiment"] = df["sentiment"].map(normalize_label)
    if "review_id" not in df.columns:
        df["review_id"] = df.index.astype(str)
    if "id" not in df.columns:
        df["id"] = df["review_id"]
    df["review_text_norm"] = df["review_text"].map(normalize_text)
    return df


def align_datasets(human_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Ghép dữ liệu người gán nhãn với dự đoán theo review_id / id / text.

    Trả về DataFrame gồm: review_id, review_text, true_label, pred_label.
    """
    merged: Optional[pd.DataFrame] = None

    # 1) Join theo review_id trước
    join_cols = [c for c in ["review_id", "id"] if c in human_df.columns and c in pred_df.columns]
    if join_cols:
        key_col = join_cols[0]
        merged = human_df[["review_id", "id", "review_text", "review_text_norm", "sentiment"]].merge(
            pred_df[[key_col, "sentiment", "review_text_norm"]].rename(columns={"sentiment": "pred_sentiment"}),
            how="inner",
            on=key_col,
        )

    # 2) Nếu không join bằng id, thử join theo normalized text
    if merged is None or merged.empty:
        merged = human_df[["review_id", "id", "review_text", "review_text_norm", "sentiment"]].merge(
            pred_df[["review_id", "id", "review_text_norm", "sentiment"]].rename(columns={"sentiment": "pred_sentiment"}),
            how="inner",
            on="review_text_norm",
        )

    if merged is None or merged.empty:
        raise ValueError("Không thể ghép nhãn thủ công với dự đoán. Kiểm tra review_id hoặc review_text.")

    # Nếu merge tạo ra nhiều dòng lặp cho cùng review_text, giữ bản ghi duy nhất tốt nhất.
    merged = merged.drop_duplicates(subset=["review_id", "review_text_norm"]).copy()

    if "pred_sentiment" not in merged.columns:
        merged["pred_sentiment"] = merged.get("sentiment_y", merged.get("sentiment_x", "mixed"))

    merged = merged.rename(columns={"sentiment_x": "true_label", "sentiment_y": "pred_label"})
    if "true_label" not in merged.columns:
        if "sentiment" in merged.columns:
            merged = merged.rename(columns={"sentiment": "true_label"})
    if "pred_label" not in merged.columns:
        if "pred_sentiment" in merged.columns:
            merged = merged.rename(columns={"pred_sentiment": "pred_label"})

    if "true_label" not in merged.columns or "pred_label" not in merged.columns:
        raise ValueError("Sau khi ghép dữ liệu, thiếu cột true_label / pred_label.")

    merged["true_label"] = merged["true_label"].map(normalize_label)
    merged["pred_label"] = merged["pred_label"].map(normalize_label)
    return merged[["review_id", "review_text", "true_label", "pred_label"]].reset_index(drop=True)


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Tính các metric quan trọng cho classification."""
    y_true = [normalize_label(x) for x in y_true]
    y_pred = [normalize_label(x) for x in y_pred]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }


def analyze_errors(df: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
    """Trích các trường hợp phân loại sai để giải thích lỗi."""
    errors = df[df["true_label"] != df["pred_label"]].copy()
    if errors.empty:
        return {"total_errors": 0, "by_pair": {}, "examples": []}

    errors["pair"] = errors["true_label"] + "->" + errors["pred_label"]
    by_pair = (
        errors.groupby("pair").size().sort_values(ascending=False).to_dict()
    )

    examples = []
    for _, row in errors.head(top_n).iterrows():
        examples.append(
            {
                "review_id": row.get("review_id", ""),
                "true_label": row.get("true_label", ""),
                "pred_label": row.get("pred_label", ""),
                "review_text": row.get("review_text", ""),
            }
        )

    return {
        "total_errors": int(len(errors)),
        "by_pair": by_pair,
        "examples": examples,
    }


def evaluate_method(
    labels_path: str | Path,
    pred_path: str | Path,
    output_dir: str | Path = "reports",
    top_errors: int = 20,
) -> Dict[str, Any]:
    """Đánh giá một phương pháp dự đoán so với nhãn thủ công."""
    human_df = load_human_labels(labels_path)
    pred_df = load_prediction_csv(pred_path)
    aligned = align_datasets(human_df, pred_df)

    metrics = compute_metrics(aligned["true_label"].tolist(), aligned["pred_label"].tolist())
    errors = analyze_errors(aligned, top_n=top_errors)

    result = {
        "method": Path(pred_path).stem,
        "n_samples": len(aligned),
        "metrics": metrics,
        "error_analysis": errors,
    }

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{Path(pred_path).stem}_evaluation.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def evaluate_all(
    labels_path: Optional[str | Path] = None,
    baseline_path: Optional[str | Path] = None,
    llm_path: Optional[str | Path] = None,
    output_dir: str | Path = "reports",
    top_errors: int = 20,
) -> Dict[str, Any]:
    """Đánh giá đồng thời baseline và LLM trên cùng tập dữ liệu."""
    config = read_config(DEFAULT_CONFIG)
    labels_path = labels_path or find_label_file(config)
    baseline_path = baseline_path or (PROJECT_ROOT / "data" / "processed" / "baseline_review_annotations.csv")
    llm_path = llm_path or (PROJECT_ROOT / "data" / "processed" / "llm_review_annotations.csv")

    baseline_result = evaluate_method(labels_path, baseline_path, output_dir=output_dir, top_errors=top_errors)
    llm_result = evaluate_method(labels_path, llm_path, output_dir=output_dir, top_errors=top_errors)

    summary = {
        "labels_file": str(labels_path),
        "n_labeled_samples": len(load_human_labels(labels_path)),
        "baseline": baseline_result,
        "llm": llm_result,
    }

    overall_path = Path(output_dir)
    if not overall_path.is_absolute():
        overall_path = PROJECT_ROOT / overall_path
    overall_path.mkdir(parents=True, exist_ok=True)
    summary_path = overall_path / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Đánh giá baseline và LLM trên tập review được gán nhãn thủ công.")
    parser.add_argument("--labels", type=str, default=None, help="Đường dẫn file dữ liệu đã gán nhãn thủ công")
    parser.add_argument("--baseline", type=str, default=None, help="Đường dẫn file dự đoán của baseline")
    parser.add_argument("--llm", type=str, default=None, help="Đường dẫn file dự đoán của LLM")
    parser.add_argument("--output-dir", type=str, default="reports", help="Thư mục lưu báo cáo đánh giá")
    parser.add_argument("--top-errors", type=int, default=20, help="Số lỗi sai phân loại cần nêu ra trong báo cáo")
    args = parser.parse_args()

    result = evaluate_all(
        labels_path=args.labels,
        baseline_path=args.baseline,
        llm_path=args.llm,
        output_dir=args.output_dir,
        top_errors=args.top_errors,
    )

    print(json.dumps({
        "n_labeled_samples": result["n_labeled_samples"],
        "baseline": {
            "accuracy": result["baseline"]["metrics"]["accuracy"],
            "macro_f1": result["baseline"]["metrics"]["macro_f1"],
            "precision_macro": result["baseline"]["metrics"]["precision_macro"],
            "recall_macro": result["baseline"]["metrics"]["recall_macro"],
        },
        "llm": {
            "accuracy": result["llm"]["metrics"]["accuracy"],
            "macro_f1": result["llm"]["metrics"]["macro_f1"],
            "precision_macro": result["llm"]["metrics"]["precision_macro"],
            "recall_macro": result["llm"]["metrics"]["recall_macro"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
