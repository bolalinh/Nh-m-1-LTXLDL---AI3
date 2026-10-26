"""Module trực quan hóa dữ liệu để trả lời các câu hỏi kinh doanh.

Những chart ưu tiên:
1. KPI theo snapshot (điểm giá, occupancy, review rate, ...)
2. KPI theo khu vực / room type
3. Phân bố sentiment review
4. Aspect được khen/chê nhiều nhất
5. So sánh baseline vs LLM

Các chart cuối cùng phụ thuộc vào dữ liệu thực tế có trong dataset và output của
kpi.py / llm.py / baseline.py. Module này nên tự động kiểm tra cột tồn tại trước khi vẽ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sydney.yaml"


def read_config(config_path: str | Path) -> Dict[str, Any]:
    """Đọc file cấu hình YAML."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    if not config_file.exists():
        return {}
    import yaml
    with open(config_file, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_kpi_tables(processed_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Đọc các file KPI đầu ra từ kpi.py khi có sẵn."""
    processed_dir = Path(processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir

    tables: Dict[str, pd.DataFrame] = {}
    candidates = {
        "kpi_summary": "kpi_summary.csv",
        "kpi_by_time": "kpi_by_time.csv",
        "kpi_by_area": "kpi_by_area.csv",
        "kpi_by_room_type": "kpi_by_room_type.csv",
        "kpi_review_rate": "kpi_review_rate.csv",
        "kpi_review_score": "kpi_review_score.csv",
        "kpi_listings_per_host": "kpi_listings_per_host.csv",
    }
    for name, filename in candidates.items():
        path = processed_dir / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def load_annotation_tables(processed_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Đọc bảng annotation từ baseline.py / llm.py nếu có."""
    processed_dir = Path(processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir

    tables: Dict[str, pd.DataFrame] = {}
    candidates = {
        "baseline": "baseline_review_annotations.csv",
        "llm": "llm_review_annotations.csv",
    }
    for name, filename in candidates.items():
        path = processed_dir / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def plot_kpi_by_snapshot(df: pd.DataFrame, output_dir: Path) -> None:
    """So sánh các KPI theo snapshot.

    Ưu tiên các cột như:
    median_price_aud, average_price_clean_aud, availability_rate_pct,
    estimated_occupancy_rate_pct, review_rate_pct, avg_review_score_100,
    listings_per_host.
    """
    if df.empty:
        return
    numeric_cols = [
        c for c in [
            "median_price_aud",
            "average_price_clean_aud",
            "availability_rate_pct",
            "estimated_occupancy_rate_pct",
            "review_rate_pct",
            "avg_review_score_100",
            "listings_per_host",
        ] if c in df.columns
    ]
    if not numeric_cols or "snapshot" not in df.columns:
        return

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        ax.plot(df["snapshot"], df[col], marker="o", linewidth=2)
        ax.set_title(f"{col} theo snapshot")
        ax.set_xlabel("Snapshot")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.setp(ax.get_xticklabels(), rotation=30)

    fig.tight_layout()
    out = output_dir / "kpi_by_snapshot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_kpi_by_area(df: pd.DataFrame, output_dir: Path) -> None:
    """Biểu đồ KPI theo khu vực."""
    if df.empty:
        return
    area_col = "analysis_area" if "analysis_area" in df.columns else "neighbourhood_cleansed"
    if area_col not in df.columns:
        return
    value_cols = [c for c in ["median_price_aud", "average_price_clean_aud", "review_rate_pct", "avg_review_score_100", "availability_rate_pct"] if c in df.columns]
    if not value_cols:
        return

    fig, axes = plt.subplots(len(value_cols), 1, figsize=(10, 4 * len(value_cols)))
    if len(value_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, value_cols):
        sns.barplot(data=df, x=area_col, y=col, ax=ax, palette="viridis")
        ax.set_title(f"{col} theo khu vực")
        ax.set_xlabel("Khu vực")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out = output_dir / "kpi_by_area.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_kpi_by_room_type(df: pd.DataFrame, output_dir: Path) -> None:
    """Biểu đồ KPI theo loại phòng."""
    if df.empty or "room_type" not in df.columns:
        return
    cols = [c for c in ["average_price_clean_aud", "availability_rate_pct", "review_rate_pct", "avg_review_score_100"] if c in df.columns]
    if not cols:
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 4 * len(cols)))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sns.barplot(data=df, x="room_type", y=col, ax=ax, palette="magma")
        ax.set_title(f"{col} theo room type")
        ax.set_xlabel("Loại phòng")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out = output_dir / "kpi_by_room_type.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sentiment_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Phân bố sentiment trong review."""
    if df.empty:
        return
    sentiment_col = "sentiment" if "sentiment" in df.columns else None
    if sentiment_col is None:
        return

    counts = df[sentiment_col].value_counts().reindex(["positive", "mixed", "negative"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#f1c40f", "#e74c3c"])
    ax.set_title("Phân bố sentiment review")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Số review")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out = output_dir / "sentiment_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aspect_frequency(df: pd.DataFrame, output_dir: Path, column_name: str = "aspects") -> None:
    """Biểu đồ các khía cạnh được khen/chê nhiều nhất."""
    if df.empty or column_name not in df.columns:
        return

    aspect_list = []
    for value in df[column_name].dropna():
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = value
            if isinstance(parsed, list):
                aspect_list.extend(parsed)
            else:
                aspect_list.append(parsed)
        elif isinstance(value, list):
            aspect_list.extend(value)

    if not aspect_list:
        return

    aspect_counts = pd.Series(aspect_list).str.strip().str.lower().value_counts()
    top = aspect_counts.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Các khía cạnh nổi bật nhất trong review")
    ax.set_xlabel("Số lần xuất hiện")
    ax.set_ylabel("Khía cạnh")
    fig.tight_layout()
    out = output_dir / "aspect_frequency.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(baseline_df: pd.DataFrame, llm_df: pd.DataFrame, output_dir: Path) -> None:
    """So sánh baseline và LLM theo sentiment distribution."""
    if baseline_df.empty or llm_df.empty:
        return
    for name, df in {"baseline": baseline_df, "llm": llm_df}.items():
        if "sentiment" not in df.columns:
            continue
        if name == "baseline":
            base_counts = df["sentiment"].value_counts().reindex(["positive", "mixed", "negative"], fill_value=0)
        else:
            llm_counts = df["sentiment"].value_counts().reindex(["positive", "mixed", "negative"], fill_value=0)

    if "base_counts" in locals() and "llm_counts" in locals():
        combined = pd.concat([base_counts.rename("baseline"), llm_counts.rename("llm")], axis=1)
        fig, ax = plt.subplots(figsize=(10, 5))
        combined.plot(kind="bar", ax=ax)
        ax.set_title("So sánh phân bố sentiment: Baseline vs LLM")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Số review")
        ax.legend()
        fig.tight_layout()
        out = output_dir / "method_comparison.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)


def build_business_charts(config_path: str | Path = DEFAULT_CONFIG, output_dir: str | Path = "figures") -> Dict[str, List[str]]:
    """Tạo tập biểu đồ phù hợp với câu hỏi kinh doanh dựa trên dữ liệu sẵn có."""
    config = read_config(config_path)
    processed_dir = config.get("processed_dir", "data/processed")
    output_dir = ensure_dir(output_dir)

    kpi_tables = load_kpi_tables(processed_dir)
    annotation_tables = load_annotation_tables(processed_dir)
    created: Dict[str, List[str]] = {"plots": []}

    if "kpi_by_time" in kpi_tables:
        plot_kpi_by_snapshot(kpi_tables["kpi_by_time"], output_dir)
        created["plots"].append("kpi_by_snapshot.png")

    if "kpi_by_area" in kpi_tables:
        plot_kpi_by_area(kpi_tables["kpi_by_area"], output_dir)
        created["plots"].append("kpi_by_area.png")

    if "kpi_by_room_type" in kpi_tables:
        plot_kpi_by_room_type(kpi_tables["kpi_by_room_type"], output_dir)
        created["plots"].append("kpi_by_room_type.png")

    if "baseline" in annotation_tables and "llm" in annotation_tables:
        plot_method_comparison(annotation_tables["baseline"], annotation_tables["llm"], output_dir)
        created["plots"].append("method_comparison.png")

    if "baseline" in annotation_tables:
        plot_sentiment_distribution(annotation_tables["baseline"], output_dir)
        created["plots"].append("sentiment_distribution.png")
    elif "llm" in annotation_tables:
        plot_sentiment_distribution(annotation_tables["llm"], output_dir)
        created["plots"].append("sentiment_distribution.png")

    if "baseline" in annotation_tables:
        plot_aspect_frequency(annotation_tables["baseline"], output_dir)
        created["plots"].append("aspect_frequency.png")
    elif "llm" in annotation_tables:
        plot_aspect_frequency(annotation_tables["llm"], output_dir)
        created["plots"].append("aspect_frequency.png")

    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo các biểu đồ phục vụ phân tích Airbnb theo KPI và review sentiment.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="File YAML cấu hình")
    parser.add_argument("--output-dir", type=str, default="figures", help="Thư mục lưu chart")
    args = parser.parse_args()

    result = build_business_charts(args.config, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
