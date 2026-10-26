"""
Pipeline orchestration cho project Airbnb.

Thứ tự thực thi:
1. download
2. qa
3. kpi
4. baseline
5. llm
6. evaluate
7. viz

Mục tiêu:
- kiểm tra đầu vào/đầu ra thống nhất
- bỏ qua bước nếu thiếu API/key hoặc file dữ liệu
- tạo output tập trung trong data/processed và figures
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.download import run_download
    from src.qa import run_qa
    from src.kpi import run_kpi
    from src.baseline import run_baseline
    from src.llm import run_llm_annotation, get_model_and_api_key
    from src.evaluate import evaluate_all
    from src.viz import build_business_charts
except Exception as exc:  # pragma: no cover
    print(f"[PIPELINE] Không import được module cần thiết: {exc}")
    raise


def resolve_path(path_value: str | Path | None, default: str = "") -> Path:
    candidate = Path(path_value) if path_value is not None else Path(default)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def step_header(name: str) -> None:
    print("\n" + "=" * 80)
    print(f"[PIPELINE] {name}")
    print("=" * 80)


def check_required_files(config_path: str | Path) -> List[str]:
    """Kiểm tra file đầu vào cơ bản theo config snapshots."""
    config_path = resolve_path(config_path)
    if not config_path.exists():
        return [f"missing_config:{config_path}"]

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        return [f"config_parse_error:{exc}"]

    missing: List[str] = []
    data_root = str(cfg.get("data_root", "data/raw"))
    root = resolve_path(data_root)
    snapshots = [str(s) for s in cfg.get("snapshots", [])]
    if not snapshots:
        missing.append("no_snapshots_in_config")
    for snap in snapshots:
        snap_dir = root / str(snap)
        if not snap_dir.exists():
            missing.append(f"missing_snapshot_dir:{snap}")
            continue
        for filename in ["listings.csv", "calendar.csv", "reviews.csv"]:
            if not (snap_dir / filename).exists():
                missing.append(f"missing_file:{snap}/{filename}")
    return missing


def run_pipeline(
    config_path: str | Path = "configs/sydney.yaml",
    skip_download: bool = False,
    skip_qa: bool = False,
    skip_kpi: bool = False,
    skip_baseline: bool = False,
    skip_llm: bool = False,
    skip_evaluate: bool = False,
    skip_viz: bool = False,
    max_llm_samples: int = 300,
    sample_size_per_group: int = 25,
):
    config_path = resolve_path(config_path)
    project_root = PROJECT_ROOT
    data_processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "figures"

    print(f"[PIPELINE] Project root: {project_root}")
    print(f"[PIPELINE] Config: {config_path}")

    missing_inputs = check_required_files(config_path)
    if missing_inputs:
        print("[PIPELINE] Thiếu/không hợp lệ một số đầu vào ban đầu:")
        for item in missing_inputs:
            print(f"  - {item}")

    if not skip_download:
        step_header("DOWNLOAD")
        try:
            run_download(str(config_path))
        except Exception as exc:
            print(f"[PIPELINE][DOWNLOAD] Lỗi: {exc}")

    if not skip_qa:
        step_header("QA")
        try:
            run_qa(config_path=str(config_path), clean=True)
        except Exception as exc:
            print(f"[PIPELINE][QA] Lỗi: {exc}")

    if not skip_kpi:
        step_header("KPI")
        try:
            run_kpi(config_path=str(config_path), processed_dir="data/processed", output_dir="data/processed")
        except FileNotFoundError as exc:
            print(f"[PIPELINE][KPI] Dữ liệu đầu vào thiếu: {exc}")
        except Exception as exc:
            print(f"[PIPELINE][KPI] Lỗi: {exc}")

    if not skip_baseline:
        step_header("BASELINE")
        try:
            run_baseline(config_path=str(config_path), output_dir=data_processed_dir, max_rows=None)
        except FileNotFoundError as exc:
            print(f"[PIPELINE][BASELINE] Dữ liệu đầu vào thiếu: {exc}")
        except Exception as exc:
            print(f"[PIPELINE][BASELINE] Lỗi: {exc}")

    if not skip_llm:
        step_header("LLM")
        try:
            api_model, api_key = get_model_and_api_key({"llm_params": {"model": "gpt-4o-mini"}})
            if not api_key:
                raise ValueError("Không tìm thấy API key trong environment / .env.")
            run_llm_annotation(
                config_path=str(config_path),
                sample_size_per_group=sample_size_per_group,
                max_total=max_llm_samples,
                output_dir=data_processed_dir,
            )
        except FileNotFoundError as exc:
            print(f"[PIPELINE][LLM] Dữ liệu đầu vào thiếu: {exc}")
        except ValueError as exc:
            print(f"[PIPELINE][LLM] Bỏ qua vì thiếu key hoặc config: {exc}")
        except Exception as exc:
            print(f"[PIPELINE][LLM] Lỗi API hoặc parse JSON: {exc}")

    if not skip_evaluate:
        step_header("EVALUATE")
        baseline_path = data_processed_dir / "baseline_review_annotations.csv"
        llm_path = data_processed_dir / "llm_review_annotations.csv"

        if baseline_path.exists() or llm_path.exists():
            try:
                evaluate_all(
                    labels_path=None,
                    baseline_path=baseline_path if baseline_path.exists() else None,
                    llm_path=llm_path if llm_path.exists() else None,
                    output_dir=project_root / "reports",
                    top_errors=20,
                )
            except Exception as exc:
                print(f"[PIPELINE][EVALUATE] Lỗi: {exc}")
        else:
            print("[PIPELINE][EVALUATE] Không có file dự đoán baseline/llm để đánh giá.")

    if not skip_viz:
        step_header("VIZ")
        try:
            output_summary = build_business_charts(config_path=str(config_path), output_dir=str(figures_dir))
            print(f"[PIPELINE][VIZ] Đã tạo {len(output_summary.get('plots', []))} chart(s):")
            for p in output_summary.get('plots', []):
                print(f"  - {p}")
        except Exception as exc:
            print(f"[PIPELINE][VIZ] Lỗi: {exc}")

    print("\n[PIPELINE] HOÀN THÀNH CHẠY QUY TRÌNH")
    return {
        "config": str(config_path),
        "processed_dir": str(data_processed_dir),
        "figures_dir": str(figures_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline điều phối toàn bộ quá trình xử lý Airbnb.")
    parser.add_argument("--config", type=str, default="configs/sydney.yaml", help="Đường dẫn file YAML cấu hình")
    parser.add_argument("--skip-download", action="store_true", help="Bỏ qua bước download")
    parser.add_argument("--skip-qa", action="store_true", help="Bỏ qua bước QA")
    parser.add_argument("--skip-kpi", action="store_true", help="Bỏ qua bước KPI")
    parser.add_argument("--skip-baseline", action="store_true", help="Bỏ qua baseline")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua LLM")
    parser.add_argument("--skip-evaluate", action="store_true", help="Bỏ qua đánh giá")
    parser.add_argument("--skip-viz", action="store_true", help="Bỏ qua tạo chart")
    parser.add_argument("--max-llm-samples", type=int, default=300, help="Số review tối đa gửi cho LLM")
    parser.add_argument("--sample-size-per-group", type=int, default=25, help="Số review tối đa mỗi nhóm khi chọn mẫu")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        skip_download=args.skip_download,
        skip_qa=args.skip_qa,
        skip_kpi=args.skip_kpi,
        skip_baseline=args.skip_baseline,
        skip_llm=args.skip_llm,
        skip_evaluate=args.skip_evaluate,
        skip_viz=args.skip_viz,
        max_llm_samples=args.max_llm_samples,
        sample_size_per_group=args.sample_size_per_group,
    )
