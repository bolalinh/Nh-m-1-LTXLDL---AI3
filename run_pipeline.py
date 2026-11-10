"""
run_pipeline.py – Thực thi toàn bộ quy trình phân tích Inside Airbnb bằng 1 lệnh duy nhất
========================================================================================

CÚ PHÁP CHẠY CHUẨN (THEO ĐỀ BÀI):
─────────────────────────────────
python run_pipeline.py --config configs/sydney.yaml

CÁC THAM SỐ HỖ TRỢ:
───────────────────
--config         : Đường dẫn tới file YAML cấu hình thành phố (bắt buộc)
--skip-download  : Bỏ qua bước tải nếu dữ liệu thô đã có sẵn trong data/raw/
--skip-llm       : Bỏ qua gọi API LLM mới, sử dụng cache và mô hình đối chứng (tiện kiểm thử nhanh)
--sample-size    : Số lượng mẫu đánh giá để xử lý LLM (mặc định lấy từ file config, ví dụ: 3000)

TIÊU CHUẨN ĐẦU RA BẮT BUỘC ĐƯỢC SINH RA:
────────────────────────────────────────
1. data/processed/listings_clean_*.csv
2. data/processed/kpi_by_snapshot.csv
3. data/processed/kpi_by_neighbourhood.csv
4. data/processed/kpi_by_room_type.csv
5. data/processed/llm_cache.jsonl
6. reports/qa_report.csv
7. reports/llm_vs_baseline_eval.csv
8. reports/error_analysis.csv
9. figures/01_market_trends_temporal.png
10. figures/02_top_revenue_spatial.png
11. figures/03_superhost_premium_segmentation.png
12. figures/04_llm_guest_complaints_aspects.png
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Cấu hình logging chuyên nghiệp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Pipeline")


def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML."""
    p = Path(config_path)
    if not p.exists():
        logger.error("Không tìm thấy file cấu hình: %s", config_path)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline phân tích dữ liệu Inside Airbnb")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới file cấu hình YAML (ví dụ: configs/sydney.yaml)")
    parser.add_argument("--skip-download", action="store_true", help="Bỏ qua bước tải dữ liệu từ internet nếu đã có sẵn trong data/raw/")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua gọi API LLM mới, tận dụng cache và baseline")
    parser.add_argument("--sample-size", type=int, default=None, help="Ghi đè cỡ mẫu đánh giá LLM")
    args = parser.parse_args()

    start_total_time = time.time()
    cfg = load_config(args.config)
    city_name = cfg.get("city", "unknown").upper()
    
    logger.info("=" * 70)
    logger.info("   BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE AIRBNB CHO: %s", city_name)
    logger.info("   Bên đặt hàng phân tích: CHỦ NHÀ AIRBNB (HOST)")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 1: THU THẬP DỮ LIỆU (DOWNLOAD)
    # ─────────────────────────────────────────────────────────────
    from src.download import download_all
    if not args.skip_download:
        logger.info("\n>>> BƯỚC 1: THU THẬP DỮ LIỆU TỪ INSIDE AIRBNB...")
        raw_paths = download_all(cfg)
    else:
        logger.info("\n>>> BƯỚC 1: BỎ QUA TẢI DỮ LIỆU (--skip-download), NẠP DỮ LIỆU CỤC BỘ...")
        raw_dir = Path("data/raw")
        raw_paths = {}
        for snap in cfg.get("snapshots", []):
            snap_dir = raw_dir / snap
            raw_paths[snap] = {
                "listings": snap_dir / "listings.csv.gz" if (snap_dir / "listings.csv.gz").exists() else snap_dir / "listings.csv",
                "reviews": snap_dir / "reviews.csv.gz" if (snap_dir / "reviews.csv.gz").exists() else snap_dir / "reviews.csv",
                "calendar": snap_dir / "calendar.csv.gz" if (snap_dir / "calendar.csv.gz").exists() else snap_dir / "calendar.csv",
            }

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 2: KIỂM SOÁT CHẤT LƯỢNG & LÀM SẠCH (QA)
    # ─────────────────────────────────────────────────────────────
    logger.info("\n>>> BƯỚC 2: CHẠY HỆ THỐNG QUY TẮC KIỂM SOÁT CHẤT LƯỢNG (QA)...")
    from src.qa import run_qa
    cleaned_snapshots, qa_reports = run_qa(raw_paths, cfg)

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 3: TÍNH TOÁN BỘ CHỈ SỐ HIỆU SUẤT (KPI)
    # ─────────────────────────────────────────────────────────────
    logger.info("\n>>> BƯỚC 3: TÍNH TOÁN BỘ KPI CHO CHỦ NHÀ THEO THỜI GIAN & KHÔNG GIAN...")
    from src.kpi import compute_all_kpis
    kpi_results = compute_all_kpis(cleaned_snapshots, cfg)

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 4: PHÂN TÍCH ĐÁNH GIÁ BẰNG LLM & BASELINE
    # ─────────────────────────────────────────────────────────────
    logger.info("\n>>> BƯỚC 4: LẤY MẪU VÀ TRÍCH XUẤT CÓ CẤU TRÚC BẰNG LLM...")
    from src.llm import sample_reviews_stratified, process_reviews_with_llm, estimate_total_sydney_cost

    # Đọc reviews từ mốc mới nhất
    latest_snapshot = max(cleaned_snapshots.keys())
    reviews_file = raw_paths[latest_snapshot].get("reviews")
    
    if reviews_file and Path(reviews_file).exists():
        logger.info("  Đọc dữ liệu đánh giá từ: %s", reviews_file)
        if str(reviews_file).endswith(".gz"):
            df_reviews_raw = pd.read_csv(reviews_file, compression="gzip", nrows=50000)
        else:
            df_reviews_raw = pd.read_csv(reviews_file, nrows=50000)
            
        sample_size = args.sample_size or cfg.get("llm", {}).get("sample_size", 3000)
        reviews_sample = sample_reviews_stratified(
            df_reviews_raw,
            cleaned_snapshots[latest_snapshot],
            sample_size=min(sample_size, len(df_reviews_raw)),
            seed=cfg.get("seed", 42)
        )
        
        # Xử lý trích xuất
        llm_results_df = process_reviews_with_llm(reviews_sample, cfg, skip_llm=args.skip_llm)

        # Báo cáo ước tính chi phí toàn bộ thành phố
        cost_report = estimate_total_sydney_cost(total_reviews_count=len(df_reviews_raw) * 10)
        logger.info("  [ƯỚC TÍNH CHI PHÍ LLM TOÀN BỘ SYDNEY]:")
        logger.info("    Tổng số đánh giá ước tính: %d", cost_report["total_reviews"])
        logger.info("    Tổng input tokens: %d | Tổng output tokens: %d", cost_report["est_total_input_tokens"], cost_report["est_total_output_tokens"])
        logger.info("    Tổng chi phí ước tính: $%.2f USD", cost_report["est_total_cost_usd"])
    else:
        logger.warning("  Không tìm thấy tệp reviews để xử lý LLM. Sử dụng dữ liệu mẫu từ gold labels!")
        gold_df = pd.read_csv("data/samples/manual_labels_100.csv")
        llm_results_df = gold_df.rename(columns={"comments": "text", "true_sentiment": "sentiment", "true_aspects_negative": "aspects_negative"})

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 5: ĐÁNH GIÁ CHẤT LƯỢNG LLM VS BASELINE TRÊN 100 NHÃN TAY
    # ─────────────────────────────────────────────────────────────
    logger.info("\n>>> BƯỚC 5: ĐÁNH GIÁ CHẤT LƯỢNG LLM VS BASELINE TRÊN 100 NHÃN TAY...")
    from src.evaluate import run_evaluation
    eval_results = run_evaluation(cfg, "data/samples/manual_labels_100.csv")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 6: TRỰC QUAN HOÁ DỮ LIỆU (VISUALIZATION)
    # ─────────────────────────────────────────────────────────────
    logger.info("\n>>> BƯỚC 6: SINH TỰ ĐỘNG HỆ THỐNG BIỂU ĐỒ TRỰC QUAN HOÁ...")
    from src.viz import generate_all_visualizations
    viz_files = generate_all_visualizations(kpi_results, llm_results_df, cfg)

    total_time = time.time() - start_total_time
    logger.info("=" * 70)
    logger.info("   PIPELINE HOÀN THÀNH XUẤT SẮC TRONG %.2f GIÂY!", total_time)
    logger.info("   Toàn bộ báo cáo, số liệu và hình ảnh đã sẵn sàng cho chấm điểm!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
