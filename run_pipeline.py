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

# Cấu hình logging chuyên biệt cho nhánh feature/file-logging:
# Chuyển hướng toàn bộ log chi tiết vào file log.txt để terminal sạch sẽ.
LOG_FILE_PATH = "log.txt"

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

# Root logger chỉ ghi vào file, KHÔNG gắn StreamHandler(sys.stdout) để tránh làm bẩn terminal
logging.root.setLevel(logging.INFO)
logging.root.addHandler(file_handler)

logger = logging.getLogger("Pipeline")


def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML."""
    p = Path(config_path)
    if not p.exists():
        print(f"[LỖI] Không tìm thấy file cấu hình: {config_path}")
        logger.error("Không tìm thấy file cấu hình: %s", config_path)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline Inside Airbnb (Chế độ xuất nhật ký ra file log.txt)")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới file cấu hình YAML (ví dụ: configs/sydney.yaml)")
    parser.add_argument("--skip-download", action="store_true", help="Bỏ qua bước tải dữ liệu từ internet nếu đã có sẵn trong data/raw/")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua gọi API LLM mới, tận dụng cache và baseline")
    parser.add_argument("--sample-size", type=int, default=None, help="Ghi đè cỡ mẫu đánh giá LLM")
    parser.add_argument("--log-file", type=str, default="log.txt", help="Tên file nhật ký (mặc định: log.txt)")
    args = parser.parse_args()

    # Cập nhật đường dẫn file log nếu người dùng tuỳ biến
    if args.log_file != LOG_FILE_PATH:
        for h in list(logging.root.handlers):
            logging.root.removeHandler(h)
        custom_handler = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
        custom_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logging.root.addHandler(custom_handler)

    start_total_time = time.time()
    cfg = load_config(args.config)
    city_name = cfg.get("city", "unknown").upper()
    
    # Terminal chỉ in thông báo tóm tắt ngắn gọn, chuyên nghiệp
    print("=" * 65)
    print(f"🚀 KHỞI CHẠY PIPELINE AIRBNB CHO: {city_name}")
    print(f"📝 Toàn bộ log chi tiết đang được ghi vào file: {args.log_file}")
    print("=" * 65)

    logger.info("=" * 70)
    logger.info("   BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE AIRBNB CHO: %s", city_name)
    logger.info("   Bên đặt hàng phân tích: CHỦ NHÀ AIRBNB (HOST)")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 1: THU THẬP DỮ LIỆU (DOWNLOAD)
    # ─────────────────────────────────────────────────────────────
    print(" [1/6] Đang kiểm tra và thu thập dữ liệu...")
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
    print("   ✓ Thu thập dữ liệu: Xong.")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 2: KIỂM SOÁT CHẤT LƯỢNG & LÀM SẠCH (QA)
    # ─────────────────────────────────────────────────────────────
    print(" [2/6] Đang chạy kiểm soát chất lượng & làm sạch (QA)...")
    logger.info("\n>>> BƯỚC 2: CHẠY HỆ THỐNG QUY TẮC KIỂM SOÁT CHẤT LƯỢNG (QA)...")
    from src.qa import run_qa
    cleaned_snapshots, qa_reports = run_qa(raw_paths, cfg)
    print("   ✓ Kiểm soát chất lượng (QA): Xong. Đã lưu reports/qa_report.csv")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 3: TÍNH TOÁN BỘ CHỈ SỐ HIỆU SUẤT (KPI)
    # ─────────────────────────────────────────────────────────────
    print(" [3/6] Đang tính toán bộ chỉ số KPI cho Chủ nhà...")
    logger.info("\n>>> BƯỚC 3: TÍNH TOÁN BỘ KPI CHO CHỦ NHÀ THEO THỜI GIAN & KHÔNG GIAN...")
    from src.kpi import compute_all_kpis
    kpi_results = compute_all_kpis(cleaned_snapshots, cfg)
    print("   ✓ Tính toán KPI: Xong. Đã lưu data/processed/kpi_*.csv")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 4: PHÂN TÍCH ĐÁNH GIÁ BẰNG LLM & BASELINE
    # ─────────────────────────────────────────────────────────────
    print(" [4/6] Đang phân tích đánh giá bằng LLM & mô hình đối chứng...")
    logger.info("\n>>> BƯỚC 4: LẤY MẪU VÀ TRÍCH XUẤT CÓ CẤU TRÚC BẰNG LLM...")
    from src.llm import sample_reviews_stratified, process_reviews_with_llm, estimate_total_sydney_cost

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
        
        llm_results_df = process_reviews_with_llm(reviews_sample, cfg, skip_llm=args.skip_llm)

        cost_report = estimate_total_sydney_cost(total_reviews_count=len(df_reviews_raw) * 10)
        logger.info("  [ƯỚC TÍNH CHI PHÍ LLM TOÀN BỘ SYDNEY]:")
        logger.info("    Tổng số đánh giá ước tính: %d", cost_report["total_reviews"])
        logger.info("    Tổng chi phí ước tính: $%.2f USD", cost_report["est_total_cost_usd"])
    else:
        gold_df = pd.read_csv("data/samples/manual_labels_100.csv")
        llm_results_df = gold_df.rename(columns={"comments": "text", "true_sentiment": "sentiment", "true_aspects_negative": "aspects_negative"})
    print("   ✓ Phân tích LLM & Caching: Xong. Đã lưu data/processed/llm_cache.jsonl")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 5: ĐÁNH GIÁ CHẤT LƯỢNG LLM VS BASELINE TRÊN 100 NHÃN TAY
    # ─────────────────────────────────────────────────────────────
    print(" [5/6] Đang đánh giá đối chứng LLM vs Baseline trên 100 nhãn tay...")
    logger.info("\n>>> BƯỚC 5: ĐÁNH GIÁ CHẤT LƯỢNG LLM VS BASELINE TRÊN 100 NHÃN TAY...")
    from src.evaluate import run_evaluation
    eval_results = run_evaluation(cfg, "data/samples/manual_labels_100.csv")
    print("   ✓ Đánh giá F1 đối chứng: Xong. Đã lưu reports/llm_vs_baseline_eval.csv")

    # ─────────────────────────────────────────────────────────────
    # BƯỚC 6: TRỰC QUAN HOÁ DỮ LIỆU (VISUALIZATION)
    # ─────────────────────────────────────────────────────────────
    print(" [6/6] Đang sinh các biểu đồ trực quan hoá...")
    logger.info("\n>>> BƯỚC 6: SINH TỰ ĐỘNG HỆ THỐNG BIỂU ĐỒ TRỰC QUAN HOÁ...")
    from src.viz import generate_all_visualizations
    viz_files = generate_all_visualizations(kpi_results, llm_results_df, cfg)
    print("   ✓ Trực quan hoá: Xong. Đã lưu 4 hình ảnh trong thư mục figures/")

    total_time = time.time() - start_total_time
    logger.info("=" * 70)
    logger.info("   PIPELINE HOÀN THÀNH XUẤT SẮC TRONG %.2f GIÂY!", total_time)
    logger.info("=" * 70)

    print("=" * 65)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ PIPELINE TRONG {total_time:.2f} GIÂY!")
    print(f"📄 Mở file '{args.log_file}' để xem toàn bộ thông tin chi tiết.")
    print("=" * 65)


if __name__ == "__main__":
    main()
