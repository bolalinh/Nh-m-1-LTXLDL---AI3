"""
src/logger.py – Module quản lý cấu hình Logging độc lập
=========================================================

MỤC ĐÍCH:
─────────
Tách biệt toàn bộ cấu hình nhật ký (logging) ra khỏi run_pipeline.py.
Giúp run_pipeline.py giữ nguyên sự tinh gọn, chỉ tập trung vào điều phối quy trình.

CƠ CHẾ:
───────
Chuyển hướng toàn bộ log chi tiết của hệ thống vào file (mặc định: log.txt).
Màn hình Terminal giữ sạch sẽ, chỉ hiển thị tiến trình tổng quan.
"""

import logging
from pathlib import Path


def setup_logging(log_file: str = "log.txt", level: int = logging.INFO) -> logging.Logger:
    """Khởi tạo cấu hình logging ghi ra file riêng biệt, không làm bẩn terminal."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Xóa các handler cũ nếu có
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Cấu hình FileHandler ghi vào file log.txt
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)

    return logging.getLogger("Pipeline")
