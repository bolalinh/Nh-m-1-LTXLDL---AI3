"""
src/logger.py – Module quản lý Logger chuẩn mực UET
===================================================

THIẾT KẾ DỰA TRÊN QUY CHUẨN BÀI TẬP LỚN VIỆN TRÍ TUỆ NHÂN TẠO (UET):
──────────────────────────────────────────────────────────────────────
1. Hạn chế tối đa dùng print/std::cout bừa bãi làm bẩn màn hình.
2. Phân cấp mức độ ghi log rõ ràng:
   - INFO: Các cột mốc quan trọng trong luồng xử lý dữ liệu.
   - DEBUG: Dữ liệu trung gian & thời gian thực thi chi tiết từng hàm.
   - WARNING / ERROR: Cảnh báo bất thường dữ liệu và lỗi phát sinh.
3. Tích hợp công cụ đo lường thời gian thực thi (Performance Profiling)
   tương đương hàm measureExecutionTime() trong tài liệu UET.
4. Hỗ trợ cấu hình Run Configuration: xuất ra console hoặc chuyển hướng
   hoàn toàn vào file log.txt để phục vụ chấm điểm tự động (Judge Mode).
"""

import functools
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Any

# Root logger dùng chung cho toàn bộ dự án
LOGGER_NAME = "AirbnbPipeline"


class PipelineLogger:
    """Bộ quản lý cấu hình Logging mô phỏng theo GameLogger trong tài liệu UET."""

    @staticmethod
    def setup(
        log_file: str = "log.txt",
        to_console: bool = False,
        debug_mode: bool = True
    ) -> logging.Logger:
        """Khởi tạo hệ thống logging theo cấu hình RunConfig."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Xóa các handler cũ nếu có
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

        level = logging.DEBUG if debug_mode else logging.INFO
        root.setLevel(level)

        # Giữ mức WARNING cho các thư viện bên thứ 3 để tránh làm loãng file log
        for noisy in ["matplotlib", "PIL", "urllib3"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Định dạng chuẩn mực: Thời gian | Cấp độ | Module | Nội dung
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] [%(name)s]: %(message)s",
            datefmt="%H:%M:%S"
        )

        # 1. FileHandler: Luôn ghi toàn bộ chi tiết (cả DEBUG và INFO) vào file log
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        # 2. ConsoleHandler: Chỉ bật nếu to_console=True
        if to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            root.addHandler(console_handler)

        logger = logging.getLogger(LOGGER_NAME)
        logger.info("=== HỆ THỐNG LOGGING ĐƯỢC KHỞI TẠO (File: %s) ===", log_file)
        return logger


def measure_execution_time(label: str = None) -> Callable:
    """Decorator đo thời gian thực thi tương đương measureExecutionTime() trong tài liệu UET.
    
    Tự động đo thời gian chạy của hàm và ghi lại ở mức độ DEBUG trong log.
    Giúp tối ưu hiệu năng và là minh chứng khoa học khi viết báo cáo.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = label or func.__name__
            logger = logging.getLogger(f"Timer.{func_name}")
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            
            # Ghi nhận vào log ở mức DEBUG theo đúng chuẩn mục VI.2 tài liệu UET
            logger.debug("execution time of [%s()] = %.4fs", func_name, duration)
            return result
        return wrapper
    return decorator


def setup_logging(
    log_file: str = "log.txt",
    to_console: bool = False,
    debug_mode: bool = True
) -> logging.Logger:
    """Hàm tiện ích tương thích ngược để khởi tạo nhanh logging."""
    return PipelineLogger.setup(log_file=log_file, to_console=to_console, debug_mode=debug_mode)

