"""
src/download.py – Tải dữ liệu Inside Airbnb
=============================================

WHY (tại sao làm thế này):
─────────────────────────
1. Dùng script tự động thay vì tải tay → đảm bảo TÁI LẬP ĐƯỢC (reproducible).
   Giảng viên chạy lại trên mốc khác chỉ cần sửa config.
2. URL được DỰNG TỰ ĐỘNG từ config (data_root + snapshot date)
   → không ghi cứng URL trong code.
3. Kiểm tra file đã tồn tại trước khi tải → tiết kiệm bandwidth, idempotent.
4. Kiểm tra cột khoá sau khi tải → phát hiện sớm nếu schema thay đổi.

Kỹ thuật từ bài giảng:
─────────────────────
- Bài 6 (Kết nối & truy xuất dữ liệu ngoài): urllib/requests, pathlib
- Bài 2 (Python cơ bản): pathlib, f-string, exception handling
"""

import gzip
import hashlib
import logging
import ssl
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

import certifi
import requests
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Danh sách các file cần tải cho mỗi mốc chụp
DATA_FILES = [
    ("data/listings.csv.gz", "listings"),
    ("data/calendar.csv.gz", "calendar"),
    ("data/reviews.csv.gz", "reviews"),
]
GEO_FILE = ("visualisations/neighbourhoods.geojson", "neighbourhoods")

# Cột khoá BẮT BUỘC phải có trong mỗi bảng
# → nếu thiếu, dữ liệu bị lỗi hoặc schema đã thay đổi
REQUIRED_KEYS = {
    "listings": ["id", "name", "host_id", "neighbourhood_cleansed",
                 "latitude", "longitude", "price"],
    "calendar": ["listing_id", "date", "available", "price"],
    "reviews": ["listing_id", "id", "date", "comments"],
}

BASE_URL = "https://data.insideairbnb.com"


def _build_url(data_root: str, snapshot: str, file_path: str) -> str:
    """Dựng URL tải từ config.

    Ví dụ: https://data.insideairbnb.com/australia/nsw/sydney/2026-06-16/data/listings.csv.gz
    """
    return f"{BASE_URL}/{data_root}/{snapshot}/{file_path}"


def _download_one(url: str, dest: Path) -> Path:
    """Tải một file bằng requests stream kết hợp certifi để tránh lỗi SSL macOS.
    Bỏ qua nếu đã tồn tại (idempotent).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        logger.info("  ✓ Đã có: %s (%.1f MB)", dest.name, dest.stat().st_size / (1024 * 1024))
        return dest

    logger.info("  ↓ Đang tải: %s", url)
    try:
        # Sử dụng requests với stream=True để không load toàn bộ vào RAM
        resp = requests.get(url, stream=True, timeout=60, verify=certifi.where())
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("  ✓ Xong: %s (%.1f MB)", dest.name, size_mb)
    except Exception as e:
        logger.warning("  ⚠ Requests gặp lỗi (%s), thử fallback urllib unverified...", e)
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(url, context=ctx, timeout=60) as u_in, open(dest, "wb") as f_out:
                f_out.write(u_in.read())
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info("  ✓ Xong (fallback): %s (%.1f MB)", dest.name, size_mb)
        except Exception as e2:
            logger.error("  ✗ Lỗi không thể tải %s: %s", url, e2)
            if dest.exists():
                dest.unlink()
            raise e2
    return dest


def _verify_columns(file_path: Path, table_name: str) -> None:
    """Kiểm tra cột khoá có tồn tại trong file vừa tải.

    Tại sao chỉ đọc 5 dòng đầu?
    → Chỉ cần header để kiểm tra cột, không cần load cả triệu dòng.
      Tiết kiệm RAM và thời gian (Bài 4: lazy loading).
    """
    if table_name not in REQUIRED_KEYS:
        return

    required = REQUIRED_KEYS[table_name]
    try:
        if str(file_path).endswith(".csv.gz"):
            df_head = pd.read_csv(file_path, nrows=5, compression="gzip")
        elif str(file_path).endswith(".csv"):
            df_head = pd.read_csv(file_path, nrows=5)
        else:
            return  # geojson → bỏ qua

        missing = [c for c in required if c not in df_head.columns]
        if missing:
            logger.warning(
                "  ⚠ %s thiếu cột khoá: %s", file_path.name, missing
            )
        else:
            logger.info(
                "  ✓ %s: đủ %d cột khoá", file_path.name, len(required)
            )
    except Exception as e:
        logger.warning("  ⚠ Không đọc được header %s: %s", file_path.name, e)


def download_snapshot(data_root: str, snapshot: str, raw_dir: Path) -> dict:
    """Tải toàn bộ dữ liệu của MỘT mốc chụp.

    Returns:
        dict: mapping tên bảng → đường dẫn file đã tải
    """
    snapshot_dir = raw_dir / snapshot
    paths = {}

    logger.info("─── Mốc chụp: %s ───", snapshot)
    for file_path, table_name in DATA_FILES:
        url = _build_url(data_root, snapshot, file_path)
        dest = snapshot_dir / file_path.split("/")[-1]
        _download_one(url, dest)
        _verify_columns(dest, table_name)
        paths[table_name] = dest

    # GeoJSON chỉ cần tải 1 lần (không đổi giữa các mốc)
    geo_url = _build_url(data_root, snapshot, GEO_FILE[0])
    geo_dest = raw_dir / "neighbourhoods.geojson"
    if not geo_dest.exists():
        _download_one(geo_url, geo_dest)
    paths["neighbourhoods"] = geo_dest

    return paths


def download_all(cfg: dict) -> dict:
    """Tải dữ liệu cho TẤT CẢ các mốc chụp trong config.

    Returns:
        dict: mapping snapshot_date → {table_name → file_path}
    """
    data_root = cfg["data_root"]
    raw_dir = Path("data/raw")
    all_paths = {}

    logger.info("═══ BẮT ĐẦU TẢI DỮ LIỆU: %s ═══", cfg["city"].upper())
    for snapshot in cfg["snapshots"]:
        all_paths[snapshot] = download_snapshot(data_root, snapshot, raw_dir)

    logger.info("═══ HOÀN TẤT TẢI DỮ LIỆU ═══\n")
    return all_paths


# ── CLI test ──
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    with open("configs/sydney.yaml") as f:
        cfg = yaml.safe_load(f)
    download_all(cfg)
