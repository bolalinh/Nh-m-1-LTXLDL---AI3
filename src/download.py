"""
Module: src/download.py
Chức năng: Tải tự động 4 mốc dữ liệu Inside Airbnb và kiểm tra tính toàn vẹn.
Cách chạy độc lập: python src/download.py --config configs/sydney.yaml
"""

import os
import gzip
import argparse
import yaml
import requests


def download_file(url: str, save_path: str) -> bool:
    """
    Tải 1 file từ URL và lưu vào save_path.
    Nếu file đã có trên máy và > 0 bytes -> Bỏ qua không tải lại (Caching).
    """
    # 1. Kiểm tra nếu đã tải rồi thì thôi
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"   [ĐÃ CÓ] {os.path.basename(save_path)} -> Bỏ qua tải lại.")
        return True

    print(f"   [ĐANG TẢI] {url} ...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. Gửi request giả lập trình duyệt để không bị server Inside Airbnb chặn
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as response:
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    # Ghi từng khối 1MB vào đĩa để tránh tràn RAM
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                print(f"   [XONG] Đã lưu vào {save_path}")
                return True
            else:
                print(f"   [THẤT BẠI] Mã lỗi HTTP: {response.status_code}")
                return False
    except Exception as e:
        print(f"   [LỖI MẠNG] {e}")
        # Xóa file rác nếu tải dở dang
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def check_integrity(file_path: str, key_column: str = None) -> bool:
    """
    Kiểm tra tính toàn vẹn theo yêu cầu đề bài:
    1. File có tồn tại và kích thước > 0 không?
    2. Nếu là file .csv.gz, mở ra xem có chứa cột khóa (key column) không?
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"   [LỖI TOÀN VẸN] File rỗng hoặc không tồn tại: {file_path}")
        return False

    # Nếu bảng cần kiểm tra cột khóa và là định dạng nén gzip
    if key_column and file_path.endswith(".csv.gz"):
        try:
            with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                header = f.readline() # Đọc dòng đầu tiên chứa tên các cột
                columns = [c.strip().strip('"') for c in header.split(",")]
                if key_column not in columns:
                    print(f"   [LỖI TOÀN VẸN] Thiếu cột khóa chính '{key_column}' trong {file_path}")
                    return False
        except Exception as e:
            print(f"   [LỖI TOÀN VẸN] Không mở được file nén: {e}")
            return False

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"   [HỢP LỆ] Đã kiểm tra tính toàn vẹn ({file_size_mb:.2f} MB)")
    return True


def run_download(config_path: str):
    """Đọc file config và tải đủ các bảng cho 4 mốc snapshots."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    city = config.get("city_name", config.get("city", "Sydney"))
    data_root = config["data_root"]       # 'australia/nsw/sydney'
    snapshots = config["snapshots"]       # 4 mốc ngày

    print(f"==================================================")
    print(f" BẮT ĐẦU TẢI DỮ LIỆU: {city.upper()}")
    print(f" Data root: {data_root}")
    print(f" Số mốc snapshots: {len(snapshots)}")
    print(f"==================================================")

    # 4 tệp dữ liệu cần thiết theo đề bài (Đường dẫn tương đối trên web, Cột khóa cần check)
    target_tables = [
        ("data/listings.csv.gz", "id"),
        ("data/calendar.csv.gz", "listing_id"),
        ("data/reviews.csv.gz", "listing_id"),
        ("visualisations/neighbourhoods.geojson", None)
    ]

    base_url = "https://data.insideairbnb.com"

    for date in snapshots:
        print(f"\n>>> Đang xử lý mốc snapshot: {date}")
        for rel_path, key_col in target_tables:
            filename = os.path.basename(rel_path)
            download_url = f"{base_url}/{data_root}/{date}/{rel_path}"
            
            # Lưu vào: data/raw/<mốc_ngày>/<tên_file>
            save_path = os.path.join("data", "raw", date, filename)

            # 1. Tải
            success = download_file(download_url, save_path)
            
            # 2. Kiểm tra tính toàn vẹn
            if success:
                check_integrity(save_path, key_column=key_col)

    print(f"\n==================================================")
    print(f" HOÀN TẤT BƯỚC THU THẬP DỮ LIỆU!")
    print(f" Dữ liệu đã sẵn sàng trong thư mục: data/raw/")
    print(f"==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải dữ liệu Inside Airbnb tự động.")
    parser.add_argument("--config", type=str, default="configs/sydney.yaml", help="Đường dẫn file yaml")
    args = parser.parse_args()

    run_download(args.config)