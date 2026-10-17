
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


# Project root: thư mục cha của src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sydney.yaml"

# Các cột tối thiểu để sử dụng từng loại dữ liệu
REQUIRED_COLUMNS = {
    "listings.csv": ["id", "latitude", "longitude"],
    "calendar.csv": ["listing_id", "date", "available"],
    "reviews.csv": ["listing_id", "id", "date", "comments"],
    "neighbourhoods.csv": ["neighbourhood"],
}

CHUNK_SIZE = 100_000


class QAReport:
    """Lưu kết quả kiểm tra chất lượng dữ liệu."""

    def __init__(self):
        self.checks = []

    def add(self, level, snapshot, file, check, message):
        self.checks.append(
            {
                "level": level,
                "snapshot": snapshot,
                "file": file,
                "check": check,
                "message": message,
            }
        )

    def to_dict(self):
        counts = {
            level: sum(
                1 for item in self.checks
                if item["level"] == level
            )
            for level in ("ERROR", "WARNING", "INFO", "PASS")
        }

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary": counts,
            "checks": self.checks,
        }


def load_config(config_path):
    """Đọc file cấu hình YAML."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("File YAML không chứa cấu hình hợp lệ.")

    return config


def check_csv(csv_path, snapshot, report, min_length=5):
    """Kiểm tra CSV theo từng chunk để hạn chế RAM."""

    filename = csv_path.name
    required = REQUIRED_COLUMNS.get(filename, [])

    # Bước 1: Đọc header
    try:
        header = pd.read_csv(csv_path, nrows=0)
        columns = list(header.columns)
    except Exception as exc:
        report.add(
            "ERROR", snapshot, filename, "read_csv",
            f"Không đọc được header: {exc}",
        )
        return

    missing_columns = sorted(set(required) - set(columns))

    if missing_columns:
        report.add(
            "ERROR", snapshot, filename, "required_columns",
            f"Thiếu cột bắt buộc: {missing_columns}",
        )
        return

    report.add(
        "PASS", snapshot, filename, "required_columns",
        f"Đã tìm thấy các cột bắt buộc: {required}",
    )

    # Bước 2: Đọc từng phần và thống kê
    total_rows = 0
    null_counts = {column: 0 for column in required}
    invalid_dates = 0
    invalid_coordinates = 0
    invalid_prices = 0
    short_comments = 0
    missing_comment_rows = 0
    duplicate_rows_in_chunks = 0
    seen_ids = set()
    duplicate_ids = 0

    id_column = "id" if filename in {
        "listings.csv", "reviews.csv"
    } else None

    # Calendar có thể chứa hàng triệu dòng.
    # Không lưu toàn bộ khóa (listing_id, date) vào RAM.
    try:
        for chunk in pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            total_rows += len(chunk)

            # Kiểm tra giá trị trống ở các cột bắt buộc
            for column in required:
                null_counts[column] += int(
                    chunk[column].isna().sum()
                )

            # Kiểm tra ID trùng giữa các chunk
            
            # Kiểm tra ID trùng trong chunk hiện tại
            # và giữa các chunk đã đọc trước đó
            if id_column:
                ids = chunk[id_column].dropna()

                duplicate_ids += int(ids.isin(seen_ids).sum())
                duplicate_ids += int(ids.duplicated().sum())

                seen_ids.update(ids.tolist())


            # Kiểm tra ngày tháng
            if "date" in chunk.columns:
                parsed_dates = pd.to_datetime(
                    chunk["date"], errors="coerce"
                )
                invalid_dates += int(
                    (chunk["date"].notna() & parsed_dates.isna()).sum()
                )

            # Kiểm tra tọa độ
            if {"latitude", "longitude"}.issubset(chunk.columns):
                lat = pd.to_numeric(
                    chunk["latitude"], errors="coerce"
                )
                lon = pd.to_numeric(
                    chunk["longitude"], errors="coerce"
                )

                bad_coordinates = (
                    lat.notna()
                    & lon.notna()
                    & (
                        ~lat.between(-90, 90)
                        | ~lon.between(-180, 180)
                    )
                )
                invalid_coordinates += int(bad_coordinates.sum())

            # Kiểm tra giá: chỉ thống kê các giá trị không đọc được.
            # Giá trị trống sẽ được báo riêng ở phần missing.
            if "price" in chunk.columns:
                prices = chunk["price"].dropna().astype(str)
                cleaned = prices.str.replace(
                    r"[$,]", "", regex=True
                )
                numeric_prices = pd.to_numeric(
                    cleaned, errors="coerce"
                )
                invalid_prices += int(numeric_prices.isna().sum())

            # Kiểm tra nội dung review
            if "comments" in chunk.columns:
                comments = chunk["comments"]
                missing_comment_rows += int(comments.isna().sum())

                text_comments = comments.dropna().astype(str)
                short_comments += int(
                    (text_comments.str.strip().str.len() < min_length).sum()
                )

            # Chỉ phát hiện dòng trùng trong từng chunk.
            # Không khẳng định đây là số dòng trùng trên toàn file.
            duplicate_rows_in_chunks += int(
                chunk.duplicated().sum()
            )

    except Exception as exc:
        report.add(
            "ERROR", snapshot, filename, "read_csv_chunks",
            f"Lỗi khi đọc dữ liệu: {exc}",
        )
        return

    report.add(
        "INFO", snapshot, filename, "row_count",
        f"Tổng số dòng đọc được: {total_rows:,}",
    )

    # Bước 3: Báo cáo missing values
    for column, count in null_counts.items():
        if column == "comments":
            continue
        if count:
            percentage = count / total_rows * 100 if total_rows else 0
            level = "ERROR" if column in {
                "id", "listing_id"
            } else "WARNING"

            report.add(
                level, snapshot, filename, f"missing_{column}",
                f"{count:,}/{total_rows:,} giá trị trống "
                f"({percentage:.2f}%).",
            )

    # Kiểm tra ID trùng
    if id_column:
        if duplicate_ids:
            report.add(
                "ERROR", snapshot, filename, "duplicate_ids",
                f"Phát hiện {duplicate_ids:,} ID trùng "
                f"trong cột {id_column}.",
            )
        else:
            report.add(
                "PASS", snapshot, filename, "duplicate_ids",
                f"Không phát hiện ID trùng trong cột {id_column}.",
            )

    # Kiểm tra ngày tháng
    if "date" in columns:
        if invalid_dates:
            report.add(
                "WARNING", snapshot, filename, "invalid_dates",
                f"Có {invalid_dates:,} giá trị ngày không hợp lệ.",
            )
        else:
            report.add(
                "PASS", snapshot, filename, "invalid_dates",
                "Không phát hiện giá trị ngày không hợp lệ.",
            )

    # Kiểm tra tọa độ
    if "latitude" in columns and "longitude" in columns:
        if invalid_coordinates:
            report.add(
                "WARNING", snapshot, filename, "invalid_coordinates",
                f"Có {invalid_coordinates:,} dòng có tọa độ ngoài phạm vi.",
            )
        else:
            report.add(
                "PASS", snapshot, filename, "invalid_coordinates",
                "Không phát hiện tọa độ ngoài phạm vi.",
            )

    # Kiểm tra giá phòng
    
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    if "price" in columns:
        price_missing = 0
        invalid_prices = 0
        # Đọc riêng cột price theo chunks để tiết kiệm bộ nhớ.
        for price_chunk in pd.read_csv(
            csv_path,
            usecols=["price"],
            chunksize=CHUNK_SIZE,
        ):
            prices = price_chunk["price"]
            price_missing += int(prices.isna().sum())

            # Phân biệt giá bị thiếu với giá không chuyển được thành số.
            # Chuẩn hóa giá dạng chuỗi như "$24.70" thành số
            clean_prices = (
                prices.astype("string")
                .str.replace(r"[$,]", "", regex=True)
                .str.strip()
            )

            numeric_prices = pd.to_numeric(clean_prices, errors="coerce")
            invalid_prices += int(
                (prices.notna() & numeric_prices.isna()).sum()
            )

        if price_missing == total_rows and total_rows > 0:
            report.add(
                "WARNING", snapshot, filename, "price_all_missing",
                "Toàn bộ cột price bị thiếu; không thể tính KPI giá "
                "từ cột này.",
            )
        elif price_missing > 0:
            report.add(
                "WARNING", snapshot, filename, "price_missing",
                f"Cột price thiếu {price_missing:,} giá trị.",
            )

        if invalid_prices > 0:
            report.add(
                "WARNING", snapshot, filename, "invalid_prices",
                f"Có {invalid_prices:,} giá trị price không chuyển "
                "được thành số.",
            )

        if price_missing == 0 and invalid_prices == 0:
            report.add(
                "PASS", snapshot, filename, "price",
                "Cột price không có giá trị trống hoặc giá trị "
                "không chuyển được thành số.",
            )


    # Kiểm tra nội dung review
    
    # Kiểm tra nội dung review
    if "comments" in columns:
        if missing_comment_rows:
            report.add(
                "WARNING", snapshot, filename, "missing_comments",
                f"Có {missing_comment_rows:,}/{total_rows:,} "
                f"review không có comments "
                f"({missing_comment_rows / total_rows * 100:.2f}%).",
            )

        if short_comments:
            report.add(
                "INFO", snapshot, filename, "short_comments",
                f"Có {short_comments:,} comments ngắn hơn "
                f"{min_length} ký tự.",
            )


    if duplicate_rows_in_chunks:
        report.add(
            "WARNING", snapshot, filename, "duplicate_rows",
            f"Phát hiện {duplicate_rows_in_chunks:,} dòng trùng "
            "trong phạm vi từng chunk; chưa kiểm tra trùng dòng "
            "giữa các chunk.",
        )
    else:
        report.add(
            "INFO", snapshot, filename, "duplicate_rows",
            "Không phát hiện dòng trùng trong phạm vi từng chunk; "
            "chưa kiểm tra trùng dòng giữa các chunk.",
        )


def check_geojson(path, snapshot, report):
    """Kiểm tra cấu trúc GeoJSON cơ bản."""

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if data.get("type") != "FeatureCollection":
            report.add(
                "WARNING", snapshot, path.name, "geojson_type",
                "GeoJSON không có type = FeatureCollection.",
            )
            return

        features = data.get("features", [])
        report.add(
            "PASS", snapshot, path.name, "geojson_structure",
            f"GeoJSON hợp lệ về cấu trúc cơ bản; "
            f"có {len(features):,} features.",
        )

    except Exception as exc:
        report.add(
            "ERROR", snapshot, path.name, "geojson_read",
            f"Không đọc được GeoJSON: {exc}",
        )


def write_reports(report):
    """Xuất báo cáo QA ra JSON và Markdown."""

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = report.to_dict()

    json_path = output_dir / "qa_report.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Data Quality Report",
        "",
        f"- Generated at: {result['generated_at']}",
        "",
        "## Summary",
        "",
    ]

    for level, count in result["summary"].items():
        lines.append(f"- **{level}:** {count}")

    lines.extend(["", "## Details", ""])

    for item in result["checks"]:
        lines.append(
            f"- **[{item['level']}]** "
            f"`{item['snapshot']}/{item['file']}` — "
            f"{item['check']}: {item['message']}"
        )

    md_path = output_dir / "qa_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== QA SUMMARY ===")
    for level, count in result["summary"].items():
        print(f"{level}: {count}")

    print(f"\nĐã lưu báo cáo:\n- {md_path}\n- {json_path}")


def run_qa(config_path=DEFAULT_CONFIG):
    """Chạy QA cho tất cả snapshot được khai báo trong YAML."""

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)

    data_root = PROJECT_ROOT / config["data_root"]
    snapshots = config.get("snapshots", [])
    min_length = config.get("qa_params", {}).get("min_length", 5)

    report = QAReport()

    if not snapshots:
        report.add(
            "ERROR", "-", "-", "snapshots_config",
            "Không có snapshot nào trong cấu hình.",
        )

    for snapshot in snapshots:
        snapshot_dir = data_root / snapshot

        print(f"\nĐang kiểm tra snapshot: {snapshot}")

        if not snapshot_dir.exists():
            report.add(
                "ERROR", snapshot, "-", "snapshot_exists",
                f"Không tìm thấy thư mục: {snapshot_dir}",
            )
            continue

        report.add(
            "PASS", snapshot, "-", "snapshot_exists",
            "Thư mục snapshot tồn tại.",
        )

        required_files = [
            "listings.csv",
            "calendar.csv",
            "reviews.csv",
            "neighbourhoods.csv",
            "neighbourhoods.geojson",
        ]

        for filename in required_files:
            file_path = snapshot_dir / filename

            if not file_path.exists():
                report.add(
                    "ERROR", snapshot, filename, "file_exists",
                    "Thiếu file dữ liệu theo cấu trúc dự kiến.",
                )
                continue

            report.add(
                "PASS", snapshot, filename, "file_exists",
                "File tồn tại.",
            )

            if filename.endswith(".csv"):
                check_csv(
                    file_path,
                    snapshot,
                    report,
                    min_length=min_length,
                )
            elif filename.endswith(".geojson"):
                check_geojson(file_path, snapshot, report)

    write_reports(report)


if __name__ == "__main__":
    run_qa()
