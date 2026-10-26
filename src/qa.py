
from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent if FILE_DIR.name == "src" else FILE_DIR
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sydney.yaml"
CHUNK_SIZE = 100_000
LEVELS = ("ERROR", "WARNING", "INFO", "PASS")

REQUIRED_COLUMNS = {
    "listings.csv": ["id", "latitude", "longitude"],
    "calendar.csv": ["listing_id", "date", "available"],
    "reviews.csv": ["listing_id", "id", "date", "comments"],
    "neighbourhoods.csv": ["neighbourhood"],
}

OPTIONAL_INFO_COLUMNS = {
    "description",
    "neighborhood_overview",
    "host_since",
    "host_location",
    "host_about",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_name",
    "host_profile_url",
    "host_profile_id",
    "host_listings_count",
    "host_total_listings_count",
    "host_verifications",
    "host_has_profile_pic",
    "host_identity_verified",
    "host_is_superhost",
    "hosts_time_as_user_years",
    "hosts_time_as_user_months",
    "hosts_time_as_host_years",
    "hosts_time_as_host_months",
    "license",
    "calendar_updated",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "price_quote_total_price",
    "price_quote_price_per_night",
    "price_quote_raw",
    "adjusted_price",
    "reviewer_name",
}



def add(report, level, snapshot, filename, check, message):
    report.append({"level": level, "snapshot": str(snapshot), "file": str(filename),
                   "check": check, "message": str(message)})


def read_config(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Cấu hình YAML phải là một mapping.")
    return config


def numeric_price(series):
    """Convert values such as '$1,234.00' to numeric; invalid strings become NaN."""
    cleaned = (series.astype("string").str.replace(r"[$,]", "", regex=True)
               .str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}))
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_available(series):
    """Return nullable Boolean values and a mask of values not understood."""
    raw = series.astype("string").str.strip().str.lower()
    mapping = {"t": True, "true": True, "1": True, "yes": True,
               "f": False, "false": False, "0": False, "no": False}
    normalized = raw.map(mapping).astype("boolean")
    invalid = series.notna() & normalized.isna()
    return normalized, invalid


def check_csv(path, snapshot, report, min_length=5, clean=False, processed_root=None):
    name = path.name
    required = REQUIRED_COLUMNS.get(name, [])
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        add(report, "ERROR", snapshot, name, "read_header", f"Không đọc được header: {exc}")
        return

    missing_cols = sorted(set(required) - set(columns))
    if missing_cols:
        add(report, "ERROR", snapshot, name, "required_columns", f"Thiếu cột bắt buộc: {missing_cols}")
        return
    add(report, "PASS", snapshot, name, "required_columns", f"Đủ cột bắt buộc: {required}")

    total = 0
    missing = {c: 0 for c in columns}
    duplicate_ids = 0
    seen_ids = set()
    invalid_dates = invalid_coords = missing_coords = 0
    price_missing = price_unparseable = price_nonpositive = 0
    price_values = []
    available_invalid = available_missing = 0
    calendar_dup_keys = 0
    missing_comments = short_comments = duplicate_rows = 0
    review_negative_count = 0
    review_score_invalid = 0
    output_path = None
    if clean and processed_root is not None:
        output_path = processed_root / str(snapshot) / name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

    # SQLite keeps calendar keys on disk instead of retaining millions of keys in RAM.
    db = None
    db_path = None
    if name == "calendar.csv":
        tmp = tempfile.NamedTemporaryFile(prefix="airbnb_qa_keys_", suffix=".sqlite", delete=False)
        db_path = Path(tmp.name)
        tmp.close()
        db = sqlite3.connect(db_path)
        db.execute("CREATE TABLE keys (listing_id TEXT NOT NULL, date TEXT NOT NULL, PRIMARY KEY (listing_id, date))")

    try:
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            total += len(chunk)
            for col in columns:
                missing[col] += int(chunk[col].isna().sum())
            duplicate_rows += int(chunk.duplicated().sum())

            if name in {"listings.csv", "reviews.csv"} and "id" in chunk:
                ids = chunk["id"].dropna().astype(str)
                duplicate_ids += int(ids.duplicated().sum())
                duplicate_ids += int(ids.isin(seen_ids).sum())
                seen_ids.update(ids.tolist())

            if "date" in chunk:
                dates = pd.to_datetime(chunk["date"], errors="coerce")
                invalid_dates += int((chunk["date"].notna() & dates.isna()).sum())
                if name == "calendar.csv" and "listing_id" in chunk:
                    valid = chunk["listing_id"].notna() & dates.notna()
                    keys = zip(chunk.loc[valid, "listing_id"].astype(str), dates.loc[valid].dt.strftime("%Y-%m-%d"))
                    batch = []
                    for listing_id, date in keys:
                        batch.append((listing_id, date))
                    before = db.total_changes
                    db.executemany("INSERT OR IGNORE INTO keys VALUES (?, ?)", batch)
                    calendar_dup_keys += len(batch) - (db.total_changes - before)
                    db.commit()

            if {"latitude", "longitude"}.issubset(chunk.columns):
                lat = pd.to_numeric(chunk["latitude"], errors="coerce")
                lon = pd.to_numeric(chunk["longitude"], errors="coerce")
                missing_coords += int((lat.isna() | lon.isna()).sum())
                invalid_coords += int((lat.notna() & lon.notna() & (~lat.between(-90, 90) | ~lon.between(-180, 180))).sum())

            if "price" in chunk:
                prices = numeric_price(chunk["price"])
                price_missing += int(chunk["price"].isna().sum())
                price_unparseable += int((chunk["price"].notna() & prices.isna()).sum())
                price_nonpositive += int((prices.notna() & (prices <= 0)).sum())
                # Keep a bounded sample for outlier diagnostics, not for deleting records.
                valid_prices = prices[prices > 0].dropna()
                if len(price_values) < 200_000:
                    price_values.extend(valid_prices.head(200_000 - len(price_values)).tolist())

            if "available" in chunk:
                normalized, invalid = normalize_available(chunk["available"])
                available_invalid += int(invalid.sum())
                available_missing += int(chunk["available"].isna().sum())

            if "comments" in chunk:
                comments = chunk["comments"]
                missing_comments += int((comments.isna() | comments.astype("string").str.strip().eq("").fillna(False)).sum())
                text = comments.dropna().astype(str).str.strip()
                short_comments += int((text.str.len() < min_length).sum())

            if "number_of_reviews" in chunk:
                n = pd.to_numeric(chunk["number_of_reviews"], errors="coerce")
                review_negative_count += int((n < 0).sum())
            if "review_scores_rating" in chunk:
                scores = pd.to_numeric(chunk["review_scores_rating"], errors="coerce")
                review_score_invalid += int((scores.notna() & ~scores.between(0, 100)).sum())

            if clean:
                cleaned = chunk.copy()
                if name == "listings.csv":
                    if "price" in cleaned:
                        p = numeric_price(cleaned["price"])
                        p = p.mask(p <= 0)
                        cleaned["price"] = p
                    if {"latitude", "longitude"}.issubset(cleaned.columns):
                        lat = pd.to_numeric(cleaned["latitude"], errors="coerce")
                        lon = pd.to_numeric(cleaned["longitude"], errors="coerce")
                        bad = lat.isna() | lon.isna() | ~lat.between(-90, 90) | ~lon.between(-180, 180)
                        cleaned["latitude"] = lat.mask(bad)
                        cleaned["longitude"] = lon.mask(bad)
                    if "number_of_reviews" in cleaned:
                        n = pd.to_numeric(cleaned["number_of_reviews"], errors="coerce")
                        cleaned["number_of_reviews"] = n.mask(n < 0)
                    if "review_scores_rating" in cleaned:
                        scores = pd.to_numeric(cleaned["review_scores_rating"], errors="coerce")
                        cleaned["review_scores_rating"] = scores.mask(~scores.between(0, 100))
                elif name == "calendar.csv":
                    if "date" in cleaned:
                        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    if "available" in cleaned:
                        normalized, _ = normalize_available(cleaned["available"])
                        cleaned["available"] = normalized
                    if "price" in cleaned:
                        p = numeric_price(cleaned["price"])
                        cleaned["price"] = p.mask(p <= 0)
                elif name == "reviews.csv":
                    if "date" in cleaned:
                        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    if "comments" in cleaned:
                        cleaned["comments"] = cleaned["comments"].astype("string").str.strip().replace("", pd.NA)
                cleaned.to_csv(output_path, mode="a", index=False, header=not output_path.exists(), encoding="utf-8")

    except Exception as exc:
        add(report, "ERROR", snapshot, name, "read_or_process", f"Lỗi khi đọc/xử lý CSV: {exc}")
        return
    finally:
        if db is not None:
            db.close()
        if db_path is not None:
            db_path.unlink(missing_ok=True)

    add(report, "INFO", snapshot, name, "row_count", f"Đọc được {total:,} dòng.")
    
    for col, count in missing.items():
        if count == 0:
            continue

        # Các trường đã có kiểm tra chuyên biệt bên dưới.
        if col in {
            "price",
            "comments",
            "available",
            "latitude",
            "longitude",
        }:
            continue

        pct = count / total * 100 if total else 0

        if col in {"id", "listing_id"}:
            level = "ERROR"
        elif col in OPTIONAL_INFO_COLUMNS:
            level = "INFO"
        else:
            level = "WARNING"

        add(
            report,
            level,
            snapshot,
            name,
            f"missing_{col}",
            f"Thiếu {count:,}/{total:,} ({pct:.2f}%)."
        )


    if name in {"listings.csv", "reviews.csv"}:
        add(report, "ERROR" if duplicate_ids else "PASS", snapshot, name, "duplicate_ids",
            f"Phát hiện {duplicate_ids:,} ID trùng." if duplicate_ids else "Không phát hiện ID trùng.")
    if "date" in columns:
        add(report, "WARNING" if invalid_dates else "PASS", snapshot, name, "invalid_dates",
            f"Có {invalid_dates:,} ngày không hợp lệ." if invalid_dates else "Không phát hiện ngày không hợp lệ.")
    if {"latitude", "longitude"}.issubset(columns):
        add(report, "WARNING" if missing_coords or invalid_coords else "PASS", snapshot, name, "coordinates",
            f"Tọa độ thiếu/không đọc được: {missing_coords:,}; ngoài phạm vi: {invalid_coords:,}.")
    if "price" in columns:
            
     if name == "listings.csv" and "price" in columns:
        if price_missing == total and total > 0:
            add(
                report,
                "ERROR",
                snapshot,
                name,
                "price_all_missing",
                "Toàn bộ price bị thiếu; KPI giá không thể tính từ cột này."
            )

        if price_missing:
            add(
                report, "WARNING", snapshot, name, "price_missing",
                f"Thiếu {price_missing:,} giá."
            )

        if price_unparseable:
            add(
                report, "WARNING", snapshot, name, "price_unparseable",
                f"Có {price_unparseable:,} giá không chuyển được thành số."
            )

        if price_nonpositive:
            add(
                report, "WARNING", snapshot, name, "price_nonpositive",
                f"Có {price_nonpositive:,} giá bằng 0 hoặc âm; cleaning chuyển thành missing."
            )

        if price_values:
            q1, q3 = pd.Series(price_values).quantile([0.25, 0.75])
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            outliers = sum(x > upper for x in price_values)

            add(
                report, "INFO", snapshot, name, "price_iqr_outliers_sample",
                f"Mẫu tối đa {len(price_values):,} giá dương: "
                f"ngưỡng IQR trên={upper:.2f}; {outliers:,} giá trong mẫu "
                "vượt ngưỡng. Không tự động xóa ngoại lai."
            )

    if "available" in columns:
        add(report, "WARNING" if available_invalid else "PASS", snapshot, name, "available_domain",
            f"Có {available_invalid:,} giá trị available không nhận diện được." if available_invalid else "Các giá trị available đều thuộc miền Boolean nhận diện được.")
        if available_missing:
            add(report, "WARNING", snapshot, name, "available_missing", f"Thiếu {available_missing:,} giá trị available.")
    if name == "calendar.csv":
        add(report, "WARNING" if calendar_dup_keys else "PASS", snapshot, name, "duplicate_listing_date",
            f"Phát hiện {calendar_dup_keys:,} khóa (listing_id, date) trùng." if calendar_dup_keys else "Không phát hiện khóa (listing_id, date) trùng.")
    if "comments" in columns:
        if missing_comments:
            add(report, "WARNING", snapshot, name, "missing_comments", f"Có {missing_comments:,} review rỗng/thiếu nội dung.")
        add(report, "INFO", snapshot, name, "short_comments", f"Có {short_comments:,} comments ngắn hơn {min_length} ký tự.")
    if review_negative_count:
        add(report, "WARNING", snapshot, name, "negative_number_of_reviews", f"Có {review_negative_count:,} number_of_reviews âm.")
    if review_score_invalid:
        add(report, "WARNING", snapshot, name, "review_scores_rating_range", f"Có {review_score_invalid:,} review_scores_rating ngoài 0–100.")
    add(report, "WARNING" if duplicate_rows else "PASS", snapshot, name, "duplicate_rows_within_chunk",
        f"Có {duplicate_rows:,} dòng trùng hoàn toàn trong cùng chunk; chưa kiểm tra trùng dòng hoàn chỉnh giữa các chunk." if duplicate_rows else "Không phát hiện dòng trùng hoàn toàn trong cùng chunk; kiểm tra trùng hoàn chỉnh xuyên chunk chưa được thực hiện.")

    if clean and output_path is not None:
        add(report, "INFO", snapshot, name, "cleaned_output", f"Đã ghi bản làm sạch vào {output_path.relative_to(PROJECT_ROOT)}; dữ liệu gốc được giữ nguyên.")


def check_geojson(path, snapshot, report):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
            add(report, "ERROR", snapshot, path.name, "geojson_type", "GeoJSON phải có type=FeatureCollection.")
            return
        features = data.get("features")
        if not isinstance(features, list):
            add(report, "ERROR", snapshot, path.name, "geojson_features", "Trường features không phải danh sách.")
            return
        invalid = 0
        missing_geometry = 0
        for feature in features:
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                invalid += 1
                continue
            geom = feature.get("geometry")
            if not isinstance(geom, dict) or not geom.get("type") or geom.get("coordinates") is None:
                missing_geometry += 1
        add(report, "INFO", snapshot, path.name, "geojson_feature_count", f"Có {len(features):,} features.")
        add(report, "WARNING" if invalid else "PASS", snapshot, path.name, "geojson_features", f"Feature không hợp lệ: {invalid:,}.")
        add(report, "WARNING" if missing_geometry else "PASS", snapshot, path.name, "geojson_geometry", f"Feature thiếu geometry/tọa độ: {missing_geometry:,}.")
    except Exception as exc:
        add(report, "ERROR", snapshot, path.name, "geojson_read", f"Không đọc được GeoJSON: {exc}")


def write_reports(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {level: sum(x["level"] == level for x in report) for level in LEVELS}
    result = {"generated_at": datetime.now().isoformat(timespec="seconds"), "summary": summary, "checks": report}
    (output_dir / "qa_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Data Quality Report", "", f"- Generated at: {result['generated_at']}", "", "## Summary", ""]
    lines.extend(f"- **{level}:** {count}" for level, count in summary.items())
    lines.extend(["", "## Details", ""])
    lines.extend(f"- **[{x['level']}]** `{x['snapshot']}/{x['file']}` — {x['check']}: {x['message']}" for x in report)
    (output_dir / "qa_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n=== QA SUMMARY ===")
    for level, count in summary.items():
        print(f"{level}: {count}")
    print(f"\nReports saved to: {output_dir}")


def run_qa(config_path=DEFAULT_CONFIG, clean=True):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = read_config(config_path)
    data_root = Path(config.get("data_root", "data/raw"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    processed_root = PROJECT_ROOT / "data" / "processed"
    min_length = int(config.get("qa_params", {}).get("min_length", 5))
    snapshots = config.get("snapshots", [])
    report = []
    if not snapshots:
        add(report, "ERROR", "-", "-", "snapshots_config", "Không có snapshot nào trong YAML.")

    for snapshot in snapshots:
        snapshot_dir = data_root / str(snapshot)
        print(f"\nĐang kiểm tra snapshot: {snapshot}")
        if not snapshot_dir.is_dir():
            add(report, "ERROR", snapshot, "-", "snapshot_exists", f"Không tìm thấy thư mục {snapshot_dir}.")
            continue
        add(report, "PASS", snapshot, "-", "snapshot_exists", "Thư mục snapshot tồn tại.")
        # neighbourhood files are optional for non-spatial KPIs; report missing files as WARNING.
        required_files = ["listings.csv", "calendar.csv", "reviews.csv"]
        optional_files = ["neighbourhoods.csv", "neighbourhoods.geojson"]
        for filename in required_files + optional_files:
            path = snapshot_dir / filename
            if not path.is_file():
                level = "ERROR" if filename in required_files else "WARNING"
                add(report, level, snapshot, filename, "file_exists", "Thiếu file; kiểm tra tên file/đường dẫn và tải lại nếu cần.")
                continue
            add(report, "PASS", snapshot, filename, "file_exists", "File tồn tại.")
            if filename.endswith(".csv"):
                check_csv(path, snapshot, report, min_length=min_length, clean=clean and filename in {"listings.csv", "calendar.csv", "reviews.csv"}, processed_root=processed_root)
            elif filename.endswith(".geojson"):
                check_geojson(path, snapshot, report)

    write_reports(report, PROJECT_ROOT / "reports")
    if clean:
        print(f"Cleaned CSV files (where source files exist) are saved under: {processed_root}")
    return report


if __name__ == "__main__":
    run_qa()