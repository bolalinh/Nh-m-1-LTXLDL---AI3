"""
src/qa.py – Làm sạch dữ liệu (Quality Assurance)
===================================================

WHY (tại sao thiết kế thế này):
───────────────────────────────
1. MỖI QUY TẮC = 1 hàm riêng biệt, trả về:
   - DataFrame đã xử lý
   - dict thống kê (tên quy tắc, số dòng bị ảnh hưởng, hành động)
   → Dễ test, dễ bật/tắt từng quy tắc, dễ giải thích khi vấn đáp.

2. ƯU TIÊN GẮN CỜ thay vì xoá:
   - Thêm cột `_qa_flag` thay vì xoá dòng ngay
   - Giảng viên có thể kiểm tra lại quyết định của nhóm
   - Báo cáo QA ghi rõ: bao nhiêu dòng bị gắn cờ, vì lý do gì
   - Chỉ xoá khi thực sự cần (trùng lặp, thiếu khoá chính)

3. GHI qa_report.csv:
   - Mỗi dòng = 1 quy tắc QA đã áp dụng
   - Ghi số dòng trước/sau, KPI trước/sau → thấy tác động rõ ràng

Kỹ thuật từ bài giảng:
─────────────────────
- Bài 10 (Làm sạch dữ liệu): missing values, outliers, duplicates,
  dtype coercion, categorical validation
- Bài 5 (DataFrame chuyên sâu): .pipe(), method chaining, boolean indexing
- Bài 7 (Xử lý chuỗi): str accessor, regex cho parse price
- Bài 3 (NumPy): vectorized ops thay vì for loop
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# PARSE & CHUẨN HOÁ
# ══════════════════════════════════════════════════════════════

def parse_price(series: pd.Series) -> pd.Series:
    """Chuyển cột price từ chuỗi '$1,234.56' → float 1234.56.

    Tại sao dùng str.replace + astype thay vì apply(lambda)?
    → Vectorized string ops nhanh hơn gấp 10-100x so với apply
      trên DataFrame lớn (Bài 3: tư duy vector hoá, Bài 7: str accessor).
    """
    return (
        series
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu dữ liệu đúng cho các cột quan trọng.

    Tại sao ép kiểu ngay từ đầu?
    → Nhiều cột đọc từ CSV bị nhận sai kiểu (ví dụ: price là str,
      host_response_rate là str có %).
      Nếu không ép kiểu, các phép tính KPI sẽ lỗi hoặc cho kết quả sai.
      (Bài 10: kiểu sai là một trong 5 loại lỗi dữ liệu cần xử lý)
    """
    # Price: str → float
    if "price" in df.columns:
        df["price"] = parse_price(df["price"])

    # Host response rate: '95%' → 0.95
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = (
            df["host_response_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace(["N/A", "nan", ""], np.nan)
            .astype(float) / 100
        )

    # Host acceptance rate
    if "host_acceptance_rate" in df.columns:
        df["host_acceptance_rate"] = (
            df["host_acceptance_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace(["N/A", "nan", ""], np.nan)
            .astype(float) / 100
        )

    # Boolean columns: 't'/'f' → True/False
    bool_cols = ["host_is_superhost", "instant_bookable",
                 "has_availability"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"t": True, "f": False})

    # Numeric columns
    num_cols = [
        "latitude", "longitude", "minimum_nights", "maximum_nights",
        "number_of_reviews", "availability_365", "reviews_per_month",
        "review_scores_rating", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_checkin",
        "review_scores_communication", "review_scores_location",
        "review_scores_value", "calculated_host_listings_count",
        "bedrooms", "beds", "accommodates",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ══════════════════════════════════════════════════════════════
# CÁC QUY TẮC QA – mỗi hàm = 1 quy tắc
# ══════════════════════════════════════════════════════════════

def rule_missing_key_columns(df: pd.DataFrame, cfg: dict) -> tuple:
    """QA-01: Xoá dòng thiếu cột khoá định danh bắt buộc (id, host_id).

    Tại sao chỉ bắt buộc id và host_id thay vì xoá luôn price?
    → Nếu thiếu id hoặc host_id thì dòng đó vô giá trị, không thể định danh.
      Còn nếu thiếu price do lỗi scraping của bản chụp (như mốc 2025),
      chúng ta đã có bước liên kết giá chéo mốc (Cross-snapshot Linkage)
      hoặc nhóm tự quyết định, tránh xoá sạch toàn bộ mốc dữ liệu.
    """
    key_cols = ["id", "host_id"]
    existing_keys = [c for c in key_cols if c in df.columns]
    before = len(df)
    mask = df[existing_keys].isna().any(axis=1)
    n_affected = mask.sum()
    df = df[~mask].copy()
    return df, {
        "rule": "QA-01: Thiếu cột khoá định danh",
        "condition": f"NaN trong {existing_keys}",
        "reason": "Không thể định danh listing nếu thiếu id/host_id",
        "action": "XOÁ",
        "rows_before": before,
        "rows_after": len(df),
        "rows_affected": int(n_affected),
    }


def rule_price_outliers(df: pd.DataFrame, cfg: dict) -> tuple:
    """QA-02: Gắn cờ giá ngoại lai.

    Tại sao dùng ngưỡng cứng từ config thay vì IQR?
    → Phân phối giá Airbnb rất lệch phải (right-skewed).
      IQR cắt quá nhiều giá cao hợp lệ (penthouse, villa).
      Ngưỡng cứng do nhóm đặt dựa trên khảo sát thực tế:
      - price < 10 AUD/đêm → gần chắc là lỗi nhập liệu
      - price > 10,000 AUD/đêm → cực kỳ hiếm, có thể là lỗi
      Giảng viên kiểm tra được bằng cách xem qa_report.csv.
    """
    qa_cfg = cfg.get("qa", {})
    p_min = qa_cfg.get("price_min", 10)
    p_max = qa_cfg.get("price_max", 10000)

    mask = (df["price"] < p_min) | (df["price"] > p_max)
    n_affected = mask.sum()
    df.loc[mask, "_qa_flag"] = df.loc[mask, "_qa_flag"].astype(str) + "|price_outlier"
    return df, {
        "rule": "QA-02: Giá ngoại lai",
        "condition": f"price < {p_min} hoặc > {p_max} {cfg.get('currency', 'AUD')}",
        "reason": "Giá quá thấp = lỗi nhập; quá cao = ngoại lai",
        "action": "GẮN CỜ",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_affected": int(n_affected),
    }


def rule_minimum_nights_cap(df: pd.DataFrame, cfg: dict) -> tuple:
    """QA-03: Gắn cờ minimum_nights bất thường.

    Tại sao ngưỡng 365 ngày?
    → minimum_nights > 365 nghĩa là chủ nhà đặt tối thiểu > 1 năm.
      Thực tế hiếm ai thuê Airbnb > 1 năm. Những listing này thường
      là chủ nhà muốn "ẩn" listing mà không xoá, hoặc nhập nhầm.
    """
    qa_cfg = cfg.get("qa", {})
    cap = qa_cfg.get("min_nights_cap", 365)

    if "minimum_nights" not in df.columns:
        return df, None

    mask = df["minimum_nights"] > cap
    n_affected = mask.sum()
    df.loc[mask, "_qa_flag"] = (
        df.loc[mask, "_qa_flag"].astype(str) + "|high_min_nights"
    )
    return df, {
        "rule": "QA-03: minimum_nights bất thường",
        "condition": f"minimum_nights > {cap}",
        "reason": "Listing có thể bị ẩn hoặc nhập sai",
        "action": "GẮN CỜ",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_affected": int(n_affected),
    }


def rule_duplicates_within(df: pd.DataFrame, cfg: dict) -> tuple:
    """QA-04: Xoá trùng lặp TRONG cùng một mốc chụp.

    Tại sao dùng (id, last_scraped) làm khoá trùng?
    → Cùng id + cùng last_scraped = cùng 1 bản ghi bị trùng.
      Giữ bản đầu tiên (keep='first') vì thứ tự không quan trọng
      khi các bản hoàn toàn giống nhau.
    """
    qa_cfg = cfg.get("qa", {})
    keys = qa_cfg.get("dedupe_keys", ["id", "last_scraped"])
    existing_keys = [c for c in keys if c in df.columns]

    if not existing_keys:
        return df, None

    before = len(df)
    mask = df.duplicated(subset=existing_keys, keep="first")
    n_affected = mask.sum()
    df = df[~mask].copy()
    return df, {
        "rule": "QA-04: Trùng lặp trong mốc",
        "condition": f"duplicated({existing_keys})",
        "reason": "Cùng bản ghi xuất hiện nhiều lần → giữ 1",
        "action": "XOÁ",
        "rows_before": before,
        "rows_after": len(df),
        "rows_affected": int(n_affected),
    }


def rule_duplicates_across(listings_all: dict, cfg: dict) -> tuple:
    """QA-05: Phát hiện trùng ID giữa các mốc chụp.

    Tại sao KHÔNG xoá mà chỉ gắn cờ?
    → Cùng listing xuất hiện ở nhiều mốc là BÌNH THƯỜNG (listing tồn tại
      qua nhiều quý). Ta cần giữ lại để so sánh biến động theo thời gian.
      Chỉ gắn cờ để biết listing nào là "cũ" vs "mới xuất hiện".
    """
    all_ids = {}
    for snapshot, df in listings_all.items():
        for lid in df["id"].unique():
            all_ids.setdefault(lid, []).append(snapshot)

    multi_snapshot = {k: v for k, v in all_ids.items() if len(v) > 1}
    return None, {
        "rule": "QA-05: Trùng ID giữa các mốc",
        "condition": "Cùng id xuất hiện ở ≥2 mốc",
        "reason": "Bình thường – listing tồn tại qua nhiều quý",
        "action": "GHI NHẬN (không xoá)",
        "rows_before": sum(len(d) for d in listings_all.values()),
        "rows_after": sum(len(d) for d in listings_all.values()),
        "rows_affected": len(multi_snapshot),
    }


def rule_category_validation(df: pd.DataFrame, cfg: dict) -> tuple:
    """QA-06: Kiểm tra giá trị danh mục hợp lệ.

    Tại sao kiểm tra room_type?
    → Airbnb chỉ có 4 loại phòng chính thức. Nếu xuất hiện giá trị lạ
      → dữ liệu bị lỗi hoặc schema đã thay đổi → cần điều tra.
    """
    valid_room_types = {
        "Entire home/apt", "Private room", "Shared room", "Hotel room"
    }
    if "room_type" not in df.columns:
        return df, None

    mask = ~df["room_type"].isin(valid_room_types)
    n_affected = mask.sum()
    if n_affected > 0:
        invalid_values = df.loc[mask, "room_type"].unique().tolist()
        df.loc[mask, "_qa_flag"] = (
            df.loc[mask, "_qa_flag"].astype(str) + "|invalid_room_type"
        )
        logger.warning("  room_type lạ: %s", invalid_values)

    return df, {
        "rule": "QA-06: Danh mục room_type lệch",
        "condition": f"room_type ∉ {valid_room_types}",
        "reason": "Giá trị không nằm trong 4 loại chính thức",
        "action": "GẮN CỜ",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_affected": int(n_affected),
    }


def rule_missing_stats(df: pd.DataFrame, cfg: dict) -> tuple:
    """QA-07: Thống kê tỷ lệ thiếu cho từng cột quan trọng.

    Tại sao chỉ thống kê mà không xoá/điền?
    → Mỗi cột có chiến lược xử lý khác nhau:
      - review_scores_rating thiếu → listing mới chưa có review → bỏ qua
      - bedrooms thiếu → có thể điền bằng median theo room_type
      Ghi vào report để nhóm quyết định từng trường hợp.
    """
    important = [
        "price", "review_scores_rating", "bedrooms", "beds",
        "host_response_rate", "neighbourhood_cleansed",
        "reviews_per_month",
    ]
    existing = [c for c in important if c in df.columns]
    missing_stats = df[existing].isna().sum()
    missing_pct = (missing_stats / len(df) * 100).round(2)

    details = "; ".join(
        f"{c}: {missing_pct[c]}%" for c in existing if missing_pct[c] > 0
    )
    return df, {
        "rule": "QA-07: Thống kê tỷ lệ thiếu",
        "condition": "isna() trên các cột quan trọng",
        "reason": "Ghi nhận để quyết định chiến lược điền giá trị",
        "action": f"GHI NHẬN: {details}",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_affected": int(missing_stats.sum()),
    }


def rule_impute_missing(df: pd.DataFrame, cfg: dict) -> tuple:
    """QA-08: Điền giá trị thiếu cho cột số.

    Tại sao dùng median theo room_type thay vì mean toàn bộ?
    → Mean bị ảnh hưởng bởi outlier (Bài 10).
      Median theo room_type chính xác hơn vì giá/số phòng khác nhau
      rất lớn giữa Entire home vs Shared room.
    """
    impute_cols = ["bedrooms", "beds", "review_scores_rating",
                   "reviews_per_month"]
    existing = [c for c in impute_cols if c in df.columns]
    total_filled = 0

    if "room_type" in df.columns:
        for col in existing:
            before_na = df[col].isna().sum()
            # Điền bằng median THEO room_type (grouped imputation)
            df[col] = df.groupby("room_type")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Nếu vẫn còn NaN (nhóm toàn NaN), điền median toàn bộ
            df[col] = df[col].fillna(df[col].median())
            total_filled += before_na - df[col].isna().sum()
    else:
        for col in existing:
            before_na = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].median())
            total_filled += before_na - df[col].isna().sum()

    return df, {
        "rule": "QA-08: Điền giá trị thiếu (median theo room_type)",
        "condition": "NaN trong cột số quan trọng",
        "reason": "Median theo nhóm chính xác hơn mean toàn bộ",
        "action": "ĐIỀN GIÁ TRỊ",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_affected": int(total_filled),
    }


# ══════════════════════════════════════════════════════════════
# PIPELINE QA CHÍNH
# ══════════════════════════════════════════════════════════════

def clean_listings(df: pd.DataFrame, cfg: dict) -> tuple:
    """Chạy toàn bộ pipeline QA trên một DataFrame listings.

    Tại sao dùng danh sách hàm thay vì if-else?
    → Design pattern "pipeline of transformations" (Bài 5: .pipe()).
      Dễ thêm/bớt quy tắc, dễ test từng bước riêng lẻ.
    """
    # Khởi tạo cột flag
    df["_qa_flag"] = ""

    # Bước 0: Ép kiểu
    df = coerce_dtypes(df)

    # Danh sách quy tắc (THỨ TỰ QUAN TRỌNG):
    # 1. Xoá thiếu khoá trước
    # 2. Xoá trùng
    # 3. Gắn cờ ngoại lai
    # 4. Kiểm tra danh mục
    # 5. Thống kê thiếu
    # 6. Điền giá trị
    rules = [
        rule_missing_key_columns,
        rule_duplicates_within,
        rule_price_outliers,
        rule_minimum_nights_cap,
        rule_category_validation,
        rule_missing_stats,
        rule_impute_missing,
    ]

    reports = []
    for rule_fn in rules:
        df, report = rule_fn(df, cfg)
        if report:
            reports.append(report)
            logger.info(
                "  %s: %d dòng bị ảnh hưởng (%s)",
                report["rule"], report["rows_affected"], report["action"]
            )

    return df, reports


def apply_latest_snapshot_price_policy(all_dfs: dict, cfg: dict) -> list:
    """QA-00: Áp dụng chính sách neo giá chỉ dùng giá của mốc cuối cùng (Latest Snapshot Price Policy).
    
    YÊU CẦU ĐẶC BIỆT THEO CHỈ ĐẠO:
    - Toàn bộ giá của các mốc trước đó (2025-09-12, 2025-12-12, 2026-03-21) dù có sẵn giá trong file gốc
      hay không, đều BỊ THAY THẾ HOÀN TOÀN bằng giá niêm yết của mốc chụp cuối cùng (2026-06-16).
    - Quy trình thực hiện triệt để:
      1. Chuẩn hoá và làm sạch bảng giá của mốc cuối cùng (2026-06-16).
      2. Với mọi mốc trước đó, xoá bỏ giá gốc và ánh xạ 100% theo ID từ bảng giá mốc cuối cùng.
      3. Nếu căn phòng nào ở mốc cũ không còn tồn tại ở mốc cuối cùng, gán bằng Median giá của mốc cuối
         theo đúng (neighbourhood_cleansed, room_type).
    """
    reports = []
    latest_snapshot = max(all_dfs.keys())
    ref_df = all_dfs[latest_snapshot]

    logger.info("  [CHÍNH SÁCH GIÁ MỐC CUỐI]: Sử dụng duy nhất bảng giá của mốc: %s", latest_snapshot)

    # 1. Tính toán trước giá trung vị theo nhóm của mốc cuối cùng để làm phương án dự phòng
    group_medians_latest = ref_df.groupby(["neighbourhood_cleansed", "room_type"], observed=True)["price"].median().to_dict()
    global_median_latest = ref_df["price"].median()

    # Điền giá thiếu cho chính mốc cuối cùng trước nếu có
    ref_keys = list(zip(ref_df["neighbourhood_cleansed"], ref_df["room_type"]))
    ref_imputed = pd.Series([group_medians_latest.get(k, global_median_latest) for k in ref_keys], index=ref_df.index)
    ref_df["price"] = ref_df["price"].fillna(ref_imputed)

    # Bảng giá chuẩn tuyệt đối của mốc cuối cùng theo ID
    id_to_latest_price = ref_df.set_index("id")["price"].dropna().to_dict()

    for snap, df in all_dfs.items():
        if snap == latest_snapshot:
            # Mốc cuối cùng: đã có giá chuẩn
            continue

        # ĐẾM SỐ PHÒNG CÓ GIÁ GỐC TRƯỚC ĐÓ BỊ THAY THẾ
        original_prices_count = df["price"].notna().sum()

        # BƯỚC 1: ÁNH XẠ GIÁ THEO ID TỪ MỐC CUỐI CÙNG (GHI ĐÈ HOÀN TOÀN GIÁ GỐC NẾU CÓ)
        price_from_latest_id = df["id"].map(id_to_latest_price)

        # BƯỚC 2: VỚI CÁC PHÒNG CŨ KHÔNG XUẤT HIỆN Ở MỐC CUỐI, DÙNG MEDIAN THEO NHÓM CỦA MỐC CUỐI
        snap_keys = list(zip(df["neighbourhood_cleansed"], df["room_type"]))
        group_imputed_latest = pd.Series([group_medians_latest.get(k, global_median_latest) for k in snap_keys], index=df.index)

        # GHI ĐÈ 100% CỘT PRICE
        df["price"] = price_from_latest_id.fillna(group_imputed_latest)

        reports.append({
            "rule": "QA-00: Thay thế toàn bộ giá bằng giá mốc cuối cùng (2026-06-16)",
            "condition": f"Mốc {snap} có {original_prices_count} phòng có giá gốc -> Thay thế 100% bằng giá mốc {latest_snapshot}",
            "reason": "Chính sách phân tích: chỉ dùng giá của mốc cuối cùng, xoá bỏ giá các mốc trước để đồng nhất thị trường",
            "action": "THAY THẾ TOÀN BỘ BẰNG GIÁ MỐC CUỐI",
            "rows_before": len(df),
            "rows_after": len(df),
            "rows_affected": len(df),
            "snapshot": snap
        })
        logger.info(
            "  ✓ Mốc %s: Đã thay thế toàn bộ giá (%s giá gốc bị ghi đè) bằng bảng giá mốc cuối %s",
            snap, str(original_prices_count), latest_snapshot
        )

    return reports


def run_qa(raw_paths: dict, cfg: dict) -> tuple:
    """Chạy QA cho tất cả các mốc chụp.

    Returns:
        all_clean: dict[snapshot → DataFrame sạch]
        all_reports: list[dict] – tổng hợp báo cáo QA
    """
    logger.info("═══ BẮT ĐẦU QA ═══")
    all_raw = {}
    for snapshot, paths in raw_paths.items():
        listings_path = paths["listings"]
        df = pd.read_csv(listings_path, compression="gzip" if str(listings_path).endswith(".gz") else None)
        df["_snapshot"] = snapshot
        all_raw[snapshot] = coerce_dtypes(df)

    # Áp dụng chính sách neo giá chỉ dùng giá của mốc cuối cùng theo yêu cầu
    all_reports = apply_latest_snapshot_price_policy(all_raw, cfg)

    all_clean = {}
    for snapshot, df in all_raw.items():
        logger.info("─── QA mốc: %s ───", snapshot)
        df_clean, reports = clean_listings(df, cfg)
        for r in reports:
            r["snapshot"] = snapshot
        all_reports.extend(reports)
        all_clean[snapshot] = df_clean
        logger.info(
            "  Kết quả: %d → %d dòng\n",
            len(df), len(df_clean),
        )

    # QA-05: Trùng giữa các mốc
    _, cross_report = rule_duplicates_across(all_clean, cfg)
    cross_report["snapshot"] = "all"
    all_reports.append(cross_report)

    # Lưu report
    report_dir = Path(cfg.get("reports_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(all_reports)
    report_path = report_dir / "qa_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info("Đã lưu: %s", report_path)

    # Lưu dữ liệu sạch (Tối ưu: chỉ giữ lại các cột phục vụ tính KPI & Trực quan hoá để dung lượng < 50MB commit được lên Git)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_cols = [
        "id", "name", "host_id", "host_is_superhost", "host_response_rate",
        "neighbourhood_cleansed", "latitude", "longitude", "room_type",
        "accommodates", "bedrooms", "beds", "price", "minimum_nights",
        "maximum_nights", "availability_365", "number_of_reviews",
        "review_scores_rating", "review_scores_cleanliness", "reviews_per_month",
        "_qa_flag", "_snapshot"
    ]

    for snapshot, df_clean in all_clean.items():
        out_path = processed_dir / f"listings_clean_{snapshot}.csv"
        # Lọc các cột phân tích hiện có để tối ưu dung lượng
        cols_to_save = [c for c in analysis_cols if c in df_clean.columns]
        df_clean[cols_to_save].to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info("Đã lưu dữ liệu sạch tối ưu: %s (%.1f MB, %d dòng)", out_path.name, size_mb, len(df_clean))

    logger.info("═══ HOÀN TẤT QA ═══\n")
    return all_clean, all_reports
