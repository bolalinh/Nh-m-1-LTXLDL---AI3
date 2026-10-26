"""
Module: src/kpi.py
Học phần: Lập trình xử lý dữ liệu (UET.DSE2049) - Nhóm: DSE2049-C03-G01-sydney
Thành phố: Sydney, Australia | Tiền tệ: AUD

Chức năng: Tính toán toàn bộ 7 chỉ số KPI của thị trường Airbnb Sydney:
    1. Median Price          - Giá thuê trung vị đại diện (AUD/đêm)
    2. Average Price         - Giá thuê trung bình sau khi lọc ngoại lai P99 (AUD/đêm)
    3. Availability Rate     - Tỷ lệ ngày mở phòng trống trong 365 ngày (%)
    4. Estimated Occupancy   - Tỷ lệ lấp đầy phòng ước tính (%)
    5. Review Rate           - Tỷ lệ listing có ít nhất 1 review (%)
    6. Average Review Score  - Điểm đánh giá trung bình (quy chuẩn về thang 0-100)
    7. Listings per Host     - Số listing trung bình mỗi chủ nhà (+ % host đa-listing)

Đầu vào (do src/qa.py sinh ra, đã qua QA):
    data/processed/listings_clean.csv (hoặc listings_clean_<snapshot>.csv)

Đầu ra:
    data/processed/kpi_summary.csv             - Bảng tổng quan 7 KPI mốc mới nhất
    data/processed/kpi_by_time.csv             - 7 KPI theo 4 mốc chụp thời gian
    data/processed/kpi_by_area.csv             - 7 KPI theo 8 vùng phân tích (A01 - A08)
    data/processed/kpi_by_room_type.csv        - 7 KPI theo cơ cấu loại phòng
    data/processed/kpi_review_rate.csv         - Tỷ lệ review chi tiết theo snapshot × khu vực
    data/processed/kpi_review_score.csv        - Điểm review quy chuẩn theo snapshot × khu vực
    data/processed/kpi_listings_per_host.csv   - Chỉ số chủ nhà chi tiết theo từng snapshot

Cách chạy độc lập:
    python src/kpi.py --config configs/sydney.yaml
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import yaml

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Định nghĩa các hằng số tên cột dữ liệu
REVIEW_COUNT_COL = "number_of_reviews"
RATING_COL = "review_scores_rating"
HOST_COL = "host_id"
LISTING_COL = "id"
AREA_COL = "neighbourhood_cleansed"
ANALYSIS_AREA_COL = "analysis_area"


# ==============================================================================
# HÀM ĐỌC CẤU HÌNH & TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================

def load_config(config_path: str) -> dict:
    """Đọc file yaml cấu hình (dùng utf-8-sig để xử lý BOM nếu có)."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def clean_price(series: pd.Series) -> pd.Series:
    """Làm sạch cột giá tiền thô từ chuỗi ký tự ($350.00) sang số thực (float)."""
    return (
        series.astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace(['nan', 'None', '', 'NaN'], np.nan)
        .astype(float)
    )


def detect_rating_scale(s: pd.Series) -> str:
    """Phát hiện thang điểm của review_scores_rating trong một snapshot.

    Dữ liệu Inside Airbnb đời mới dùng thang 0-5, bản cũ dùng 0-100.
    Quy tắc: nếu mọi giá trị <= 5 thì là thang 0-5, ngược lại là 0-100.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "0-5"
    return "0-5" if s.max() <= 5.0 else "0-100"


# ==============================================================================
# PHẦN 1: 4 CHỈ SỐ KPI ĐẦU TIÊN (GIÁ CẢ VÀ TỶ LỆ LẤP ĐẦY)
# Vai trò bên liên quan: Cơ quan Quản lý Du lịch & Quy hoạch Đô thị
# ==============================================================================

def calculate_median_price(df: pd.DataFrame) -> float:
    """KPI 1: Median Price (Giá thuê trung vị).

    Mục tiêu: Phản ánh mức giá điển hình của du khách, không bị kéo lệch bởi căn hộ siêu sang.
    Công thức: Median(price) trên các căn có giá hợp lệ (> 0, không rỗng).
    Đơn vị: AUD/đêm.
    """
    col = 'price_clean' if 'price_clean' in df.columns else 'price'
    prices = df[col] if col == 'price_clean' else clean_price(df[col])
    valid_prices = prices[prices > 0].dropna()
    if len(valid_prices) == 0:
        return 0.0
    return round(float(valid_prices.median()), 2)


def calculate_average_price(df: pd.DataFrame):
    """KPI 2: Average Price (Giá thuê trung bình).

    Mục tiêu: Đo lường mức giá bình quân thị trường cho nhà đầu tư định giá phòng.
    Công thức: Mean(price) sau khi lọc bỏ giá ngoại lai vượt phân vị 99% (P99).
    Đơn vị: AUD/đêm.
    """
    col = 'price_clean' if 'price_clean' in df.columns else 'price'
    prices = df[col] if col == 'price_clean' else clean_price(df[col])
    valid_prices = prices[prices > 0].dropna()
    if len(valid_prices) == 0:
        return 0.0, 0.0
    raw_mean = round(float(valid_prices.mean()), 2)
    p99 = valid_prices.quantile(0.99)
    clean_prices = valid_prices[valid_prices <= p99]
    clean_mean = round(float(clean_prices.mean()), 2)
    return clean_mean, raw_mean


def calculate_availability_rate(df: pd.DataFrame) -> float:
    """KPI 3: Availability Rate (Tỷ lệ ngày mở phòng trống).

    Mục tiêu: Đo lường mức độ sẵn sàng cung ứng phòng của thị trường trong 365 ngày tới.
    Công thức: (Mean(availability_365) / 365) * 100%.
    Đơn vị: %.
    """
    if len(df) == 0:
        return 0.0
    s = pd.to_numeric(df['availability_365'], errors='coerce').clip(0, 365)
    return round(float((s.mean() / 365.0) * 100.0), 2)


def calculate_estimated_occupancy_rate(df: pd.DataFrame) -> float:
    """KPI 4: Estimated Occupancy Rate (Tỷ lệ lấp đầy phòng ước tính).

    Mục tiêu: Đo lường hiệu suất khai thác phòng và tác động tới quỹ nhà ở dân sinh.
    Công thức: 100% - Availability Rate = ((365 - Mean(availability_365)) / 365) * 100%.
    Đơn vị: %.
    """
    avail_rate = calculate_availability_rate(df)
    return round(100.0 - avail_rate, 2)


# ==============================================================================
# PHẦN 2: 3 CHỈ SỐ KPI TIẾP THEO (ĐÁNH GIÁ VÀ CHỦ NHÀ)
# Vai trò bên liên quan: Chủ nhà cho thuê (Landlord / Host)
# ==============================================================================

def kpi_review_rate(df: pd.DataFrame) -> dict:
    """KPI 5: Tỷ lệ listing có ít nhất 1 review.

    Công thức: Số listing có number_of_reviews >= 1 / Tổng số listing × 100%.
    Đơn vị: %.
    """
    s = pd.to_numeric(df[REVIEW_COUNT_COL], errors="coerce")
    valid = s[s.notna() & (s >= 0)]
    n_total = int(valid.shape[0])
    n_with = int((valid >= 1).sum())
    rate = round(n_with / n_total * 100, 2) if n_total else 0.0
    return {
        "n_listings": n_total,
        "n_with_reviews": n_with,
        "review_rate_pct": rate,
    }


def kpi_avg_review_score(df: pd.DataFrame, scale: str = "0-100") -> dict:
    """KPI 6: Điểm đánh giá trung bình, quy về thang 0-100 để so sánh giữa các snapshot.

    Công thức: Trung bình review_scores_rating trên các listing có điểm hợp lệ,
    nhân 20 nếu dữ liệu gốc ở thang 0-5.
    Đơn vị: điểm (thang 0-100).
    """
    s = pd.to_numeric(df[RATING_COL], errors="coerce")
    valid = s[s.notna() & (s >= 0)]
    if scale == "0-5":
        valid = valid[valid <= 5] * 20.0
    else:
        valid = valid[valid <= 100.0]
    n = int(valid.shape[0])
    mean = round(float(valid.mean()), 2) if n else 0.0
    return {"n_rated": n, "avg_review_score": mean, "rating_scale": scale}


def kpi_listings_per_host(df: pd.DataFrame) -> dict:
    """KPI 7: Số listing trung bình mỗi chủ nhà + % listing thuộc host đa-listing.

    Công thức: Tổng số listing / Số host phân biệt.
    Đơn vị: listing/chủ nhà.
    """
    d = df[df[HOST_COL].notna() & df[LISTING_COL].notna()].drop_duplicates(subset=[LISTING_COL])
    n_listings = int(d.shape[0])
    n_hosts = int(d[HOST_COL].nunique())
    per_host = round(n_listings / n_hosts, 2) if n_hosts else 0.0
    counts = d.groupby(HOST_COL)[LISTING_COL].nunique()
    n_multi = int(counts[counts >= 2].sum())
    pct_multi = round(n_multi / n_listings * 100, 2) if n_listings else 0.0
    return {
        "n_listings": n_listings,
        "n_hosts": n_hosts,
        "listings_per_host": per_host,
        "n_multi_host_listings": n_multi,
        "pct_listings_multi_host": pct_multi,
    }


# Các hàm tương thích để tích hợp giá trị vô hướng (scalar) vào các bảng phân tích
def calculate_review_rate(df: pd.DataFrame) -> float:
    """Trả về giá trị vô hướng tỷ lệ review (%) cho DataFrame."""
    return kpi_review_rate(df).get("review_rate_pct", 0.0)


def calculate_average_review_score(df: pd.DataFrame) -> float:
    """Trả về điểm đánh giá trung bình quy đổi thang 0-100 cho DataFrame."""
    scale = detect_rating_scale(df[RATING_COL]) if RATING_COL in df.columns else "0-5"
    return kpi_avg_review_score(df, scale=scale).get("avg_review_score", 0.0)


def calculate_listings_per_host(df: pd.DataFrame) -> float:
    """Trả về số listing bình quân mỗi chủ nhà cho DataFrame."""
    return kpi_listings_per_host(df).get("listings_per_host", 0.0)


def breakdown(df: pd.DataFrame, func) -> pd.DataFrame:
    """Tính KPI cho từng snapshot: một dòng tổng (ALL) + một dòng mỗi khu vực."""
    rows = []
    for snap, g in df.groupby("snapshot"):
        rows.append({"snapshot": snap, AREA_COL: "ALL", **func(g)})
        if AREA_COL in g.columns:
            for area, gg in g.groupby(AREA_COL):
                rows.append({"snapshot": snap, AREA_COL: area, **func(gg)})
    return pd.DataFrame(rows)


# ==============================================================================
# HÀM ĐIỀU PHỐI VÀ XUẤT CÁC BẢNG BÁO CÁO KPI (7 CHỈ SỐ LIỀN MẠCH)
# ==============================================================================

def run_kpi(config_path: str = "configs/sydney.yaml",
            processed_dir: Optional[str] = None,
            output_dir: Optional[str] = None) -> dict:
    """Hàm chạy toàn bộ quy trình tính toán 7 KPI theo file cấu hình và lưu kết quả."""
    cfg = load_config(config_path)
    snapshots = [str(s) for s in cfg.get("snapshots", [])]
    if not snapshots:
        snapshots = ["2025-09-12", "2025-12-12", "2026-03-21", "2026-06-16"]

    processed_dir = processed_dir or cfg.get("processed_dir", "data/processed")
    output_dir = output_dir or processed_dir
    os.makedirs(output_dir, exist_ok=True)

    # Đọc dữ liệu: Ưu tiên ghép các file snapshot riêng nếu có, hoặc dùng file listings_clean.csv
    frames = []
    for snap in snapshots:
        path = os.path.join(processed_dir, f"listings_clean_{snap}.csv")
        if os.path.exists(path):
            df_snap = pd.read_csv(path)
            df_snap["snapshot"] = snap
            frames.append(df_snap)

    if frames:
        data = pd.concat(frames, ignore_index=True)
    else:
        combined_path = os.path.join(processed_dir, "listings_clean.csv")
        if os.path.exists(combined_path):
            data = pd.read_csv(combined_path)
        else:
            fallback = os.path.join(processed_dir, "listings_subset.csv")
            data = pd.read_csv(fallback)

    if 'price_clean' not in data.columns and 'price' in data.columns:
        data['price_clean'] = clean_price(data['price'])

    data['availability_365'] = pd.to_numeric(data['availability_365'], errors='coerce').clip(0, 365)

    outputs = {}

    # 1. Phát hiện thang điểm đánh giá theo từng snapshot
    scales = {snap: detect_rating_scale(g[RATING_COL])
              for snap, g in data.groupby("snapshot")}
    for snap, sc in scales.items():
        print(f"   [THANG ĐIỂM] {snap}: {sc} -> quy về 0-100")

    # 2. Xuất 3 bảng KPI chuyên sâu của nhóm (KPI 5, 6, 7)
    tbl_rev_rate = breakdown(data, kpi_review_rate)
    path_rev_rate = os.path.join(output_dir, "kpi_review_rate.csv")
    tbl_rev_rate.to_csv(path_rev_rate, index=False)
    outputs["kpi_review_rate"] = tbl_rev_rate
    print(f"   [GHI] {path_rev_rate}: {len(tbl_rev_rate)} dòng")

    score_rows = []
    for snap, g in data.groupby("snapshot"):
        sc = scales[snap]
        score_rows.append({"snapshot": snap, AREA_COL: "ALL", **kpi_avg_review_score(g, sc)})
        if AREA_COL in g.columns:
            for area, gg in g.groupby(AREA_COL):
                score_rows.append({"snapshot": snap, AREA_COL: area, **kpi_avg_review_score(gg, sc)})
    tbl_score = pd.DataFrame(score_rows)
    path_score = os.path.join(output_dir, "kpi_review_score.csv")
    tbl_score.to_csv(path_score, index=False)
    outputs["kpi_review_score"] = tbl_score
    print(f"   [GHI] {path_score}: {len(tbl_score)} dòng")

    host_rows = [{"snapshot": snap, **kpi_listings_per_host(g)}
                 for snap, g in data.groupby("snapshot")]
    tbl_hosts = pd.DataFrame(host_rows)
    path_hosts = os.path.join(output_dir, "kpi_listings_per_host.csv")
    tbl_hosts.to_csv(path_hosts, index=False)
    outputs["kpi_listings_per_host"] = tbl_hosts
    print(f"   [GHI] {path_hosts}: {len(tbl_hosts)} dòng")

    # 3. Tạo bảng tổng kết toàn diện 7 KPI cho mốc mới nhất
    latest_snap = sorted(data["snapshot"].unique())[-1]
    df_latest = data[data["snapshot"] == latest_snap]

    med_price = calculate_median_price(df_latest)
    clean_mean, raw_mean = calculate_average_price(df_latest)
    avail_rate = calculate_availability_rate(df_latest)
    occ_rate = calculate_estimated_occupancy_rate(df_latest)
    rev_rate_dict = kpi_review_rate(df_latest)
    score_dict = kpi_avg_review_score(df_latest, scale=scales.get(latest_snap, "0-5"))
    host_dict = kpi_listings_per_host(df_latest)

    summary_rows = [
        {'kpi_name': 'Median Price', 'value': med_price, 'unit': 'AUD/night', 'description': 'Giá trung vị đại diện thị trường, loại trừ listing thiếu giá'},
        {'kpi_name': 'Average Price (Sau QA)', 'value': clean_mean, 'unit': 'AUD/night', 'description': 'Giá trung bình sau khi lọc ngoại lai phân vị 99%'},
        {'kpi_name': 'Average Price (Trước QA)', 'value': raw_mean, 'unit': 'AUD/night', 'description': 'Giá trung bình thô bị kéo lệch bởi giá ngoại lai'},
        {'kpi_name': 'Availability Rate', 'value': avail_rate, 'unit': '%', 'description': 'Tỷ lệ ngày mở phòng trong 365 ngày tới'},
        {'kpi_name': 'Estimated Occupancy Rate', 'value': occ_rate, 'unit': '%', 'description': 'Tỷ lệ lấp đầy phòng ước tính (365 - avail)/365'},
        {'kpi_name': 'Review Rate', 'value': rev_rate_dict['review_rate_pct'], 'unit': '%', 'description': f'Tỷ lệ listing có review ({rev_rate_dict["n_with_reviews"]}/{rev_rate_dict["n_listings"]})'},
        {'kpi_name': 'Average Review Score', 'value': score_dict['avg_review_score'], 'unit': 'điểm (0-100)', 'description': f'Điểm đánh giá trung bình quy về thang 100 ({score_dict["n_rated"]} listing)'},
        {'kpi_name': 'Listings per Host', 'value': host_dict['listings_per_host'], 'unit': 'listings/host', 'description': f'Số listing trung bình mỗi chủ nhà ({host_dict["n_hosts"]} hosts)'},
        {'kpi_name': 'Multi-host Listing Share', 'value': host_dict['pct_listings_multi_host'], 'unit': '%', 'description': 'Tỷ lệ listing thuộc các chủ nhà sở hữu từ 2 căn trở lên'}
    ]
    df_summary = pd.DataFrame(summary_rows)
    path_summary = os.path.join(output_dir, "kpi_summary.csv")
    df_summary.to_csv(path_summary, index=False)
    outputs["kpi_summary"] = df_summary
    print(f"   [GHI] {path_summary}: {len(df_summary)} chỉ số")

    # 4. Bảng tính 7 KPI theo 4 mốc thời gian
    time_rows = []
    for s in sorted(data["snapshot"].unique()):
        sub = data[data["snapshot"] == s]
        c_mean, _ = calculate_average_price(sub)
        scale_s = scales.get(s, "0-5")
        sc_d = kpi_avg_review_score(sub, scale=scale_s)
        h_d = kpi_listings_per_host(sub)
        time_rows.append({
            'snapshot': s,
            'total_listings': len(sub),
            'median_price_aud': calculate_median_price(sub),
            'average_price_clean_aud': c_mean,
            'availability_rate_pct': calculate_availability_rate(sub),
            'estimated_occupancy_rate_pct': calculate_estimated_occupancy_rate(sub),
            'review_rate_pct': calculate_review_rate(sub),
            'avg_review_score_100': sc_d['avg_review_score'],
            'listings_per_host': h_d['listings_per_host'],
            'pct_listings_multi_host': h_d['pct_listings_multi_host']
        })
    df_temporal = pd.DataFrame(time_rows)
    path_temporal = os.path.join(output_dir, "kpi_by_time.csv")
    df_temporal.to_csv(path_temporal, index=False)
    outputs["kpi_by_time"] = df_temporal
    print(f"   [GHI] {path_temporal}: {len(df_temporal)} dòng")

    # 5. Bảng tính 7 KPI theo 8 vùng phân tích A01 - A08 (mốc mới nhất)
    area_rows = []
    area_col = ANALYSIS_AREA_COL if ANALYSIS_AREA_COL in df_latest.columns else AREA_COL
    for a in sorted(df_latest[area_col].dropna().unique()):
        sub = df_latest[df_latest[area_col] == a]
        c_mean, _ = calculate_average_price(sub)
        sc_d = kpi_avg_review_score(sub, scale=scales.get(latest_snap, "0-5"))
        h_d = kpi_listings_per_host(sub)
        area_rows.append({
            'analysis_area': a,
            'total_listings': len(sub),
            'median_price_aud': calculate_median_price(sub),
            'average_price_clean_aud': c_mean,
            'availability_rate_pct': calculate_availability_rate(sub),
            'estimated_occupancy_rate_pct': calculate_estimated_occupancy_rate(sub),
            'review_rate_pct': calculate_review_rate(sub),
            'avg_review_score_100': sc_d['avg_review_score'],
            'listings_per_host': h_d['listings_per_host']
        })
    df_spatial = pd.DataFrame(area_rows)
    path_spatial = os.path.join(output_dir, "kpi_by_area.csv")
    df_spatial.to_csv(path_spatial, index=False)
    outputs["kpi_by_area"] = df_spatial
    print(f"   [GHI] {path_spatial}: {len(df_spatial)} dòng")

    # 6. Bảng tính KPI theo cơ cấu loại phòng
    room_rows = []
    for r in sorted(df_latest['room_type'].dropna().unique()):
        sub = df_latest[df_latest['room_type'] == r]
        c_mean, _ = calculate_average_price(sub)
        sc_d = kpi_avg_review_score(sub, scale=scales.get(latest_snap, "0-5"))
        room_rows.append({
            'room_type': r,
            'total_listings': len(sub),
            'median_price_aud': calculate_median_price(sub),
            'average_price_clean_aud': c_mean,
            'availability_rate_pct': calculate_availability_rate(sub),
            'estimated_occupancy_rate_pct': calculate_estimated_occupancy_rate(sub),
            'review_rate_pct': calculate_review_rate(sub),
            'avg_review_score_100': sc_d['avg_review_score']
        })
    df_room = pd.DataFrame(room_rows)
    path_room = os.path.join(output_dir, "kpi_by_room_type.csv")
    df_room.to_csv(path_room, index=False)
    outputs["kpi_by_room_type"] = df_room
    print(f"   [GHI] {path_room}: {len(df_room)} dòng")

    # 7. In tóm tắt tổng quan ra màn hình
    print("\n" + "=" * 68)
    print("HOÀN THÀNH TÍNH TOÀN BỘ 7 CHỈ SỐ KPI (SYDNEY - AUD)")
    print("=" * 68)
    for _, row in df_summary.iterrows():
        print(f"* {row['kpi_name']:<30}: {row['value']} {row['unit']}")

    print("\n--- Tóm tắt KPI (mức toàn thị trường theo từng snapshot) ---")
    for snap in sorted(data["snapshot"].unique()):
        r = outputs["kpi_review_rate"]
        s = outputs["kpi_review_score"]
        h = outputs["kpi_listings_per_host"]
        rr = r[(r["snapshot"] == snap) & (r[AREA_COL] == "ALL")].iloc[0]
        ss = s[(s["snapshot"] == snap) & (s[AREA_COL] == "ALL")].iloc[0]
        hh = h[h["snapshot"] == snap].iloc[0]
        print(f"[{snap}] Review Rate: {rr['review_rate_pct']}% "
              f"({rr['n_with_reviews']}/{rr['n_listings']}) | "
              f"Avg Score: {ss['avg_review_score']}/100 "
              f"(thang gốc {ss['rating_scale']}, {ss['n_rated']} listing) | "
              f"Listings/Host: {hh['listings_per_host']} "
              f"({hh['n_hosts']} host, {hh['pct_listings_multi_host']}% đa-listing)")

    return outputs


def run_kpi_computation(input_path: str = "data/processed/listings_clean.csv",
                        output_dir: str = "data/processed",
                        config: Optional[dict] = None):
    """Hàm wrapper đảm bảo tương thích ngược 100% với run_pipeline.py."""
    config_path = "configs/sydney.yaml"
    outputs = run_kpi(config_path=config_path, processed_dir=output_dir, output_dir=output_dir)
    return outputs["kpi_summary"], outputs["kpi_by_time"], outputs["kpi_by_area"], outputs["kpi_by_room_type"]


def main():
    parser = argparse.ArgumentParser(
        description="Tính toàn bộ 7 KPI thị trường Inside Airbnb Sydney (UET.DSE2049)."
    )
    parser.add_argument("--config", type=str, default="configs/sydney.yaml",
                        help="Đường dẫn file yaml cấu hình")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Thư mục chứa dữ liệu sạch (listings_clean.csv)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Thư mục ghi các file kpi_*.csv")
    args = parser.parse_args()

    print("=== TÍNH TOÀN BỘ 7 CHỈ SỐ KPI (SYDNEY, AUSTRALIA) ===")
    try:
        run_kpi(args.config, args.processed_dir, args.output_dir)
    except (FileNotFoundError, SystemExit) as e:
        print(f"[LỖI] {e}")
        sys.exit(1)
    print("=== HOÀN TẤT TÍNH KPI ===")


if __name__ == "__main__":
    main()