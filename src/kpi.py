"""
src/kpi.py – Tính toán các chỉ số hiệu suất chính (KPIs)
=========================================================

BÊN ĐẶT HÀNG: CHỦ NHÀ AIRBNB (HOST / PROSPECTIVE HOST)
─────────────────────────────────────────────────────────
Chủ nhà cần gì khi đầu tư và vận hành Airbnb?
1. Định giá phòng thế nào để tối đa hoá lợi nhuận nhưng không bị ế?
2. Tỷ lệ lấp đầy (occupancy rate) và doanh thu ước tính theo từng khu vực tại Sydney?
3. Trở thành Superhost có giúp tăng giá phòng và tăng doanh thu thực tế hay không?
4. Khách thuê phàn nàn về điều gì nhiều nhất (từ LLM) để cải tạo và bảo trì nhà?

TẠI SAO LÀM THẾ NÀY MÀ KHÔNG LÀM KHÁC? (Quyết định thiết kế):
───────────────────────────────────────────────────────────────
1. DÙNG MEDIAN THAY VÌ MEAN CHO GIÁ & DOANH THU:
   - Phân phối giá Airbnb bị lệch phải rất mạnh (right-skewed) do có các căn biệt thự/penthouse hàng chục ngàn đô.
   - Dùng mean sẽ bị kéo lệch ảo lên cao, khiến chủ nhà định giá quá đắt và bị ế phòng.
   - Median (trung vị) phản ánh chính xác điểm neo thị trường đại chúng (Bài 3, Bài 10).

2. CÔNG THỨC OCCUPANCY RATE TỪ availability_365:
   - Occupancy = (365 - availability_365) / 365.
   - Đây là công thức chuẩn được cộng đồng nghiên cứu Inside Airbnb công nhận khi không có dữ liệu booking thực tế.
   - Bổ sung ước tính doanh thu tháng: price * 30 * occupancy_rate.

3. TÍNH KPI SAU QA (CLEANED DATA):
   - Đề bài bắt buộc KPI phải tính trên dữ liệu đã làm sạch để loại bỏ listing lỗi giá (< $10 hoặc > $10,000).

4. GROUPBY VÀ AGG VECTOR HOÁ:
   - Sử dụng `.groupby(['neighbourhood_cleansed', 'room_type']).agg(...)` của pandas (Bài 5) thay vì lặp qua từng dòng,
     đảm bảo tốc độ nhanh và code ngắn gọn, chuẩn Pythonic.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_listing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung các cột chỉ số phái sinh cấp listing (row-level metrics).
    
    Tại sao tính ở cấp dòng trước?
    - Giúp việc gom nhóm (group by) theo mốc thời gian, khu vực hoặc room_type
      trở nên đồng nhất và tái sử dụng được ở nhiều bảng tổng hợp.
    """
    df = df.copy()
    
    # 1. Tỷ lệ lấp đầy ước tính (Occupancy Rate)
    # Giới hạn trong khoảng [0, 1]
    if "availability_365" in df.columns:
        df["occupancy_rate"] = (1.0 - (df["availability_365"].clip(0, 365) / 365.0)).clip(0.0, 1.0)
    else:
        df["occupancy_rate"] = np.nan

    # 2. Doanh thu ước tính mỗi tháng (AUD)
    # Công thức: Giá đêm * 30 ngày * Tỷ lệ lấp đầy
    if "price" in df.columns and "occupancy_rate" in df.columns:
        df["estimated_monthly_revenue"] = df["price"] * 30 * df["occupancy_rate"]
    else:
        df["estimated_monthly_revenue"] = np.nan

    # 3. Chuyển đổi cờ Superhost thành boolean chuẩn
    if "host_is_superhost" in df.columns:
        df["is_superhost"] = df["host_is_superhost"].fillna(False).astype(bool)
    else:
        df["is_superhost"] = False

    return df


def compute_host_kpis_overall(df: pd.DataFrame, snapshot_label: str = "all") -> dict:
    """Tính các KPI cấp toàn thị trường Sydney phục vụ quyết định của chủ nhà."""
    kpi_dict = {
        "snapshot": snapshot_label,
        "total_active_listings": int(len(df)),
        "median_price_aud": round(float(df["price"].median()), 2) if "price" in df else np.nan,
        "mean_price_aud": round(float(df["price"].mean()), 2) if "price" in df else np.nan,
        "median_occupancy_rate": round(float(df["occupancy_rate"].median() * 100), 2) if "occupancy_rate" in df else np.nan,
        "mean_occupancy_rate": round(float(df["occupancy_rate"].mean() * 100), 2) if "occupancy_rate" in df else np.nan,
        "median_monthly_revenue_aud": round(float(df["estimated_monthly_revenue"].median()), 2) if "estimated_monthly_revenue" in df else np.nan,
        "superhost_percentage": round(float(df["is_superhost"].mean() * 100), 2) if "is_superhost" in df else np.nan,
        "avg_rating_score": round(float(df["review_scores_rating"].mean()), 2) if "review_scores_rating" in df else np.nan,
        "avg_cleanliness_score": round(float(df["review_scores_cleanliness"].mean()), 2) if "review_scores_cleanliness" in df else np.nan,
    }
    
    # Superhost Premium: Chênh lệch giá giữa Superhost vs Non-superhost
    if "is_superhost" in df.columns and "price" in df.columns:
        price_super = df[df["is_superhost"]]["price"].median()
        price_normal = df[~df["is_superhost"]]["price"].median()
        kpi_dict["superhost_median_price"] = round(float(price_super), 2)
        kpi_dict["non_superhost_median_price"] = round(float(price_normal), 2)
        kpi_dict["superhost_price_premium_pct"] = round(float(((price_super - price_normal) / price_normal) * 100), 2) if price_normal > 0 else 0.0

    return kpi_dict


def compute_neighbourhood_kpis(df: pd.DataFrame, area_col: str = "neighbourhood_cleansed") -> pd.DataFrame:
    """Tính KPI theo từng khu vực (quận/vùng) tại Sydney.
    
    Mục đích cho Chủ nhà:
    - Biết khu vực nào có giá thuê tốt nhất, khu nào có tỷ lệ lấp đầy cao nhất để chọn địa điểm đầu tư.
    """
    if area_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột phân vùng {area_col}")

    grouped = df.groupby(area_col, observed=True).agg(
        listing_count=("id", "count"),
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        median_occupancy=("occupancy_rate", "median"),
        mean_occupancy=("occupancy_rate", "mean"),
        median_monthly_revenue=("estimated_monthly_revenue", "median"),
        superhost_ratio=("is_superhost", "mean"),
        avg_rating=("review_scores_rating", "mean"),
        avg_review_count=("number_of_reviews", "mean"),
    ).reset_index()

    # Định dạng %
    grouped["median_occupancy_pct"] = (grouped["median_occupancy"] * 100).round(2)
    grouped["mean_occupancy_pct"] = (grouped["mean_occupancy"] * 100).round(2)
    grouped["superhost_pct"] = (grouped["superhost_ratio"] * 100).round(2)
    grouped = grouped.drop(columns=["median_occupancy", "mean_occupancy", "superhost_ratio"])

    # Làm tròn số
    for col in ["median_price", "mean_price", "median_monthly_revenue", "avg_rating", "avg_review_count"]:
        grouped[col] = grouped[col].round(2)

    return grouped.sort_values(by="median_monthly_revenue", ascending=False)


def compute_room_type_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Tính KPI theo từng loại phòng (Entire home vs Private room vs Shared room)."""
    if "room_type" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("room_type", observed=True).agg(
        listing_count=("id", "count"),
        median_price=("price", "median"),
        median_occupancy=("occupancy_rate", "median"),
        median_revenue=("estimated_monthly_revenue", "median"),
        superhost_pct=("is_superhost", lambda s: round(s.mean() * 100, 2)),
        avg_rating=("review_scores_rating", "mean"),
    ).reset_index()

    grouped["median_occupancy_pct"] = (grouped["median_occupancy"] * 100).round(2)
    grouped = grouped.drop(columns=["median_occupancy"])
    return grouped.sort_values(by="median_revenue", ascending=False)


def compute_all_kpis(cleaned_snapshots: dict, cfg: dict) -> dict:
    """Hàm trung tâm điều phối việc tính toàn bộ KPI qua các mốc thời gian.
    
    Lưu kết quả ra:
    - data/processed/kpi_overall_summary.csv
    - data/processed/kpi_by_snapshot.csv
    - data/processed/kpi_by_neighbourhood.csv
    - data/processed/kpi_by_room_type.csv
    """
    logger.info("═══ BẮT ĐẦU TÍNH TOÁN BỘ KPI CHO CHỦ NHÀ AIRBNB ═══")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_kpis = []
    enriched_dfs = {}

    for snapshot, df in cleaned_snapshots.items():
        enriched = calculate_listing_metrics(df)
        enriched_dfs[snapshot] = enriched
        
        # KPI theo mốc
        kpi_m = compute_host_kpis_overall(enriched, snapshot_label=snapshot)
        snapshot_kpis.append(kpi_m)

    df_snapshot_kpis = pd.DataFrame(snapshot_kpis)
    df_snapshot_kpis.to_csv(processed_dir / "kpi_by_snapshot.csv", index=False)
    logger.info("✓ Đã xuất kpi_by_snapshot.csv")

    # Ghép toàn bộ các mốc (hoặc lấy mốc mới nhất) để phân tích không gian
    all_listings = pd.concat(enriched_dfs.values(), ignore_index=True)
    latest_snapshot = max(cleaned_snapshots.keys())
    df_latest = enriched_dfs[latest_snapshot]

    # KPI theo khu vực (ở mốc gần nhất để chủ nhà có thông tin cập nhật nhất)
    area_col = cfg.get("kpi", {}).get("area_col", "neighbourhood_cleansed")
    df_neighbourhood = compute_neighbourhood_kpis(df_latest, area_col=area_col)
    df_neighbourhood.to_csv(processed_dir / "kpi_by_neighbourhood.csv", index=False)
    logger.info("✓ Đã xuất kpi_by_neighbourhood.csv")

    # KPI theo room_type
    df_room_type = compute_room_type_kpis(df_latest)
    df_room_type.to_csv(processed_dir / "kpi_by_room_type.csv", index=False)
    logger.info("✓ Đã xuất kpi_by_room_type.csv")

    logger.info("═══ HOÀN THÀNH TÍNH KPI ═══\n")
    return {
        "by_snapshot": df_snapshot_kpis,
        "by_neighbourhood": df_neighbourhood,
        "by_room_type": df_room_type,
        "enriched_snapshots": enriched_dfs,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print("KPI module ready.")
