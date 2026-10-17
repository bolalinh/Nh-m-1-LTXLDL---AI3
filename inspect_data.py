
from pathlib import Path
import json
import pandas as pd

# Thu muc du an va du lieu
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# So dong moi lan doc de tranh ton bo du lieu vao RAM
CHUNK_SIZE = 100_000
SAMPLE_ROWS = 5


def inspect_csv(path):
    """Kiem tra cau truc va chat luong co ban cua file CSV/CSV.GZ."""
    print("\n" + "=" * 80)
    print(f"FILE: {path.relative_to(PROJECT_DIR)}")
    print(f"SIZE: {path.stat().st_size:,} bytes")

    try:
        # Doc mau de xem ten cot va du lieu
        sample = pd.read_csv(path, nrows=SAMPLE_ROWS)

        print(f"\nSo cot: {len(sample.columns)}")
        print("\nDanh sach cot:")
        print(sample.columns.tolist())

        print("\nKieu du lieu cua mau:")
        print(sample.dtypes.to_string())

        print(f"\n{SAMPLE_ROWS} dong mau:")
        print(sample.to_string(index=False))

        # Doc toan bo file theo tung chunk de thong ke
        total_rows = 0
        missing_counts = None
        duplicate_counts = 0
        columns = None

        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            if columns is None:
                columns = chunk.columns.tolist()
                missing_counts = pd.Series(
                    0, index=columns, dtype="int64"
                )

            total_rows += len(chunk)
            missing_counts += chunk.isna().sum()

            # Dem cac dong trung hoan toan trong tung chunk.
            # Day chua phat hien trung lap giua cac chunk.
            duplicate_counts += int(chunk.duplicated().sum())

        print(f"\nTong so dong: {total_rows:,}")
        print("\nDu lieu thieu theo cot:")

        missing_report = pd.DataFrame({
            "missing_count": missing_counts,
            "missing_percent": (
                missing_counts / total_rows * 100
                if total_rows > 0
                else 0
            ),
        })

        print(missing_report.to_string())

        print(
            "\nSo dong trung hoan toan trong cung chunk: "
            f"{duplicate_counts:,}"
        )

        if total_rows == 0:
            print("CANH BAO: File khong co du lieu.")

    except Exception as exc:
        print(f"LOI KHI DOC FILE: {type(exc).__name__}: {exc}")


def inspect_geojson(path):
    """Kiem tra cau truc file GeoJSON."""
    print("\n" + "=" * 80)
    print(f"GEOJSON: {path.relative_to(PROJECT_DIR)}")
    print(f"SIZE: {path.stat().st_size:,} bytes")

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

        print("Cac key cap cao:", list(data.keys()))

        features = data.get("features", [])
        print("So features:", len(features))

        if features:
            first = features[0]
            print("\nCac thuoc tinh cua feature dau tien:")
            print(json.dumps(
                first.get("properties", {}),
                ensure_ascii=False,
                indent=2
            ))

            print("\nLoai hinh hoc:")
            print(first.get("geometry", {}).get("type"))

    except Exception as exc:
        print(f"LOI KHI DOC GEOJSON: {type(exc).__name__}: {exc}")


def main():
    print("=" * 80)
    print("AIRBNB DATA INSPECTOR")
    print(f"Thu muc du an: {PROJECT_DIR}")
    print(f"Thu muc du lieu: {DATA_DIR}")

    if not DATA_DIR.exists():
        print("\nLOI: Khong tim thay thu muc data/.")
        return

    # Tu dong tim file trong tat ca thu muc con cua data/
    csv_files = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
        and (
            path.name.lower().endswith(".csv")
            or path.name.lower().endswith(".csv.gz")
        )
    )

    geojson_files = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".geojson", ".json"}
    )

    if not csv_files and not geojson_files:
        print("\nKhong tim thay file CSV, CSV.GZ hoac GeoJSON.")
        print("Hay kiem tra vi tri file trong thu muc data/.")
        return

    print(f"\nTim thay {len(csv_files)} file CSV/CSV.GZ")
    print(f"Tim thay {len(geojson_files)} file JSON/GeoJSON")

    for path in csv_files:
        inspect_csv(path)

    for path in geojson_files:
        inspect_geojson(path)

    print("\n" + "=" * 80)
    print("HOAN TAT KIEM TRA.")


if __name__ == "__main__":
    main()
