
from pathlib import Path

data_dir = Path(__file__).resolve().parent / "data"

for path in sorted(data_dir.rglob("*")):
    if path.is_file():
        print(f"{path.relative_to(data_dir)} | {path.stat().st_size:,} bytes")
