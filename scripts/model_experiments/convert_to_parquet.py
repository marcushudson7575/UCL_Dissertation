"""Convert website + demo PostHog Excel exports to parquet for fast reload."""
from pathlib import Path
import pandas as pd

RAW = Path("/Users/marcushudson/Documents/GitHub/UCL Dissertation/raw data")
OUT = Path(__file__).parent / "parquet"
OUT.mkdir(exist_ok=True)

for name, path in [
    ("website", RAW / "website-export-2026-06-05.xlsx"),
    ("demo", RAW / "demo-data-export-2026-06-02.xlsx"),
]:
    print(f"--- {name}: {path.name}")
    xl = pd.ExcelFile(path)
    print("sheets:", xl.sheet_names)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        safe = sheet.replace(" ", "_").replace("/", "_")
        dest = OUT / f"{name}__{safe}.parquet"
        # object columns with mixed types break parquet; stringify them
        for c in df.columns[df.dtypes.eq(object)]:
            df[c] = df[c].astype("string")
        df.to_parquet(dest)
        print(f"  {sheet}: {df.shape[0]:,} rows x {df.shape[1]} cols -> {dest.name}")
        print(f"    cols: {list(df.columns)[:40]}")
print("DONE")
