from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from datetime import timedelta
import time

import pandas as pd


MATRIX_PATH = Path("data/processed/features/football_feature_matrix_v1_1.csv")
OUT_DIR = Path("data/raw_external/clubelo/daily_snapshots")
MANIFEST_PATH = Path("data/raw_external/clubelo/clubelo_download_manifest.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not MATRIX_PATH.exists():
    raise FileNotFoundError(f"Missing matrix: {MATRIX_PATH}")

df = pd.read_csv(MATRIX_PATH, usecols=["match_date"])
dates = pd.to_datetime(df["match_date"], errors="coerce").dropna().dt.date.unique()

# Use previous day as pre-match snapshot to avoid same-day update ambiguity.
snapshot_dates = sorted({d - timedelta(days=1) for d in dates})

rows = []

print(f"Unique match dates: {len(dates)}")
print(f"ClubElo snapshot dates to download: {len(snapshot_dates)}")

for i, d in enumerate(snapshot_dates, start=1):
    date_str = d.isoformat()
    out_path = OUT_DIR / f"clubelo_{date_str}.csv"

    if out_path.exists() and out_path.stat().st_size > 100:
        rows.append({
            "snapshot_date": date_str,
            "status": "already_exists",
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
        })
        continue

    url = f"http://api.clubelo.com/{date_str}"

    try:
        req = Request(url, headers={"User-Agent": "football-research-local-audit"})
        with urlopen(req, timeout=30) as r:
            content = r.read()

        if len(content) < 100:
            status = "too_small"
        else:
            out_path.write_bytes(content)
            status = "downloaded"

        rows.append({
            "snapshot_date": date_str,
            "status": status,
            "path": str(out_path),
            "bytes": out_path.stat().st_size if out_path.exists() else 0,
        })

    except HTTPError as e:
        rows.append({
            "snapshot_date": date_str,
            "status": f"http_error_{e.code}",
            "path": str(out_path),
            "bytes": 0,
        })

    except URLError as e:
        rows.append({
            "snapshot_date": date_str,
            "status": f"url_error_{e.reason}",
            "path": str(out_path),
            "bytes": 0,
        })

    if i % 50 == 0:
        print(f"{i}/{len(snapshot_dates)} done")

    # Polite delay.
    time.sleep(0.15)

manifest = pd.DataFrame(rows)
manifest.to_csv(MANIFEST_PATH, index=False)

print("\nDone.")
print(manifest["status"].value_counts(dropna=False))
print(f"Manifest: {MANIFEST_PATH}")
