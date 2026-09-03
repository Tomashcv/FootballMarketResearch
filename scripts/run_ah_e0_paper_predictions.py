from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format.*")
warnings.filterwarnings("ignore", message="Parsing dates.*")

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.paper_trading.ah_e0_pipeline import (  # noqa: E402
    LATEST_AH_COLUMNS,
    LATEST_CANDIDATE_PICKS,
    LATEST_SKIPPED,
    LATEST_WARNINGS,
    ah_columns_available,
    build_snapshot_manifest,
    ensure_dirs,
    normalize_prediction_rows,
    read_latest_e0_raw,
    utc_stamp,
)


def main() -> None:
    ensure_dirs()
    run_id = f"ah_e0_{utc_stamp()}"
    warnings_out: list[str] = []
    raw, paths, columns = read_latest_e0_raw()
    columns.to_csv(LATEST_AH_COLUMNS, index=False)
    manifest = build_snapshot_manifest(run_id, raw, paths, warnings_out)
    if raw.empty:
        warnings_out.append("No local E0 raw season file found.")
        pd.DataFrame().to_csv(LATEST_CANDIDATE_PICKS, index=False)
        pd.DataFrame().to_csv(LATEST_SKIPPED, index=False)
        pd.DataFrame({"warning": warnings_out}).to_csv(LATEST_WARNINGS, index=False)
        print("ah_e0_paper_pipeline_blocked_data_quality")
        return
    if not ah_columns_available(columns):
        warnings_out.append("Missing required AH columns AHh and/or AvgAHA in latest E0 raw files.")
        pd.DataFrame().to_csv(LATEST_CANDIDATE_PICKS, index=False)
        pd.DataFrame().to_csv(LATEST_SKIPPED, index=False)
        pd.DataFrame({"warning": warnings_out}).to_csv(LATEST_WARNINGS, index=False)
        print("ah_e0_paper_pipeline_blocked_missing_ah_odds")
        return
    picks, skipped = normalize_prediction_rows(raw, run_id, str(manifest["run_id"]))
    picks.to_csv(LATEST_CANDIDATE_PICKS, index=False)
    skipped.to_csv(LATEST_SKIPPED, index=False)
    pd.DataFrame({"warning": warnings_out or ["none"]}).to_csv(LATEST_WARNINGS, index=False)
    print("ah_e0_paper_pipeline_ready_research_only")
    print(f"run_id={run_id} candidate_picks={len(picks)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
