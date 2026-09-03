from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if __package__ is None or __package__ == "":
    sys.path.append(str(ROOT))

from src.paper_trading.ah_e0_pipeline import LATEST_AH_COLUMNS, LATEST_WARNINGS, PIPELINE_REPORT, ah_columns_available, ensure_dirs, write_pipeline_report  # noqa: E402


def run_step(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script)], check=True)


def main() -> None:
    ensure_dirs()
    warnings_out: list[str] = []
    decision = "ah_e0_paper_pipeline_ready_research_only"
    try:
        run_step("scripts/run_ah_e0_paper_predictions.py")
        columns = pd.read_csv(LATEST_AH_COLUMNS, low_memory=False) if LATEST_AH_COLUMNS.exists() else pd.DataFrame()
        if not ah_columns_available(columns):
            decision = "ah_e0_paper_pipeline_blocked_missing_ah_odds"
        else:
            run_step("scripts/update_ah_e0_paper_ledger.py")
            run_step("scripts/settle_ah_e0_paper_ledger.py")
            run_step("scripts/build_ah_e0_paper_html_report.py")
    except subprocess.CalledProcessError as exc:
        decision = "ah_e0_paper_pipeline_blocked_data_quality"
        warnings_out.append(f"Pipeline step failed: {exc.cmd} exit={exc.returncode}")
        raise
    finally:
        if LATEST_WARNINGS.exists():
            try:
                frame = pd.read_csv(LATEST_WARNINGS)
                for col in frame.columns:
                    warnings_out.extend(frame[col].dropna().astype(str).tolist())
            except Exception as exc:  # pragma: no cover
                warnings_out.append(f"Could not read warnings: {exc}")
        columns = pd.read_csv(LATEST_AH_COLUMNS, low_memory=False) if LATEST_AH_COLUMNS.exists() else pd.DataFrame()
        write_pipeline_report(decision, columns, [w for w in warnings_out if w != "none"])
    print(decision)
    print(f"pipeline_report={PIPELINE_REPORT}")


if __name__ == "__main__":
    main()
