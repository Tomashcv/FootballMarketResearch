from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if __package__ is None or __package__ == "":
    sys.path.append(str(ROOT))

from src.paper_trading.v3_pipeline import LATEST_WARNINGS, PIPELINE_REPORT, ensure_dirs, write_pipeline_report  # noqa: E402


def run_step(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script)], check=True)


def main() -> None:
    ensure_dirs()
    warnings: list[str] = []
    decision = "v3_paper_pipeline_ready_for_sportsedge_integration_research_only"
    try:
        run_step("scripts/run_v3_paper_predictions.py")
        run_step("scripts/update_v3_paper_ledger.py")
        run_step("scripts/settle_v3_paper_ledger.py")
        run_step("scripts/build_v3_paper_html_report.py")
    except subprocess.CalledProcessError as exc:
        decision = "v3_paper_pipeline_blocked_data_quality"
        warnings.append(f"Pipeline step failed: {exc.cmd} exit={exc.returncode}")
        raise
    finally:
        if LATEST_WARNINGS.exists():
            try:
                frame = pd.read_csv(LATEST_WARNINGS)
                for col in frame.columns:
                    warnings.extend(frame[col].dropna().astype(str).tolist())
            except Exception as exc:  # pragma: no cover
                warnings.append(f"Could not read latest warnings: {exc}")
        write_pipeline_report(decision, warnings)
    print(decision)
    print(f"pipeline_report={PIPELINE_REPORT}")


if __name__ == "__main__":
    main()
