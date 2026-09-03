from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.paper_trading.v3_pipeline import HTML_REPORT, LATEST_WARNINGS, ensure_dirs, load_ledger, read_latest_manifest, write_html_report  # noqa: E402


def main() -> None:
    ensure_dirs()
    ledger = load_ledger()
    warnings = pd.read_csv(LATEST_WARNINGS, low_memory=False) if LATEST_WARNINGS.exists() else pd.DataFrame()
    write_html_report(ledger, warnings, read_latest_manifest())
    print(f"html_report={HTML_REPORT}")


if __name__ == "__main__":
    main()
