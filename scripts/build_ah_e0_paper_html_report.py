from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.paper_trading.ah_e0_pipeline import HTML_REPORT, LATEST_AH_COLUMNS, LATEST_WARNINGS, ensure_dirs, load_ledger, write_html_report  # noqa: E402


def main() -> None:
    ensure_dirs()
    ledger = load_ledger()
    columns = pd.read_csv(LATEST_AH_COLUMNS, low_memory=False) if LATEST_AH_COLUMNS.exists() else pd.DataFrame()
    warnings = pd.read_csv(LATEST_WARNINGS, low_memory=False) if LATEST_WARNINGS.exists() else pd.DataFrame()
    write_html_report(ledger, columns, warnings)
    print(f"html_report={HTML_REPORT}")


if __name__ == "__main__":
    main()
