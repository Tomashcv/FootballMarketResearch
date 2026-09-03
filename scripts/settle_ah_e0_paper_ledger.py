from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format.*")
warnings.filterwarnings("ignore", message="Parsing dates.*")

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.paper_trading.ah_e0_pipeline import ensure_dirs, load_ledger, read_latest_e0_raw, result_lookup, settle_open_ledger, write_ledger  # noqa: E402


def main() -> None:
    ensure_dirs()
    ledger = load_ledger()
    raw, _paths, _columns = read_latest_e0_raw()
    settled = settle_open_ledger(ledger, result_lookup(raw))
    write_ledger(settled)
    changed = int((ledger.get("status", pd.Series(dtype=str)) != settled.get("status", pd.Series(dtype=str))).sum()) if len(ledger) == len(settled) else 0
    print(f"settled_updates={changed} open_remaining={int(settled['status'].eq('OPEN').sum()) if not settled.empty else 0}")


if __name__ == "__main__":
    main()
