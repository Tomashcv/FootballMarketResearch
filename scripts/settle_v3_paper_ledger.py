from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format.*")
warnings.filterwarnings("ignore", message="Parsing dates.*")

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import build_football_data_full_scope_1x2 as fd  # noqa: E402
from src.paper_trading.v3_pipeline import ensure_dirs, load_ledger, result_lookup_from_raw, settle_open_ledger, write_ledger  # noqa: E402


def build_results_lookup() -> dict[str, str]:
    fd.DATA_ROOTS = [fd.ROOT / "data/raw"]
    _inventory, norm = fd.discover_and_normalize()
    current, _warnings = __import__("src.paper_trading.v3_pipeline", fromlist=["select_current_raw_norm"]).select_current_raw_norm(norm)
    if current.empty:
        return {}
    x1, _marketed, _skipped, _warnings = __import__("src.paper_trading.v3_pipeline", fromlist=["build_paper_market_dataset"]).build_paper_market_dataset(fd, current)
    if x1.empty:
        return {}
    x1 = x1[x1["result_1x2"].astype(str).str.upper().isin(["H", "D", "A"])].copy()
    return result_lookup_from_raw(x1)


def main() -> None:
    ensure_dirs()
    ledger = load_ledger()
    lookup = build_results_lookup()
    settled = settle_open_ledger(ledger, lookup)
    write_ledger(settled)
    changed = int((ledger.get("status", pd.Series(dtype=str)) != settled.get("status", pd.Series(dtype=str))).sum()) if len(ledger) == len(settled) else 0
    print(f"settled_updates={changed} open_remaining={int(settled['status'].eq('OPEN').sum()) if not settled.empty else 0}")


if __name__ == "__main__":
    main()
