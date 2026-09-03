from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.paper_trading.ah_e0_pipeline import LATEST_CANDIDATE_PICKS, LEDGER_PATH, append_new_picks_to_ledger, ensure_dirs, load_ledger, write_ledger  # noqa: E402


def main() -> None:
    ensure_dirs()
    ledger = load_ledger()
    picks = pd.read_csv(LATEST_CANDIDATE_PICKS, low_memory=False) if LATEST_CANDIDATE_PICKS.exists() else pd.DataFrame()
    updated, new_rows = append_new_picks_to_ledger(ledger, picks)
    write_ledger(updated)
    print(f"ledger={LEDGER_PATH} existing={len(ledger)} new={len(new_rows)} total={len(updated)}")


if __name__ == "__main__":
    main()
