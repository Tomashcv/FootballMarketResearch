import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.paths import get_market_output_dir


MARKET_NAME = "asian_handicap_big_home_favorite_away"


LEDGER_COLUMNS = [
    "created_at",
    "match_date",
    "home_team",
    "away_team",
    "market",
    "variant",
    "ah_line",
    "away_handicap",
    "away_odds",
    "bookmaker",
    "rule",
    "stake",
    "status",
    "settled_profit",
    "notes",
    "event_id",
    "last_update",
]


def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path, low_memory=False)

    return pd.DataFrame()


def ensure_ledger(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(path, index=False)

    ledger = pd.read_csv(path, low_memory=False)

    for column in LEDGER_COLUMNS:
        if column not in ledger.columns:
            ledger[column] = ""

    ledger = ledger[LEDGER_COLUMNS].copy()

    return ledger


def make_unique_key(row):
    return "|".join([
        str(row.get("event_id", "")),
        str(row.get("home_team", "")),
        str(row.get("away_team", "")),
        str(row.get("market", "")),
        str(row.get("variant", "")),
        str(row.get("ah_line", "")),
        str(row.get("away_handicap", "")),
        str(row.get("bookmaker", "")),
    ])


def candidate_to_ledger_row(candidate, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "created_at": now,
        "match_date": candidate.get("match_date", ""),
        "home_team": candidate.get("home_team", ""),
        "away_team": candidate.get("away_team", ""),
        "market": candidate.get("market", "Asian Handicap / Spreads"),
        "variant": candidate.get("variant", "api_spreads"),
        "ah_line": candidate.get("ah_line", ""),
        "away_handicap": candidate.get("away_handicap", ""),
        "away_odds": candidate.get("away_odds", ""),
        "bookmaker": candidate.get("bookmaker", ""),
        "rule": candidate.get("rule", ""),
        "stake": 1.0,
        "status": status,
        "settled_profit": "",
        "notes": "Auto-added from upcoming best candidates. Watching only; not final paper bet yet.",
        "event_id": candidate.get("event_id", ""),
        "last_update": candidate.get("last_update", ""),
    }

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--status", default="watching", choices=["watching", "paper_bet"])
    args = parser.parse_args()

    league_code = args.league.upper()

    market_output_dir = get_market_output_dir(league_code, MARKET_NAME)
    paper_dir = market_output_dir / "paper_tracking"

    candidates_path = paper_dir / "upcoming_best_candidates.csv"
    ledger_path = paper_dir / "paper_ledger.csv"

    candidates = read_csv_if_exists(candidates_path)

    if len(candidates) == 0:
        print("Sem candidates para adicionar.")
        print("Esperado em:", candidates_path)
        return

    ledger = ensure_ledger(ledger_path)

    existing_keys = set()

    for _, row in ledger.iterrows():
        existing_keys.add(make_unique_key(row))

    rows_to_add = []

    for _, candidate in candidates.iterrows():
        candidate_dict = candidate.to_dict()
        key = make_unique_key(candidate_dict)

        if key in existing_keys:
            continue

        ledger_row = candidate_to_ledger_row(candidate_dict, args.status)
        rows_to_add.append(ledger_row)
        existing_keys.add(key)

    if len(rows_to_add) == 0:
        print("Nenhuma linha nova adicionada. Já estavam no ledger.")
        return

    new_rows = pd.DataFrame(rows_to_add)
    updated = pd.concat([ledger, new_rows], ignore_index=True)
    updated = updated[LEDGER_COLUMNS].copy()

    updated.to_csv(ledger_path, index=False)

    print("Linhas adicionadas:", len(rows_to_add))
    print("Ledger atualizado:", ledger_path)

    print("")
    print("Adicionadas:")
    display_columns = [
        "match_date",
        "home_team",
        "away_team",
        "ah_line",
        "away_handicap",
        "away_odds",
        "bookmaker",
        "status",
    ]

    print(new_rows[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
