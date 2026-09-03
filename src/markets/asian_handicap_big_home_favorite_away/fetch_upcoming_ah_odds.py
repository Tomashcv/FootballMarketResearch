import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.common.paths import get_market_output_dir


MARKET_NAME = "asian_handicap_big_home_favorite_away"

SPORT_KEYS = {
    "E0": "soccer_epl",
}

DEFAULT_REGIONS = "uk,eu"
DEFAULT_MARKETS = "spreads"
DEFAULT_ODDS_FORMAT = "decimal"
DEFAULT_DATE_FORMAT = "iso"

LIVE_RULE_THRESHOLD = -1.25


def get_api_key():
    load_dotenv()

    api_key = os.getenv("ODDS_API_KEY")

    if api_key is None or api_key.strip() == "":
        raise ValueError(
            "ODDS_API_KEY não encontrada. Cria um ficheiro .env com ODDS_API_KEY=..."
        )

    return api_key.strip()


def get_sport_key(league_code):
    if league_code not in SPORT_KEYS:
        raise ValueError(f"Não tenho sport key configurada para a liga: {league_code}")

    return SPORT_KEYS[league_code]


def request_upcoming_odds(api_key, sport_key, regions, bookmakers):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": DEFAULT_MARKETS,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
    }

    if bookmakers is not None and bookmakers.strip() != "":
        params["bookmakers"] = bookmakers.strip()

    response = requests.get(url, params=params, timeout=30)

    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")

    print("API status:", response.status_code)

    if remaining is not None:
        print("Requests remaining:", remaining)

    if used is not None:
        print("Requests used:", used)

    if response.status_code != 200:
        print("Resposta da API:")
        print(response.text)
        response.raise_for_status()

    return response.json()


def parse_datetime(value):
    if value is None:
        return ""

    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)

        if pd.isna(parsed):
            return str(value)

        return parsed.isoformat()
    except Exception:
        return str(value)


def extract_spread_outcomes(event):
    rows = []

    event_id = event.get("id", "")
    commence_time = parse_datetime(event.get("commence_time"))
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")

    bookmakers = event.get("bookmakers", [])

    for bookmaker in bookmakers:
        bookmaker_key = bookmaker.get("key", "")
        bookmaker_title = bookmaker.get("title", bookmaker_key)

        markets = bookmaker.get("markets", [])

        for market in markets:
            market_key = market.get("key", "")

            if market_key != "spreads":
                continue

            last_update = parse_datetime(market.get("last_update"))
            outcomes = market.get("outcomes", [])

            home_outcome = None
            away_outcome = None

            for outcome in outcomes:
                outcome_name = outcome.get("name", "")

                if outcome_name == home_team:
                    home_outcome = outcome

                if outcome_name == away_team:
                    away_outcome = outcome

            if home_outcome is None or away_outcome is None:
                continue

            home_point = home_outcome.get("point")
            away_point = away_outcome.get("point")
            home_price = home_outcome.get("price")
            away_price = away_outcome.get("price")

            if home_point is None or away_point is None:
                continue

            if home_price is None or away_price is None:
                continue

            try:
                home_point = float(home_point)
                away_point = float(away_point)
                home_price = float(home_price)
                away_price = float(away_price)
            except ValueError:
                continue

            row = {
                "event_id": event_id,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker_key": bookmaker_key,
                "bookmaker": bookmaker_title,
                "market": "spreads",
                "last_update": last_update,
                "home_point": home_point,
                "away_point": away_point,
                "home_odds": home_price,
                "away_odds": away_price,
            }

            rows.append(row)

    return rows


def build_candidates(spread_rows, threshold):
    candidate_rows = []

    for row in spread_rows:
        home_point = float(row["home_point"])
        away_point = float(row["away_point"])
        away_odds = float(row["away_odds"])

        if away_odds <= 1.0:
            continue

        is_big_home_favourite = False

        if home_point <= threshold:
            is_big_home_favourite = True

        if away_point >= abs(threshold):
            is_big_home_favourite = True

        if not is_big_home_favourite:
            continue

        candidate = {
            "match_date": row["commence_time"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "market": "Asian Handicap / Spreads",
            "variant": "api_spreads",
            "ah_line": home_point,
            "away_handicap": away_point,
            "away_odds": away_odds,
            "bookmaker": row["bookmaker"],
            "bookmaker_key": row["bookmaker_key"],
            "rule": f"Away AH when home line <= {threshold}",
            "candidate_status": "candidate",
            "notes": "Generated from upcoming API spreads odds.",
            "event_id": row["event_id"],
            "last_update": row["last_update"],
            "home_odds": row["home_odds"],
        }

        candidate_rows.append(candidate)

    return candidate_rows


def choose_best_candidate_per_event(candidates):
    if len(candidates) == 0:
        return []

    dataframe = pd.DataFrame(candidates)

    dataframe = dataframe.sort_values(
        [
            "event_id",
            "away_handicap",
            "away_odds",
        ],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    best_rows = []

    for _, group in dataframe.groupby("event_id"):
        best = group.sort_values(
            ["away_odds"],
            ascending=[False]
        ).iloc[0]

        best_rows.append(best.to_dict())

    return best_rows


def save_outputs(league_code, raw_data, spread_rows, candidates, best_candidates):
    market_output_dir = get_market_output_dir(league_code, MARKET_NAME)

    raw_dir = market_output_dir / "raw_api"
    paper_dir = market_output_dir / "paper_tracking"

    raw_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_json_path = raw_dir / f"upcoming_ah_odds_raw_{timestamp}.json"
    all_spreads_path = paper_dir / "upcoming_all_spreads.csv"
    candidates_path = paper_dir / "upcoming_candidates.csv"
    best_candidates_path = paper_dir / "upcoming_best_candidates.csv"

    raw_json_path.write_text(
        json.dumps(raw_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    pd.DataFrame(spread_rows).to_csv(all_spreads_path, index=False)
    pd.DataFrame(candidates).to_csv(candidates_path, index=False)
    pd.DataFrame(best_candidates).to_csv(best_candidates_path, index=False)

    return {
        "raw_json_path": raw_json_path,
        "all_spreads_path": all_spreads_path,
        "candidates_path": candidates_path,
        "best_candidates_path": best_candidates_path,
    }


def print_summary(spread_rows, candidates, best_candidates):
    print("")
    print("=== UPCOMING AH ODDS SUMMARY ===")
    print("Spread rows:", len(spread_rows))
    print("Candidate rows:", len(candidates))
    print("Best candidates:", len(best_candidates))

    if len(candidates) == 0:
        print("")
        print("Sem candidatos agora.")
        print("Isto pode ser normal se a Premier League ainda não tiver jogos/odds AH disponíveis.")
        return

    display = pd.DataFrame(best_candidates)

    columns = [
        "match_date",
        "home_team",
        "away_team",
        "ah_line",
        "away_handicap",
        "away_odds",
        "bookmaker",
        "rule",
    ]

    existing_columns = []

    for column in columns:
        if column in display.columns:
            existing_columns.append(column)

    print("")
    print("=== BEST CANDIDATES ===")
    print(display[existing_columns].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--bookmakers", default="")
    parser.add_argument("--threshold", type=float, default=LIVE_RULE_THRESHOLD)
    args = parser.parse_args()

    league_code = args.league.upper()

    api_key = get_api_key()
    sport_key = get_sport_key(league_code)

    print("League:", league_code)
    print("Sport key:", sport_key)
    print("Regions:", args.regions)
    print("Markets:", DEFAULT_MARKETS)
    print("Threshold:", args.threshold)

    raw_data = request_upcoming_odds(
        api_key=api_key,
        sport_key=sport_key,
        regions=args.regions,
        bookmakers=args.bookmakers
    )

    spread_rows = []

    for event in raw_data:
        event_rows = extract_spread_outcomes(event)

        for row in event_rows:
            spread_rows.append(row)

    candidates = build_candidates(spread_rows, args.threshold)
    best_candidates = choose_best_candidate_per_event(candidates)

    paths = save_outputs(
        league_code=league_code,
        raw_data=raw_data,
        spread_rows=spread_rows,
        candidates=candidates,
        best_candidates=best_candidates
    )

    print_summary(spread_rows, candidates, best_candidates)

    print("")
    print("Ficheiros guardados:")
    for key, path in paths.items():
        print(key + ":", path)


if __name__ == "__main__":
    main()
