import argparse
import math
from pathlib import Path

import pandas as pd

from src.common.paths import get_league_matches_path
from src.common.paths import get_market_output_dir


MARKET_NAME = "asian_handicap_big_home_favorite_away"

THRESHOLDS = [-1.00, -1.25, -1.50, -1.75, -2.00]

VARIANTS = {
    "main": {
        "line_col": "AHh",
        "home_col": "AvgAHH",
        "away_col": "AvgAHA",
    },
    "closing": {
        "line_col": "AHCh",
        "home_col": "AvgCAHH",
        "away_col": "AvgCAHA",
    },
}


def split_handicap(handicap):
    handicap = float(handicap)
    scaled = handicap * 4.0
    rounded = round(scaled)

    if abs(scaled - rounded) > 0.000001:
        return [handicap]

    if rounded % 2 == 0:
        return [handicap]

    lower = math.floor(handicap * 2.0) / 2.0
    upper = math.ceil(handicap * 2.0) / 2.0

    return [lower, upper]


def profit_single_line(adjusted_margin, odds):
    if adjusted_margin > 0:
        return float(odds) - 1.0

    if adjusted_margin == 0:
        return 0.0

    return -1.0


def asian_profit(team_margin, handicap, odds):
    parts = split_handicap(handicap)
    total_profit = 0.0

    for part in parts:
        adjusted_margin = team_margin + part
        part_profit = profit_single_line(adjusted_margin, odds)
        total_profit += part_profit / len(parts)

    return total_profit


def calculate_z_score(profits):
    if len(profits) < 2:
        return 0.0

    mean_profit = profits.mean()
    standard_deviation = profits.std(ddof=1)

    if standard_deviation == 0:
        return 0.0

    return mean_profit / (standard_deviation / math.sqrt(len(profits)))


def calculate_max_drawdown(dataframe):
    if len(dataframe) == 0:
        return 0.0

    ordered = dataframe.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="coerce")
    ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    cumulative_profit = ordered["profit"].astype(float).cumsum()
    running_max = cumulative_profit.cummax()
    drawdown = running_max - cumulative_profit

    return float(drawdown.max())


def summarize(dataframe):
    if len(dataframe) == 0:
        return {
            "bets": 0,
            "wins": 0,
            "pushes": 0,
            "half_wins": 0,
            "half_losses": 0,
            "losses": 0,
            "profit": 0.0,
            "roi": 0.0,
            "avg_line": 0.0,
            "avg_odds": 0.0,
            "z_score": 0.0,
            "max_drawdown": 0.0,
        }

    profits = dataframe["profit"].astype(float)

    wins = int((profits > 0.51).sum())
    half_wins = int(((profits > 0.0) & (profits <= 0.51)).sum())
    pushes = int((profits == 0.0).sum())
    half_losses = int((profits == -0.5).sum())
    losses = int((profits == -1.0).sum())

    return {
        "bets": int(len(dataframe)),
        "wins": wins,
        "pushes": pushes,
        "half_wins": half_wins,
        "half_losses": half_losses,
        "losses": losses,
        "profit": float(profits.sum()),
        "roi": float(profits.mean()),
        "avg_line": float(dataframe["ah_line"].mean()),
        "avg_odds": float(dataframe["away_ah_odds"].mean()),
        "z_score": calculate_z_score(profits),
        "max_drawdown": calculate_max_drawdown(dataframe),
    }


def prepare_data(league_code, variant_name):
    config = VARIANTS[variant_name]

    input_path = get_league_matches_path(league_code)
    dataframe = pd.read_csv(input_path, low_memory=False)

    line_col = config["line_col"]
    home_col = config["home_col"]
    away_col = config["away_col"]

    required_columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "season_end_year",
        line_col,
        home_col,
        away_col,
    ]

    for column in required_columns:
        if column not in dataframe.columns:
            raise ValueError(f"Coluna em falta: {column}")

    dataframe = dataframe.copy()

    dataframe["ah_line"] = pd.to_numeric(dataframe[line_col], errors="coerce")
    dataframe["home_ah_odds"] = pd.to_numeric(dataframe[home_col], errors="coerce")
    dataframe["away_ah_odds"] = pd.to_numeric(dataframe[away_col], errors="coerce")

    dataframe = dataframe.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "season_end_year",
            "ah_line",
            "home_ah_odds",
            "away_ah_odds",
        ]
    ).copy()

    dataframe = dataframe[dataframe["home_ah_odds"] > 1.0].copy()
    dataframe = dataframe[dataframe["away_ah_odds"] > 1.0].copy()

    dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)

    dataframe["home_margin"] = dataframe["FTHG"].astype(float) - dataframe["FTAG"].astype(float)
    dataframe["away_margin"] = dataframe["FTAG"].astype(float) - dataframe["FTHG"].astype(float)

    dataframe["away_handicap"] = -dataframe["ah_line"]

    dataframe["profit"] = dataframe.apply(
        lambda row: asian_profit(
            row["away_margin"],
            row["away_handicap"],
            row["away_ah_odds"]
        ),
        axis=1
    )

    dataframe["home_raw_implied"] = 1.0 / dataframe["home_ah_odds"]
    dataframe["away_raw_implied"] = 1.0 / dataframe["away_ah_odds"]
    dataframe["overround"] = dataframe["home_raw_implied"] + dataframe["away_raw_implied"]

    dataframe["home_market_probability"] = dataframe["home_raw_implied"] / dataframe["overround"]
    dataframe["away_market_probability"] = dataframe["away_raw_implied"] / dataframe["overround"]

    dataframe["variant"] = variant_name
    dataframe["bet_side"] = "Away AH vs big home favourite"

    return dataframe


def evaluate_threshold_on_validation(validation_data, threshold):
    selected = validation_data[validation_data["ah_line"] <= threshold].copy()

    if len(selected) == 0:
        return None

    summary = summarize(selected)

    by_year = selected.groupby("season_end_year")["profit"].mean()

    positive_years = int((by_year > 0).sum())
    min_year_roi = float(by_year.min())

    result = {
        "threshold": threshold,
        "validation_bets": summary["bets"],
        "validation_profit": summary["profit"],
        "validation_roi": summary["roi"],
        "validation_z_score": summary["z_score"],
        "validation_max_drawdown": summary["max_drawdown"],
        "validation_positive_years": positive_years,
        "validation_min_year_roi": min_year_roi,
        "validation_avg_line": summary["avg_line"],
        "validation_avg_odds": summary["avg_odds"],
    }

    return result


def select_threshold(validation_data, min_validation_bets, min_positive_validation_years):
    candidates = []

    for threshold in THRESHOLDS:
        result = evaluate_threshold_on_validation(validation_data, threshold)

        if result is None:
            continue

        if result["validation_bets"] < min_validation_bets:
            continue

        if result["validation_roi"] <= 0.0:
            continue

        if result["validation_positive_years"] < min_positive_validation_years:
            continue

        if result["validation_min_year_roi"] <= 0.0:
            continue

        candidates.append(result)

    if len(candidates) == 0:
        return None, pd.DataFrame()

    candidates_dataframe = pd.DataFrame(candidates)

    candidates_dataframe = candidates_dataframe.sort_values(
        [
            "validation_positive_years",
            "validation_min_year_roi",
            "validation_roi",
            "validation_bets",
        ],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    candidates_dataframe["validation_rank"] = candidates_dataframe.index + 1

    selected = candidates_dataframe.iloc[0].to_dict()

    return selected, candidates_dataframe


def run_nested_baseline(
    dataframe,
    min_validation_years,
    min_validation_bets,
    min_positive_validation_years
):
    years = sorted(dataframe["season_end_year"].unique().tolist())

    by_year_rows = []
    all_bets = []
    all_candidates = []

    for test_year in years:
        validation_years = []

        for year in years:
            if year < test_year:
                validation_years.append(year)

        if len(validation_years) < min_validation_years:
            continue

        validation_data = dataframe[dataframe["season_end_year"].isin(validation_years)].copy()
        test_data = dataframe[dataframe["season_end_year"] == test_year].copy()

        selected, candidates = select_threshold(
            validation_data,
            min_validation_bets,
            min_positive_validation_years
        )

        if selected is None:
            print(f"{test_year}: sem threshold válido")
            continue

        threshold = float(selected["threshold"])

        selected_test = test_data[test_data["ah_line"] <= threshold].copy()
        test_summary = summarize(selected_test)

        row = {
            "test_year": test_year,
            "selected_threshold": threshold,
            "validation_years": ",".join([str(year) for year in validation_years]),
            "validation_bets": selected["validation_bets"],
            "validation_profit": selected["validation_profit"],
            "validation_roi": selected["validation_roi"],
            "validation_z_score": selected["validation_z_score"],
            "validation_max_drawdown": selected["validation_max_drawdown"],
            "validation_positive_years": selected["validation_positive_years"],
            "validation_min_year_roi": selected["validation_min_year_roi"],
            "test_bets": test_summary["bets"],
            "test_wins": test_summary["wins"],
            "test_half_wins": test_summary["half_wins"],
            "test_pushes": test_summary["pushes"],
            "test_half_losses": test_summary["half_losses"],
            "test_losses": test_summary["losses"],
            "test_profit": test_summary["profit"],
            "test_roi": test_summary["roi"],
            "test_z_score": test_summary["z_score"],
            "test_max_drawdown": test_summary["max_drawdown"],
            "test_avg_line": test_summary["avg_line"],
            "test_avg_odds": test_summary["avg_odds"],
        }

        by_year_rows.append(row)

        if len(selected_test) > 0:
            selected_test = selected_test.copy()
            selected_test["nested_test_year"] = test_year
            selected_test["selected_threshold"] = threshold
            all_bets.append(selected_test)

        if len(candidates) > 0:
            candidates = candidates.copy()
            candidates["test_year"] = test_year
            all_candidates.append(candidates)

    by_year_dataframe = pd.DataFrame(by_year_rows)

    if len(all_bets) > 0:
        bets_dataframe = pd.concat(all_bets, ignore_index=True)
    else:
        bets_dataframe = pd.DataFrame()

    if len(all_candidates) > 0:
        candidates_dataframe = pd.concat(all_candidates, ignore_index=True)
    else:
        candidates_dataframe = pd.DataFrame()

    return by_year_dataframe, bets_dataframe, candidates_dataframe


def print_results(variant_name, by_year_dataframe, bets_dataframe):
    print("")
    print("======================================")
    print("VARIANT:", variant_name)
    print("======================================")

    print("")
    print("=== NESTED BY YEAR ===")

    if len(by_year_dataframe) == 0:
        print("Sem resultados.")
    else:
        print(by_year_dataframe.to_string(index=False))

    print("")
    print("=== OVERALL ===")

    if len(bets_dataframe) == 0:
        print("Sem bets.")
        return

    overall = summarize(bets_dataframe)

    for key, value in overall.items():
        if isinstance(value, float):
            print(f"{key}: {round(value, 6)}")
        else:
            print(f"{key}: {value}")

    print("")
    print("=== BY YEAR ===")

    rows = []

    for year, group in bets_dataframe.groupby("season_end_year"):
        summary = summarize(group)

        rows.append({
            "year": int(year),
            "bets": summary["bets"],
            "profit": round(summary["profit"], 3),
            "roi": round(summary["roi"], 4),
            "z_score": round(summary["z_score"], 4),
            "max_drawdown": round(summary["max_drawdown"], 3),
            "avg_line": round(summary["avg_line"], 3),
            "avg_odds": round(summary["avg_odds"], 3),
        })

    print(pd.DataFrame(rows).to_string(index=False))


def save_results(league_code, variant_name, by_year_dataframe, bets_dataframe, candidates_dataframe):
    output_dir = get_market_output_dir(league_code, MARKET_NAME) / variant_name / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_year_path = output_dir / "nested_baseline_by_year.csv"
    bets_path = output_dir / "nested_baseline_bets.csv"
    candidates_path = output_dir / "nested_baseline_candidates.csv"

    by_year_dataframe.to_csv(by_year_path, index=False)
    bets_dataframe.to_csv(bets_path, index=False)
    candidates_dataframe.to_csv(candidates_path, index=False)

    print("")
    print("Ficheiros guardados:")
    print(by_year_path)
    print(bets_path)
    print(candidates_path)


def run_variant(args, variant_name):
    league_code = args.league.upper()

    dataframe = prepare_data(league_code, variant_name)

    print("")
    print("A correr variant:", variant_name)
    print("Jogos com AH:", len(dataframe))
    print("Épocas:", sorted(dataframe["season_end_year"].unique().tolist()))
    print("Avg overround:", round(float(dataframe["overround"].mean()), 4))

    by_year_dataframe, bets_dataframe, candidates_dataframe = run_nested_baseline(
        dataframe=dataframe,
        min_validation_years=args.min_validation_years,
        min_validation_bets=args.min_validation_bets,
        min_positive_validation_years=args.min_positive_validation_years
    )

    print_results(variant_name, by_year_dataframe, bets_dataframe)

    save_results(
        league_code,
        variant_name,
        by_year_dataframe,
        bets_dataframe,
        candidates_dataframe
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--variant", default="both", choices=["main", "closing", "both"])
    parser.add_argument("--min-validation-years", type=int, default=2)
    parser.add_argument("--min-validation-bets", type=int, default=50)
    parser.add_argument("--min-positive-validation-years", type=int, default=2)
    args = parser.parse_args()

    if args.variant == "both":
        run_variant(args, "main")
        run_variant(args, "closing")
    else:
        run_variant(args, args.variant)


if __name__ == "__main__":
    main()
