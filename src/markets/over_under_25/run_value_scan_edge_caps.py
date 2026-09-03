import argparse
import math

import pandas as pd

from src.common.paths import get_market_predictions_dir
from src.common.paths import get_market_value_scans_dir


MARKET_NAME = "over_under_25"

EDGE_MIN_VALUES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
EDGE_MAX_VALUES = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
EV_MIN_VALUES = [0.0, 0.005, 0.01, 0.015, 0.02]

ODDS_BANDS = {
    "short_1_40_1_70": (1.40, 1.70),
    "normal_1_70_1_90": (1.70, 1.90),
    "near_even_1_90_2_10": (1.90, 2.10),
    "plus_2_10_2_50": (2.10, 2.50),
    "dogs_2_50_3_50": (2.50, 3.50),
}

SIDE_FILTERS = [
    "both",
    "Over 2.5",
    "Under 2.5",
]


def calculate_profit(row):
    if int(row["result"]) == 1:
        return float(row["odds"]) - 1.0

    return -1.0


def calculate_z_score(profits):
    if len(profits) < 2:
        return 0.0

    mean_profit = profits.mean()
    standard_deviation = profits.std(ddof=1)

    if standard_deviation == 0:
        return 0.0

    return mean_profit / (standard_deviation / math.sqrt(len(profits)))


def calculate_max_drawdown(profits):
    cumulative_profit = profits.cumsum()
    running_max = cumulative_profit.cummax()
    drawdown = running_max - cumulative_profit

    if len(drawdown) == 0:
        return 0.0

    return float(drawdown.max())


def summarize_bets(dataframe):
    if len(dataframe) == 0:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": 0.0,
            "average_odds": 0.0,
            "average_model_probability": 0.0,
            "average_market_probability": 0.0,
            "average_edge": 0.0,
            "average_ev": 0.0,
            "roi": 0.0,
            "total_profit": 0.0,
            "z_score": 0.0,
            "max_drawdown": 0.0,
        }

    profits = dataframe["profit"].astype(float)

    wins = int(dataframe["result"].sum())
    bets = int(len(dataframe))
    total_profit = float(profits.sum())
    roi = total_profit / bets

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "average_odds": float(dataframe["odds"].mean()),
        "average_model_probability": float(dataframe["model_probability"].mean()),
        "average_market_probability": float(dataframe["market_probability"].mean()),
        "average_edge": float(dataframe["edge"].mean()),
        "average_ev": float(dataframe["ev"].mean()),
        "roi": roi,
        "total_profit": total_profit,
        "z_score": calculate_z_score(profits),
        "max_drawdown": calculate_max_drawdown(profits),
    }


def make_side_rows(predictions_dataframe):
    rows = []

    for _, row in predictions_dataframe.iterrows():
        over_row = row.to_dict()
        over_row["side"] = "Over 2.5"
        over_row["odds"] = float(row["over_25_odds"])
        over_row["model_probability"] = float(row["model_probability_over"])
        over_row["market_probability"] = float(row["market_probability_over"])
        over_row["result"] = int(row["target_over_25"])
        rows.append(over_row)

        under_row = row.to_dict()
        under_row["side"] = "Under 2.5"
        under_row["odds"] = float(row["under_25_odds"])
        under_row["model_probability"] = float(row["model_probability_under"])
        under_row["market_probability"] = float(row["market_probability_under"])
        under_row["result"] = 1 - int(row["target_over_25"])
        rows.append(under_row)

    side_dataframe = pd.DataFrame(rows)

    side_dataframe["edge"] = (
        side_dataframe["model_probability"].astype(float)
        - side_dataframe["market_probability"].astype(float)
    )

    side_dataframe["ev"] = (
        side_dataframe["model_probability"].astype(float)
        * side_dataframe["odds"].astype(float)
        - 1.0
    )

    side_dataframe["profit"] = side_dataframe.apply(calculate_profit, axis=1)

    return side_dataframe


def apply_rule(dataframe, rule):
    filtered = dataframe.copy()

    if rule["side"] != "both":
        filtered = filtered[filtered["side"] == rule["side"]].copy()

    filtered = filtered[filtered["odds"] >= rule["odds_min"]].copy()
    filtered = filtered[filtered["odds"] < rule["odds_max"]].copy()

    filtered = filtered[filtered["edge"] >= rule["edge_min"]].copy()
    filtered = filtered[filtered["edge"] < rule["edge_max"]].copy()

    filtered = filtered[filtered["ev"] >= rule["ev_min"]].copy()

    return filtered


def build_rules():
    rules = []

    for odds_band_name in ODDS_BANDS:
        odds_min, odds_max = ODDS_BANDS[odds_band_name]

        for side in SIDE_FILTERS:
            for edge_min in EDGE_MIN_VALUES:
                for edge_max in EDGE_MAX_VALUES:
                    if edge_max <= edge_min:
                        continue

                    for ev_min in EV_MIN_VALUES:
                        rule = {
                            "odds_band_name": odds_band_name,
                            "odds_min": odds_min,
                            "odds_max": odds_max,
                            "side": side,
                            "edge_min": edge_min,
                            "edge_max": edge_max,
                            "ev_min": ev_min,
                        }

                        rules.append(rule)

    return rules


def evaluate_rule_by_year(side_dataframe, rule):
    selected = apply_rule(side_dataframe, rule)

    rows = []

    years = sorted(side_dataframe["season_end_year"].dropna().astype(int).unique().tolist())

    for year in years:
        year_dataframe = selected[selected["season_end_year"].astype(int) == year].copy()
        summary = summarize_bets(year_dataframe)
        summary["year"] = year
        rows.append(summary)

    by_year = pd.DataFrame(rows)

    overall_summary = summarize_bets(selected)

    if len(by_year) > 0:
        positive_years = int((by_year["roi"] > 0).sum())
        year_rows_with_bets = by_year[by_year["bets"] > 0].copy()

        if len(year_rows_with_bets) > 0:
            min_year_roi = float(year_rows_with_bets["roi"].min())
        else:
            min_year_roi = 0.0
    else:
        positive_years = 0
        min_year_roi = 0.0

    overall_summary["positive_years"] = positive_years
    overall_summary["min_year_roi"] = min_year_roi

    return overall_summary, by_year, selected


def evaluate_rules_on_validation(validation_dataframe, rules, min_validation_bets):
    candidate_rows = []

    for rule in rules:
        selected = apply_rule(validation_dataframe, rule)

        if len(selected) < min_validation_bets:
            continue

        summary = summarize_bets(selected)

        by_year_rows = []

        years = sorted(validation_dataframe["season_end_year"].dropna().astype(int).unique().tolist())

        for year in years:
            year_selected = selected[selected["season_end_year"].astype(int) == year].copy()
            year_summary = summarize_bets(year_selected)
            year_summary["year"] = year
            by_year_rows.append(year_summary)

        by_year = pd.DataFrame(by_year_rows)
        by_year_with_bets = by_year[by_year["bets"] > 0].copy()

        if len(by_year_with_bets) > 0:
            positive_years = int((by_year_with_bets["roi"] > 0).sum())
            min_year_roi = float(by_year_with_bets["roi"].min())
        else:
            positive_years = 0
            min_year_roi = 0.0

        # Filtro conservador:
        # não aceitamos regras que só são "menos más".
        # A regra tem de ter ROI positivo e z_score positivo na validação.
        if summary["roi"] <= 0.0:
            continue

        if summary["z_score"] <= 0.0:
            continue

        if positive_years < 2:
            continue

        rule_summary = dict(rule)
        rule_summary.update({
            "validation_bets": summary["bets"],
            "validation_wins": summary["wins"],
            "validation_win_rate": summary["win_rate"],
            "validation_average_odds": summary["average_odds"],
            "validation_average_edge": summary["average_edge"],
            "validation_average_ev": summary["average_ev"],
            "validation_roi": summary["roi"],
            "validation_total_profit": summary["total_profit"],
            "validation_z_score": summary["z_score"],
            "validation_max_drawdown": summary["max_drawdown"],
            "validation_positive_years": positive_years,
            "validation_min_year_roi": min_year_roi,
        })

        candidate_rows.append(rule_summary)

    if len(candidate_rows) == 0:
        return pd.DataFrame()

    candidates = pd.DataFrame(candidate_rows)

    candidates = candidates.sort_values(
        [
            "validation_positive_years",
            "validation_min_year_roi",
            "validation_z_score",
            "validation_roi",
            "validation_bets",
        ],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    candidates["validation_rank"] = candidates.index + 1

    return candidates


def run_nested_value_scan(side_dataframe, min_validation_bets, min_validation_years):
    rules = build_rules()
    years = sorted(side_dataframe["season_end_year"].dropna().astype(int).unique().tolist())

    nested_rows = []
    nested_bets = []
    all_candidates = []

    for test_year in years:
        validation_years = [year for year in years if year < test_year]

        if len(validation_years) < min_validation_years:
            continue

        validation_dataframe = side_dataframe[
            side_dataframe["season_end_year"].astype(int).isin(validation_years)
        ].copy()

        test_dataframe = side_dataframe[
            side_dataframe["season_end_year"].astype(int) == test_year
        ].copy()

        candidates = evaluate_rules_on_validation(
            validation_dataframe,
            rules,
            min_validation_bets
        )

        if len(candidates) == 0:
            print(f"{test_year}: sem regra com bets suficientes.")
            continue

        selected_rule = candidates.iloc[0].to_dict()

        test_selected = apply_rule(test_dataframe, selected_rule)
        test_summary = summarize_bets(test_selected)

        row = {
            "test_year": test_year,
            "validation_years": ",".join([str(year) for year in validation_years]),
            "selected_odds_band_name": selected_rule["odds_band_name"],
            "selected_side": selected_rule["side"],
            "selected_edge_min": selected_rule["edge_min"],
            "selected_edge_max": selected_rule["edge_max"],
            "selected_ev_min": selected_rule["ev_min"],
            "validation_bets": selected_rule["validation_bets"],
            "validation_roi": selected_rule["validation_roi"],
            "validation_z_score": selected_rule["validation_z_score"],
            "validation_positive_years": selected_rule["validation_positive_years"],
            "validation_min_year_roi": selected_rule["validation_min_year_roi"],
        }

        for key in test_summary:
            row["test_" + key] = test_summary[key]

        nested_rows.append(row)

        if len(test_selected) > 0:
            test_selected = test_selected.copy()
            test_selected["nested_test_year"] = test_year
            nested_bets.append(test_selected)

        candidates = candidates.copy()
        candidates["test_year"] = test_year
        all_candidates.append(candidates)

        print("")
        print(f"=== VALUE SCAN BEFORE TEST YEAR {test_year} ===")
        print("Selected rule:")
        print("band:", selected_rule["odds_band_name"])
        print("side:", selected_rule["side"])
        print("edge:", selected_rule["edge_min"], "<= edge <", selected_rule["edge_max"])
        print("EV >=", selected_rule["ev_min"])
        print("validation bets:", selected_rule["validation_bets"])
        print("validation ROI:", round(float(selected_rule["validation_roi"]), 4))
        print("test bets:", test_summary["bets"])
        print("test ROI:", round(float(test_summary["roi"]), 4))
        print("test profit:", round(float(test_summary["total_profit"]), 2))
        print("test z:", round(float(test_summary["z_score"]), 4))

    nested_by_year = pd.DataFrame(nested_rows)

    if len(nested_bets) > 0:
        nested_bets_dataframe = pd.concat(nested_bets, ignore_index=True)
    else:
        nested_bets_dataframe = pd.DataFrame()

    if len(all_candidates) > 0:
        candidates_dataframe = pd.concat(all_candidates, ignore_index=True)
    else:
        candidates_dataframe = pd.DataFrame()

    return nested_by_year, nested_bets_dataframe, candidates_dataframe


def print_nested_overall(nested_bets_dataframe):
    print("")
    print("=== NESTED VALUE SCAN OVERALL ===")

    if len(nested_bets_dataframe) == 0:
        print("Sem bets.")
        return

    summary = summarize_bets(nested_bets_dataframe)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {round(value, 6)}")
        else:
            print(f"{key}: {value}")

    print("")
    print("By side:")
    for side in sorted(nested_bets_dataframe["side"].unique().tolist()):
        side_dataframe = nested_bets_dataframe[nested_bets_dataframe["side"] == side].copy()
        side_summary = summarize_bets(side_dataframe)

        print("")
        print("---", side, "---")
        for key, value in side_summary.items():
            if isinstance(value, float):
                print(f"{key}: {round(value, 6)}")
            else:
                print(f"{key}: {value}")

    print("")
    print("By odds band:")
    for odds_band_name in ODDS_BANDS:
        odds_min, odds_max = ODDS_BANDS[odds_band_name]
        band_dataframe = nested_bets_dataframe[
            (nested_bets_dataframe["odds"] >= odds_min)
            & (nested_bets_dataframe["odds"] < odds_max)
        ].copy()

        if len(band_dataframe) == 0:
            continue

        band_summary = summarize_bets(band_dataframe)
        print("")
        print("---", odds_band_name, "---")
        for key, value in band_summary.items():
            if isinstance(value, float):
                print(f"{key}: {round(value, 6)}")
            else:
                print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--prediction-file", default="xgboost_market_margin_predictions.csv")
    parser.add_argument("--min-validation-bets", type=int, default=40)
    parser.add_argument("--min-validation-years", type=int, default=2)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = MARKET_NAME

    predictions_dir = get_market_predictions_dir(league_code, market_name)
    value_scans_dir = get_market_value_scans_dir(league_code, market_name)

    input_path = predictions_dir / args.prediction_file

    if not input_path.exists():
        raise FileNotFoundError(f"Não encontrei predictions: {input_path}")

    predictions_dataframe = pd.read_csv(input_path, low_memory=False)

    required_columns = [
        "season_end_year",
        "target_over_25",
        "over_25_odds",
        "under_25_odds",
        "market_probability_over",
        "market_probability_under",
        "model_probability_over",
        "model_probability_under",
    ]

    for column in required_columns:
        if column not in predictions_dataframe.columns:
            raise ValueError(f"Coluna em falta nas predictions: {column}")

    side_dataframe = make_side_rows(predictions_dataframe)

    print("A usar:", input_path)
    print("Matches:", len(predictions_dataframe))
    print("Side rows:", len(side_dataframe))
    print("Min validation bets:", args.min_validation_bets)
    print("Min validation years:", args.min_validation_years)

    nested_by_year, nested_bets, candidates = run_nested_value_scan(
        side_dataframe,
        min_validation_bets=args.min_validation_bets,
        min_validation_years=args.min_validation_years
    )

    value_scans_dir.mkdir(parents=True, exist_ok=True)

    by_year_path = value_scans_dir / "value_scan_by_year.csv"
    bets_path = value_scans_dir / "value_scan_bets.csv"
    candidates_path = value_scans_dir / "value_scan_candidates.csv"

    nested_by_year.to_csv(by_year_path, index=False)
    nested_bets.to_csv(bets_path, index=False)
    candidates.to_csv(candidates_path, index=False)

    print_nested_overall(nested_bets)

    print("")
    print("Ficheiros guardados:")
    print(by_year_path)
    print(bets_path)
    print(candidates_path)


if __name__ == "__main__":
    main()
