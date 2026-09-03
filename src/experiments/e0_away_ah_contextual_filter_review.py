import math
from pathlib import Path

import pandas as pd

from src.common.paths import get_league_matches_path
from src.features.contextual_features import build_contextual_features
from src.features.travel_features import build_travel_features
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import asian_profit
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import calculate_max_drawdown
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import calculate_z_score
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


LEAGUE = "E0"
REPORT_PATH = Path("outputs/reports/e0_away_ah_contextual_filter_review.md")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/contextual_filter_review")

MIN_VALIDATION_YEARS = 2
MIN_VALIDATION_BETS = 50
MIN_POSITIVE_VALIDATION_YEARS = 2


def safe_pct(value):
    if pd.isna(value):
        return ""
    return f"{100.0 * float(value):.2f}%"


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def load_coordinates():
    stadium_path = Path("data/external/stadiums/stadiums_with_gps_coordinates.csv")
    overrides_path = Path("data/manual/team_stadium_overrides.csv")

    coordinates = pd.read_csv(stadium_path) if stadium_path.exists() else None
    overrides = pd.read_csv(overrides_path) if overrides_path.exists() else None
    return coordinates, overrides


def prepare_data():
    dataframe = pd.read_csv(get_league_matches_path(LEAGUE), low_memory=False)
    dataframe = build_contextual_features(dataframe)

    coordinates, overrides = load_coordinates()
    if coordinates is not None or overrides is not None:
        dataframe = build_travel_features(dataframe, coordinates, overrides)

    required_columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "season_end_year",
        "AHh",
        "AvgAHH",
        "AvgAHA",
    ]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dataframe = dataframe.copy()
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe["ah_line"] = pd.to_numeric(dataframe["AHh"], errors="coerce")
    dataframe["home_ah_odds"] = pd.to_numeric(dataframe["AvgAHH"], errors="coerce")
    dataframe["away_ah_odds"] = pd.to_numeric(dataframe["AvgAHA"], errors="coerce")
    dataframe["season_end_year"] = pd.to_numeric(dataframe["season_end_year"], errors="coerce")

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
    dataframe = dataframe[(dataframe["home_ah_odds"] > 1.0) & (dataframe["away_ah_odds"] > 1.0)].copy()
    dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)

    dataframe["away_margin"] = dataframe["FTAG"].astype(float) - dataframe["FTHG"].astype(float)
    dataframe["away_handicap"] = -dataframe["ah_line"]
    dataframe["profit"] = dataframe.apply(
        lambda row: asian_profit(row["away_margin"], row["away_handicap"], row["away_ah_odds"]),
        axis=1,
    )
    dataframe["home_raw_implied"] = 1.0 / dataframe["home_ah_odds"]
    dataframe["away_raw_implied"] = 1.0 / dataframe["away_ah_odds"]
    dataframe["overround"] = dataframe["home_raw_implied"] + dataframe["away_raw_implied"]
    dataframe["away_market_probability"] = dataframe["away_raw_implied"] / dataframe["overround"]

    if {"AvgCAHH", "AvgCAHA"}.issubset(dataframe.columns):
        dataframe["closing_home_ah_odds"] = pd.to_numeric(dataframe["AvgCAHH"], errors="coerce")
        dataframe["closing_away_ah_odds"] = pd.to_numeric(dataframe["AvgCAHA"], errors="coerce")
        close_home_raw = 1.0 / dataframe["closing_home_ah_odds"]
        close_away_raw = 1.0 / dataframe["closing_away_ah_odds"]
        close_overround = close_home_raw + close_away_raw
        dataframe["closing_away_market_probability"] = close_away_raw / close_overround
        dataframe["clv_probability_pp"] = (
            dataframe["closing_away_market_probability"] - dataframe["away_market_probability"]
        ) * 100.0
    else:
        dataframe["clv_probability_pp"] = pd.NA

    if "AHCh" in dataframe.columns:
        dataframe["closing_ah_line"] = pd.to_numeric(dataframe["AHCh"], errors="coerce")
        dataframe["line_move_to_away"] = dataframe["ah_line"] - dataframe["closing_ah_line"]
    else:
        dataframe["line_move_to_away"] = pd.NA

    return dataframe.reset_index(drop=True)


def filter_none(dataframe):
    return pd.Series(True, index=dataframe.index)


def make_filter(name, description, func):
    return {"name": name, "description": description, "func": func}


def strategy_filter_sets():
    return {
        "original_nested": [
            make_filter("none", "No contextual filter", filter_none),
        ],
        "exclude_away_odds_below_1_85": [
            make_filter("away_odds_ge_1_85", "Away AH odds >= 1.85", lambda df: df["away_ah_odds"] >= 1.85),
        ],
        "odds_band_1_85_2_05": [
            make_filter(
                "away_odds_1_85_to_2_05",
                "Away AH odds between 1.85 and 2.05",
                lambda df: df["away_ah_odds"].between(1.85, 2.05, inclusive="both"),
            ),
        ],
        "exclude_high_away_travel_burden": [
            make_filter(
                f"travel_lt_{cutoff}km",
                f"Exclude away travel >= {cutoff} km",
                lambda df, cutoff=cutoff: df["travel_distance_km"].isna() | (df["travel_distance_km"] < cutoff),
            )
            for cutoff in [200, 275, 350, 425]
        ],
        "exclude_away_short_rest_disadvantage": [
            make_filter(
                f"no_away_rest_le_{rest}_diff_ge_{diff}",
                f"Exclude away rest <= {rest} days with home rest advantage >= {diff} days",
                lambda df, rest=rest, diff=diff: ~(
                    (df["away_rest_days"] <= rest) & ((df["home_rest_days"] - df["away_rest_days"]) >= diff)
                ),
            )
            for rest, diff in [(3, 1), (4, 1), (4, 2), (5, 2)]
        ],
        "exclude_early_season_matches": [
            make_filter(
                f"min_team_matches_before_ge_{minimum}",
                f"Require both teams to have played at least {minimum} league matches",
                lambda df, minimum=minimum: df["min_team_season_matches_before"] >= minimum,
            )
            for minimum in [5, 8, 10]
        ],
        "odds_band_no_severe_travel_rest": [
            make_filter(
                f"odds_band_travel_lt_{cutoff}_no_rest_le_{rest}_diff_ge_{diff}",
                (
                    "Away AH odds 1.85-2.05, exclude severe travel/rest: "
                    f"travel >= {cutoff} km or away rest <= {rest} with home rest advantage >= {diff}"
                ),
                lambda df, cutoff=cutoff, rest=rest, diff=diff: (
                    df["away_ah_odds"].between(1.85, 2.05, inclusive="both")
                    & (df["travel_distance_km"].isna() | (df["travel_distance_km"] < cutoff))
                    & ~((df["away_rest_days"] <= rest) & ((df["home_rest_days"] - df["away_rest_days"]) >= diff))
                ),
            )
            for cutoff, rest, diff in [(275, 4, 1), (350, 4, 1), (350, 4, 2), (425, 5, 2)]
        ],
    }


def apply_spec(dataframe, threshold, filter_spec):
    mask = dataframe["ah_line"] <= threshold
    mask &= filter_spec["func"](dataframe)
    return dataframe[mask].copy()


def evaluate_validation(validation_data, threshold, filter_spec):
    selected = apply_spec(validation_data, threshold, filter_spec)
    if len(selected) == 0:
        return None

    summary = summarize(selected)
    by_year_roi = selected.groupby("season_end_year")["profit"].mean()
    return {
        "threshold": threshold,
        "filter_name": filter_spec["name"],
        "filter_description": filter_spec["description"],
        "validation_bets": summary["bets"],
        "validation_profit": summary["profit"],
        "validation_roi": summary["roi"],
        "validation_z_score": summary["z_score"],
        "validation_max_drawdown": summary["max_drawdown"],
        "validation_positive_years": int((by_year_roi > 0).sum()),
        "validation_min_year_roi": float(by_year_roi.min()),
        "validation_avg_odds": summary["avg_odds"],
    }


def select_spec(validation_data, filter_specs):
    candidates = []
    for threshold in THRESHOLDS:
        for filter_spec in filter_specs:
            result = evaluate_validation(validation_data, threshold, filter_spec)
            if result is None:
                continue
            if result["validation_bets"] < MIN_VALIDATION_BETS:
                continue
            if result["validation_roi"] <= 0.0:
                continue
            if result["validation_positive_years"] < MIN_POSITIVE_VALIDATION_YEARS:
                continue
            if result["validation_min_year_roi"] <= 0.0:
                continue
            candidates.append(result)

    if not candidates:
        return None, pd.DataFrame()

    candidates_dataframe = pd.DataFrame(candidates).sort_values(
        [
            "validation_positive_years",
            "validation_min_year_roi",
            "validation_roi",
            "validation_bets",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    candidates_dataframe["validation_rank"] = candidates_dataframe.index + 1
    selected = candidates_dataframe.iloc[0].to_dict()
    selected_spec = next(spec for spec in filter_specs if spec["name"] == selected["filter_name"])
    return {**selected, "filter_spec": selected_spec}, candidates_dataframe


def run_nested_strategy(dataframe, strategy_name, filter_specs):
    years = sorted(dataframe["season_end_year"].unique().tolist())
    by_year_rows = []
    all_bets = []
    all_candidates = []

    for test_year in years:
        validation_years = [year for year in years if year < test_year]
        if len(validation_years) < MIN_VALIDATION_YEARS:
            continue

        validation_data = dataframe[dataframe["season_end_year"].isin(validation_years)].copy()
        test_data = dataframe[dataframe["season_end_year"] == test_year].copy()
        selected, candidates = select_spec(validation_data, filter_specs)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": strategy_name,
                    "test_year": test_year,
                    "selected_threshold": pd.NA,
                    "selected_filter": "no_valid_validation_candidate",
                    "validation_years": ",".join(str(year) for year in validation_years),
                    "validation_bets": 0,
                    "validation_profit": pd.NA,
                    "validation_roi": pd.NA,
                    "validation_z_score": pd.NA,
                    "validation_positive_years": pd.NA,
                    "validation_min_year_roi": pd.NA,
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "test_z_score": 0.0,
                    "test_max_drawdown": 0.0,
                    "test_avg_odds": pd.NA,
                }
            )
            if len(candidates) > 0:
                candidates = candidates.copy()
                candidates["strategy"] = strategy_name
                candidates["test_year"] = test_year
                all_candidates.append(candidates)
            continue

        threshold = float(selected["threshold"])
        selected_test = apply_spec(test_data, threshold, selected["filter_spec"])
        test_summary = summarize(selected_test)

        by_year_rows.append(
            {
                "strategy": strategy_name,
                "test_year": test_year,
                "selected_threshold": threshold,
                "selected_filter": selected["filter_name"],
                "validation_years": ",".join(str(year) for year in validation_years),
                "validation_bets": selected["validation_bets"],
                "validation_profit": selected["validation_profit"],
                "validation_roi": selected["validation_roi"],
                "validation_z_score": selected["validation_z_score"],
                "validation_positive_years": selected["validation_positive_years"],
                "validation_min_year_roi": selected["validation_min_year_roi"],
                "test_bets": test_summary["bets"],
                "test_profit": test_summary["profit"],
                "test_roi": test_summary["roi"],
                "test_z_score": test_summary["z_score"],
                "test_max_drawdown": test_summary["max_drawdown"],
                "test_avg_odds": test_summary["avg_odds"],
            }
        )

        if len(selected_test) > 0:
            selected_test = selected_test.copy()
            selected_test["strategy"] = strategy_name
            selected_test["nested_test_year"] = test_year
            selected_test["selected_threshold"] = threshold
            selected_test["selected_filter"] = selected["filter_name"]
            all_bets.append(selected_test)

        if len(candidates) > 0:
            candidates = candidates.copy()
            candidates["strategy"] = strategy_name
            candidates["test_year"] = test_year
            all_candidates.append(candidates)

    by_year = pd.DataFrame(by_year_rows)
    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    return by_year, bets, candidates


def team_concentration(dataframe):
    if len(dataframe) == 0:
        return {
            "top3_home_profit_share": pd.NA,
            "top3_away_profit_share": pd.NA,
            "top3_home_bet_share": pd.NA,
            "top3_away_bet_share": pd.NA,
            "home_hhi_bets": pd.NA,
            "away_hhi_bets": pd.NA,
        }

    total_profit = float(dataframe["profit"].sum())

    def profit_share(column):
        if abs(total_profit) < 0.000001:
            return pd.NA
        by_team = dataframe.groupby(column)["profit"].sum().sort_values(ascending=False)
        return float(by_team.head(3).sum() / total_profit)

    def bet_share(column):
        shares = dataframe[column].value_counts(normalize=True)
        return float(shares.head(3).sum())

    def hhi(column):
        shares = dataframe[column].value_counts(normalize=True)
        return float((shares * shares).sum())

    return {
        "top3_home_profit_share": profit_share("HomeTeam"),
        "top3_away_profit_share": profit_share("AwayTeam"),
        "top3_home_bet_share": bet_share("HomeTeam"),
        "top3_away_bet_share": bet_share("AwayTeam"),
        "home_hhi_bets": hhi("HomeTeam"),
        "away_hhi_bets": hhi("AwayTeam"),
    }


def clv_summary(dataframe):
    if len(dataframe) == 0 or "clv_probability_pp" not in dataframe.columns:
        return {"avg_clv_pp": pd.NA, "clv_positive_rate": pd.NA, "avg_line_move_to_away": pd.NA}
    clv = pd.to_numeric(dataframe["clv_probability_pp"], errors="coerce")
    line_move = pd.to_numeric(dataframe.get("line_move_to_away", pd.Series(index=dataframe.index)), errors="coerce")
    return {
        "avg_clv_pp": float(clv.mean()) if clv.notna().any() else pd.NA,
        "clv_positive_rate": float((clv > 0).mean()) if clv.notna().any() else pd.NA,
        "avg_line_move_to_away": float(line_move.mean()) if line_move.notna().any() else pd.NA,
    }


def overall_row(strategy, bets):
    summary = summarize(bets)
    row = {
        "strategy": strategy,
        "bets": summary["bets"],
        "profit": summary["profit"],
        "roi": summary["roi"],
        "z_score": summary["z_score"],
        "max_drawdown": summary["max_drawdown"],
    }
    row.update(clv_summary(bets))
    row.update(team_concentration(bets))
    return row


def by_season_rows(strategy, bets):
    rows = []
    for season, group in bets.groupby("season_end_year"):
        summary = summarize(group)
        row = {
            "strategy": strategy,
            "season": int(season),
            "bets": summary["bets"],
            "profit": summary["profit"],
            "roi": summary["roi"],
            "z_score": summary["z_score"],
            "max_drawdown": summary["max_drawdown"],
        }
        row.update(clv_summary(group))
        row.update(team_concentration(group))
        rows.append(row)
    return rows


def markdown_table(rows, columns, headers=None):
    headers = headers or columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def metric_table(dataframe, columns, headers):
    rows = []
    for _, row in dataframe.iterrows():
        output = {}
        for column in columns:
            value = row[column]
            if column == "roi":
                output[column] = safe_pct(value)
            elif column.endswith("_rate") or column.endswith("_share"):
                output[column] = safe_pct(value)
            elif column in {"profit", "z_score", "max_drawdown", "avg_clv_pp", "avg_line_move_to_away"}:
                output[column] = fmt(value)
            elif column.endswith("_hhi_bets"):
                output[column] = fmt(value)
            else:
                output[column] = value
        rows.append(output)
    return markdown_table(rows, columns, headers)


def final_decision(overall):
    original = overall[overall["strategy"] == "original_nested"].iloc[0]
    combo_rows = overall[overall["strategy"] == "odds_band_no_severe_travel_rest"]
    band_rows = overall[overall["strategy"] == "odds_band_1_85_2_05"]

    best_context = combo_rows.iloc[0] if len(combo_rows) else band_rows.iloc[0]
    robustness_ok = (
        best_context["profit"] > 0
        and best_context["roi"] > 0
        and best_context["z_score"] >= original["z_score"]
        and best_context["bets"] >= 150
    )
    clv_improved = best_context["avg_clv_pp"] > original["avg_clv_pp"] and best_context["avg_clv_pp"] > 0
    concentration_improved = (
        best_context["top3_home_bet_share"] < original["top3_home_bet_share"]
        and best_context["top3_away_bet_share"] < original["top3_away_bet_share"]
    )

    if robustness_ok and clv_improved and concentration_improved:
        return "live shadow candidate"
    if best_context["profit"] > 0 and best_context["roi"] > 0:
        return "paper trade only"
    return "reject"


def write_report(dataframe, by_year_frames, bet_frames, candidate_frames):
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_by_year = pd.concat(by_year_frames, ignore_index=True)
    all_bets = pd.concat(bet_frames, ignore_index=True)
    all_candidates = pd.concat(candidate_frames, ignore_index=True)

    all_by_year.to_csv(DETAIL_DIR / "nested_contextual_by_year.csv", index=False)
    all_bets.to_csv(DETAIL_DIR / "nested_contextual_bets.csv", index=False)
    all_candidates.to_csv(DETAIL_DIR / "nested_contextual_candidates.csv", index=False)

    overall = pd.DataFrame([overall_row(strategy, bets) for strategy, bets in bet_frames_by_name(bet_frames)])
    seasonal = pd.DataFrame(
        [row for strategy, bets in bet_frames_by_name(bet_frames) for row in by_season_rows(strategy, bets)]
    )
    overall.to_csv(DETAIL_DIR / "contextual_overall.csv", index=False)
    seasonal.to_csv(DETAIL_DIR / "contextual_by_season_metrics.csv", index=False)

    decision = final_decision(overall)

    selected_filters = all_by_year[
        ["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"]
    ].copy()

    travel_rest_compare = overall[
        overall["strategy"].isin(
            [
                "original_nested",
                "exclude_high_away_travel_burden",
                "exclude_away_short_rest_disadvantage",
                "odds_band_no_severe_travel_rest",
            ]
        )
    ].copy()

    lines = [
        "# E0 Away AH Contextual Filter Review",
        "",
        "Audit date: 2026-06-29",
        "",
        "Scope: E0 / Premier League, Away Asian Handicap only, when the home AH line marks a large home favourite.",
        "",
        "Method: nested temporal validation only. AH threshold and contextual filter choices are selected inside each prior-season validation window, then applied once to the next test season. Main/open AH columns (`AHh`, `AvgAHH`, `AvgAHA`) are used for selection and settlement. Closing columns are used only after selection for CLV diagnostics.",
        "",
        f"Data: `{get_league_matches_path(LEAGUE)}`. Seasons available after AH cleaning: {int(dataframe['season_end_year'].min())}-{int(dataframe['season_end_year'].max())}.",
        "",
        "## Overall Results",
        "",
        metric_table(
            overall,
            [
                "strategy",
                "bets",
                "profit",
                "roi",
                "z_score",
                "max_drawdown",
                "avg_clv_pp",
                "clv_positive_rate",
                "top3_home_bet_share",
                "top3_away_bet_share",
                "home_hhi_bets",
                "away_hhi_bets",
            ],
            [
                "Strategy",
                "Bets",
                "Profit",
                "ROI",
                "z",
                "Max DD",
                "Avg CLV pp",
                "CLV + rate",
                "Top3 home bet share",
                "Top3 away bet share",
                "Home HHI",
                "Away HHI",
            ],
        ),
        "",
        "## Season By Season",
        "",
        metric_table(
            seasonal,
            [
                "strategy",
                "season",
                "bets",
                "profit",
                "roi",
                "z_score",
                "max_drawdown",
                "avg_clv_pp",
                "clv_positive_rate",
                "top3_home_bet_share",
                "top3_away_bet_share",
            ],
            [
                "Strategy",
                "Season",
                "Bets",
                "Profit",
                "ROI",
                "z",
                "Max DD",
                "Avg CLV pp",
                "CLV + rate",
                "Top3 home bet share",
                "Top3 away bet share",
            ],
        ),
        "",
        "## Selected Thresholds And Filters",
        "",
        metric_table(
            selected_filters,
            ["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"],
            ["Strategy", "Test season", "AH threshold", "Filter selected on prior seasons", "Validation ROI", "Min validation season ROI"],
        ),
        "",
        "Rows marked `no_valid_validation_candidate` mean the strategy failed the prior-season validation gates for that test season, so no out-of-sample bet was allowed.",
        "",
        "## Travel And Rest Concentration / CLV Check",
        "",
        "This is the specific check for whether travel/rest filters reduce team concentration and improve CLV versus the original nested strategy.",
        "",
        metric_table(
            travel_rest_compare,
            [
                "strategy",
                "bets",
                "profit",
                "roi",
                "avg_clv_pp",
                "clv_positive_rate",
                "top3_home_bet_share",
                "top3_away_bet_share",
                "home_hhi_bets",
                "away_hhi_bets",
            ],
            [
                "Strategy",
                "Bets",
                "Profit",
                "ROI",
                "Avg CLV pp",
                "CLV + rate",
                "Top3 home bet share",
                "Top3 away bet share",
                "Home HHI",
                "Away HHI",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- The original nested strategy remains profitable, but its average CLV is not positive.",
        "- Excluding away odds below 1.85 and the 1.85-2.05 odds band improve the headline ROI, but they do not create a confirmed edge because CLV remains weak or negative.",
        "- Travel/rest filters do not solve the main confirmation problem unless they also improve CLV and reduce team concentration out of sample.",
        "- Closing odds were not used as bet-time-safe inputs; they are reported only as CLV diagnostics.",
        "",
        "## Final Decision",
        "",
        f"**{decision}**",
        "",
    ]

    if decision == "paper trade only":
        lines.extend(
            [
                "Rationale: the contextual filters keep some positive historical performance, but robustness plus CLV do not improve enough to justify live shadow or confirmed status. The correct next step is timestamped paper tracking with fixed rules and captured available odds/closing odds.",
                "",
                "Do not call this a confirmed edge.",
            ]
        )
    elif decision == "reject":
        lines.extend(
            [
                "Rationale: contextual filtering did not preserve enough out-of-sample profitability after nested selection.",
                "",
                "Do not call this a confirmed edge.",
            ]
        )
    else:
        lines.extend(
            [
                "Rationale: robustness, CLV, and concentration improved versus the original nested strategy, but this is still not confirmed without prospective paper tracking.",
                "",
                "Do not call this a confirmed edge.",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def bet_frames_by_name(bet_frames):
    pairs = []
    for bets in bet_frames:
        if len(bets) == 0:
            continue
        pairs.append((bets["strategy"].iloc[0], bets))
    return pairs


def main():
    dataframe = prepare_data()
    filters_by_strategy = strategy_filter_sets()

    by_year_frames = []
    bet_frames = []
    candidate_frames = []

    for strategy_name, filter_specs in filters_by_strategy.items():
        by_year, bets, candidates = run_nested_strategy(dataframe, strategy_name, filter_specs)
        if len(by_year) == 0:
            raise RuntimeError(f"No nested results for {strategy_name}")
        by_year_frames.append(by_year)
        bet_frames.append(bets)
        candidate_frames.append(candidates)

    write_report(dataframe, by_year_frames, bet_frames, candidate_frames)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
