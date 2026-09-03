from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import get_league_matches_path
from src.features.contextual_features import build_contextual_features
from src.features.travel_features import build_travel_features
from src.features.weather_features import add_weather_features
from src.features.weather_features import add_weather_shock_features
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import asian_profit
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


LEAGUE = "I1"
LEAGUE_NAME = "Serie A"
REPORT_PATH = Path("outputs/reports/i1_away_ah_contextual_memory_review.md")
SUMMARY_PATH = Path("outputs/reports/i1_away_ah_contextual_memory_summary.csv")
DETAIL_DIR = Path("outputs/I1/asian_handicap_big_home_favorite_away/contextual_memory_review")

MIN_SEASON_START = 2020
MAX_SEASON_START = 2024
MIN_VALIDATION_YEARS = 2
MIN_VALIDATION_BETS = 40
MIN_POSITIVE_VALIDATION_YEARS = 2
MEMORY_COLUMN = "memory_score_knn_profit"
MEMORY_QUANTILES = [0.50, 0.60, 0.70, 0.80]
KNN_K = 25

MEMORY_FEATURE_COLUMNS = [
    "ah_line",
    "away_ah_odds",
    "away_market_probability",
    "avg_1x2_AvgH_no_vig_probability",
    "avg_1x2_AvgD_no_vig_probability",
    "avg_1x2_AvgA_no_vig_probability",
    "travel_distance_km",
    "away_rest_days",
    "rest_days_diff",
    "away_matches_last_7d",
    "away_matches_last_14d",
    "matches_last_14d_diff",
    "min_team_season_matches_before",
    "weather_temperature_c",
    "weather_precipitation_mm",
    "weather_wind_speed_kph",
    "away_temperature_shock_c",
    "away_precipitation_shock_mm",
    "away_wind_speed_shock_kph",
    "home_internal_elo_pre",
    "away_internal_elo_pre",
    "internal_elo_diff_home_minus_away",
    "internal_elo_home_win_prob",
    "market_home_prob_minus_internal_elo_prob",
    "market_away_prob_minus_internal_elo_prob",
]


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def pct(value):
    if value is None or pd.isna(value):
        return ""
    return f"{100.0 * float(value):.2f}%"


def load_coordinates():
    coordinates = pd.read_csv("data/external/stadiums/stadiums_with_gps_coordinates.csv")
    overrides = pd.read_csv("data/manual/team_stadium_overrides.csv")
    return coordinates, overrides


def load_weather():
    weather = pd.read_csv("data/external/weather/historical_match_weather.csv")
    normals = pd.read_csv("data/external/weather/monthly_climate_normals.csv")
    return weather, normals


def prepare_data():
    matches = pd.read_csv(get_league_matches_path(LEAGUE), low_memory=False)
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce").dt.normalize()
    matches = matches[
        (pd.to_numeric(matches["season_start_year"], errors="coerce") >= MIN_SEASON_START)
        & (pd.to_numeric(matches["season_start_year"], errors="coerce") <= MAX_SEASON_START)
    ].copy()

    dataframe = build_contextual_features(matches)
    coordinates, overrides = load_coordinates()
    dataframe = build_travel_features(dataframe, coordinates, overrides)
    weather, normals = load_weather()
    dataframe = add_weather_features(dataframe, weather)
    dataframe = add_weather_shock_features(dataframe, normals)

    required = [
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
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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

    close_home = 1.0 / pd.to_numeric(dataframe.get("AvgCAHH"), errors="coerce")
    close_away = 1.0 / pd.to_numeric(dataframe.get("AvgCAHA"), errors="coerce")
    dataframe["clv_probability_pp"] = ((close_away / (close_home + close_away)) - dataframe["away_market_probability"]) * 100.0
    dataframe["line_move_to_away"] = dataframe["ah_line"] - pd.to_numeric(dataframe.get("AHCh"), errors="coerce")
    return dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def available_memory_features(dataframe):
    return [column for column in MEMORY_FEATURE_COLUMNS if column in dataframe.columns]


def fit_memory_scaler(memory, feature_columns):
    features = memory[feature_columns].apply(pd.to_numeric, errors="coerce")
    means = features.mean()
    stds = features.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return {"feature_columns": feature_columns, "means": means, "stds": stds, "fit_index": memory.index.copy()}


def transform_with_scaler(dataframe, scaler):
    features = dataframe[scaler["feature_columns"]].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(scaler["means"])
    return ((features - scaler["means"]) / scaler["stds"]).fillna(0.0).to_numpy(dtype=float)


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return matrix / norms[:, None]


def add_knn_profit_memory(dataframe):
    output = dataframe.copy()
    output["memory_value_profit"] = pd.to_numeric(output["profit"], errors="coerce")
    output[MEMORY_COLUMN] = np.nan
    feature_columns = available_memory_features(output)

    for year in sorted(output["season_end_year"].unique()):
        memory = output[output["season_end_year"] < year].copy()
        query = output[output["season_end_year"] == year].copy()
        if len(memory) == 0 or len(query) == 0:
            continue
        scaler = fit_memory_scaler(memory, feature_columns)
        memory_matrix = normalize_rows(transform_with_scaler(memory, scaler))
        query_matrix = normalize_rows(transform_with_scaler(query, scaler))
        memory_values = pd.to_numeric(memory["memory_value_profit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        scores = []
        for query_vector in query_matrix:
            similarities = memory_matrix @ query_vector
            indices = np.argsort(similarities)[-min(KNN_K, len(memory_values)) :]
            scores.append(float(np.nanmean(memory_values[indices])) if len(indices) else np.nan)
        output.loc[query.index, MEMORY_COLUMN] = scores
    return output


def true_filter(dataframe):
    return pd.Series(True, index=dataframe.index)


def strategy_defs():
    return {
        "original_nested": {"kind": "static", "filters": [("none", true_filter)]},
        "away_odds_ge_1_85": {
            "kind": "static",
            "filters": [("away_odds_ge_1_85", lambda df: pd.to_numeric(df["away_ah_odds"], errors="coerce") >= 1.85)],
        },
        "travel_rest_context_filters": {
            "kind": "static",
            "filters": [
                (
                    f"travel_lt_{travel}_no_short_rest_elo_le_{elo}",
                    lambda df, travel=travel, elo=elo: (
                        (pd.to_numeric(df["travel_distance_km"], errors="coerce") < travel)
                        & ~(
                            (pd.to_numeric(df["away_rest_days"], errors="coerce") <= 4)
                            & ((pd.to_numeric(df["home_rest_days"], errors="coerce") - pd.to_numeric(df["away_rest_days"], errors="coerce")) >= 2)
                        )
                        & (pd.to_numeric(df["internal_elo_diff_home_minus_away"], errors="coerce") <= elo)
                    ),
                )
                for travel, elo in [(300, 150), (400, 200), (500, 250), (650, 300)]
            ],
        },
        "weather_climate_shock_filters": {
            "kind": "static",
            "filters": [
                (
                    f"wind_lt_{wind}_precip_lt_{precip}",
                    lambda df, wind=wind, precip=precip: (
                        df["has_weather"]
                        & (pd.to_numeric(df["weather_wind_speed_kph"], errors="coerce") < wind)
                        & (pd.to_numeric(df["weather_precipitation_mm"], errors="coerce") < precip)
                    ),
                )
                for wind, precip in [(35, 8), (40, 10)]
            ]
            + [
                (
                    f"temp_shock_abs_le_{temp}_wind_shock_le_{wind}",
                    lambda df, temp=temp, wind=wind: (
                        df["has_weather_shock"]
                        & (pd.to_numeric(df["away_temperature_shock_c"], errors="coerce").abs() <= temp)
                        & (pd.to_numeric(df["away_wind_speed_shock_kph"], errors="coerce") <= wind)
                    ),
                )
                for temp, wind in [(8, 22), (12, 30)]
            ],
        },
        "internal_pre_match_elo_disagreement": {
            "kind": "static",
            "filters": [
                (
                    f"home_elo_diff_le_{elo}_market_home_minus_elo_le_{gap}",
                    lambda df, elo=elo, gap=gap: (
                        (pd.to_numeric(df["internal_elo_diff_home_minus_away"], errors="coerce") <= elo)
                        & (pd.to_numeric(df["market_home_prob_minus_internal_elo_prob"], errors="coerce") <= gap)
                    ),
                )
                for elo, gap in [(150, 0.00), (200, 0.05), (250, 0.05), (300, 0.10)]
            ],
        },
        "memory_knn_profit": {"kind": "memory", "requires_odds": False},
        "away_odds_ge_1_85_plus_memory_knn_profit": {"kind": "memory", "requires_odds": True},
    }


def candidate_filters(validation, strategy_def):
    if strategy_def["kind"] == "static":
        return strategy_def["filters"]
    scores = pd.to_numeric(validation[MEMORY_COLUMN], errors="coerce").dropna()
    if len(scores) == 0:
        return []
    filters = []
    for threshold in sorted(set(float(scores.quantile(q)) for q in MEMORY_QUANTILES)):
        if strategy_def["requires_odds"]:
            filters.append(
                (
                    f"away_odds_ge_1_85_and_{MEMORY_COLUMN}>={threshold:.6f}",
                    lambda df, threshold=threshold: (pd.to_numeric(df["away_ah_odds"], errors="coerce") >= 1.85)
                    & (pd.to_numeric(df[MEMORY_COLUMN], errors="coerce") >= threshold),
                )
            )
        else:
            filters.append(
                (
                    f"{MEMORY_COLUMN}>={threshold:.6f}",
                    lambda df, threshold=threshold: pd.to_numeric(df[MEMORY_COLUMN], errors="coerce") >= threshold,
                )
            )
    return filters


def evaluate_candidate(validation, ah_threshold, filter_name, filter_func):
    selected = validation[(pd.to_numeric(validation["ah_line"], errors="coerce") <= ah_threshold) & filter_func(validation)].copy()
    if len(selected) == 0:
        return None
    summary = summarize(selected)
    by_year = selected.groupby("season_end_year")["profit"].mean()
    return {
        "selected_threshold": ah_threshold,
        "selected_filter": filter_name,
        "validation_bets": summary["bets"],
        "validation_profit": summary["profit"],
        "validation_roi": summary["roi"],
        "validation_z_score": summary["z_score"],
        "validation_positive_years": int((by_year > 0).sum()),
        "validation_min_year_roi": float(by_year.min()),
    }


def select_strategy(validation, strategy_def):
    candidates = []
    for ah_threshold in THRESHOLDS:
        for filter_name, filter_func in candidate_filters(validation, strategy_def):
            result = evaluate_candidate(validation, ah_threshold, filter_name, filter_func)
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
    frame = pd.DataFrame(candidates).sort_values(
        ["validation_positive_years", "validation_min_year_roi", "validation_z_score", "validation_roi", "validation_bets"],
        ascending=[False, False, False, False, False],
    )
    frame = frame.reset_index(drop=True)
    frame["validation_rank"] = frame.index + 1
    return frame.iloc[0].to_dict(), frame


def run_nested_strategy(dataframe, strategy_name, strategy_def):
    years = sorted(dataframe["season_end_year"].unique())
    by_year_rows = []
    bet_frames = []
    candidate_frames = []
    for test_year in years:
        validation_years = [year for year in years if year < test_year]
        if len(validation_years) < MIN_VALIDATION_YEARS:
            continue
        validation = dataframe[dataframe["season_end_year"].isin(validation_years)].copy()
        test = dataframe[dataframe["season_end_year"] == test_year].copy()
        selected, candidates = select_strategy(validation, strategy_def)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": strategy_name,
                    "test_year": int(test_year),
                    "selected_threshold": pd.NA,
                    "selected_filter": "no_valid_validation_candidate",
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "test_z_score": 0.0,
                    "test_max_drawdown": 0.0,
                }
            )
            continue
        filter_func = next(func for name, func in candidate_filters(validation, strategy_def) if name == selected["selected_filter"])
        selected_test = test[(pd.to_numeric(test["ah_line"], errors="coerce") <= float(selected["selected_threshold"])) & filter_func(test)].copy()
        summary = summarize(selected_test)
        by_year_rows.append(
            {
                "strategy": strategy_name,
                "test_year": int(test_year),
                "selected_threshold": selected["selected_threshold"],
                "selected_filter": selected["selected_filter"],
                "validation_bets": selected["validation_bets"],
                "validation_roi": selected["validation_roi"],
                "validation_min_year_roi": selected["validation_min_year_roi"],
                "test_bets": summary["bets"],
                "test_profit": summary["profit"],
                "test_roi": summary["roi"],
                "test_z_score": summary["z_score"],
                "test_max_drawdown": summary["max_drawdown"],
            }
        )
        if len(selected_test):
            selected_test = selected_test.copy()
            selected_test["strategy"] = strategy_name
            selected_test["nested_test_year"] = int(test_year)
            selected_test["selected_threshold"] = selected["selected_threshold"]
            selected_test["selected_filter"] = selected["selected_filter"]
            bet_frames.append(selected_test)
        if len(candidates):
            candidates = candidates.copy()
            candidates["strategy"] = strategy_name
            candidates["test_year"] = int(test_year)
            candidate_frames.append(candidates)
    return (
        pd.DataFrame(by_year_rows),
        pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame(),
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
    )


def clv_summary(dataframe):
    if len(dataframe) == 0:
        return {"avg_clv_pp": pd.NA, "clv_positive_rate": pd.NA, "avg_line_move_to_away": pd.NA}
    clv = pd.to_numeric(dataframe["clv_probability_pp"], errors="coerce")
    line_move = pd.to_numeric(dataframe["line_move_to_away"], errors="coerce")
    return {
        "avg_clv_pp": float(clv.mean()) if clv.notna().any() else pd.NA,
        "clv_positive_rate": float((clv > 0).mean()) if clv.notna().any() else pd.NA,
        "avg_line_move_to_away": float(line_move.mean()) if line_move.notna().any() else pd.NA,
    }


def concentration(dataframe):
    if len(dataframe) == 0:
        return {"top3_home_bet_share": pd.NA, "top3_away_bet_share": pd.NA, "home_hhi_bets": pd.NA, "away_hhi_bets": pd.NA}

    def top3(column):
        return float(dataframe[column].value_counts(normalize=True).head(3).sum())

    def hhi(column):
        shares = dataframe[column].value_counts(normalize=True)
        return float((shares * shares).sum())

    return {
        "top3_home_bet_share": top3("HomeTeam"),
        "top3_away_bet_share": top3("AwayTeam"),
        "home_hhi_bets": hhi("HomeTeam"),
        "away_hhi_bets": hhi("AwayTeam"),
    }


def overall_row(strategy, dataframe):
    summary = summarize(dataframe)
    row = {
        "strategy": strategy,
        "bets": summary["bets"],
        "profit": summary["profit"],
        "roi": summary["roi"],
        "z_score": summary["z_score"],
        "max_drawdown": summary["max_drawdown"],
    }
    row.update(clv_summary(dataframe))
    row.update(concentration(dataframe))
    return row


def seasonal_rows(strategy, dataframe):
    rows = []
    for season, group in dataframe.groupby("season_end_year"):
        row = overall_row(strategy, group)
        row["season"] = int(season)
        rows.append(row)
    return rows


def exclude_rows(strategy, dataframe, best_season):
    rows = []
    for season in sorted(dataframe["season_end_year"].unique()):
        row = overall_row(strategy, dataframe[dataframe["season_end_year"] != season])
        row["excluded_season"] = int(season)
        row["exclusion_reason"] = "exclude_each_season"
        rows.append(row)
    if not pd.isna(best_season):
        row = overall_row(strategy, dataframe[dataframe["season_end_year"] != best_season])
        row["excluded_season"] = int(best_season)
        row["exclusion_reason"] = "exclude_best_profit_season"
        rows.append(row)
    return rows


def selected_filter_func(strategy_def, filter_name, validation):
    for name, func in candidate_filters(validation, strategy_def):
        if name == filter_name:
            return func
    return None


def nearby_threshold_sensitivity(dataframe, by_year, strategies):
    rows = []
    for strategy_name, strategy_def in strategies.items():
        strategy_years = by_year[by_year["strategy"] == strategy_name]
        for threshold in THRESHOLDS:
            chunks = []
            for _, selected in strategy_years.iterrows():
                if selected["selected_filter"] == "no_valid_validation_candidate" or pd.isna(selected["selected_threshold"]):
                    continue
                test_year = int(selected["test_year"])
                validation = dataframe[dataframe["season_end_year"] < test_year]
                test = dataframe[dataframe["season_end_year"] == test_year]
                func = selected_filter_func(strategy_def, selected["selected_filter"], validation)
                if func is None:
                    continue
                chunks.append(test[(pd.to_numeric(test["ah_line"], errors="coerce") <= threshold) & func(test)].copy())
            selected_bets = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            summary = summarize(selected_bets)
            rows.append(
                {
                    "strategy": strategy_name,
                    "fixed_threshold": threshold,
                    "bets": summary["bets"],
                    "profit": summary["profit"],
                    "roi": summary["roi"],
                    "z_score": summary["z_score"],
                    "max_drawdown": summary["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def markdown_table(dataframe, columns, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in dataframe.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if column == "roi" or column.endswith("_rate") or column.endswith("_share"):
                values.append(pct(value))
            elif isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def classify(overall, exclude, seasonal):
    if len(overall) == 0:
        return "reject", "No strategy produced nested out-of-sample bets."
    profitable = overall[(overall["profit"] > 0.0) & (overall["roi"] > 0.0)].copy()
    if len(profitable) == 0:
        return "reject", "No controlled strategy produced positive nested out-of-sample profit."
    viable = profitable[profitable["bets"] >= 80].copy()
    if len(viable) == 0:
        best = profitable.sort_values(["z_score", "roi", "profit"], ascending=[False, False, False]).iloc[0]
        return "research only", f"{best['strategy']} is positive, but the nested sample is too small for promotion."
    best = viable.sort_values(["z_score", "roi", "profit"], ascending=[False, False, False]).iloc[0]
    best_exclude = exclude[(exclude["strategy"] == best["strategy"]) & (exclude["exclusion_reason"] == "exclude_each_season")]
    best_seasons = seasonal[seasonal["strategy"] == best["strategy"]]
    clv_ok = bool(best["avg_clv_pp"] > 0.0 and best["clv_positive_rate"] >= 0.50)
    robust_ok = bool(
        best["z_score"] >= 1.0
        and len(best_exclude)
        and (best_exclude["profit"] > 0.0).all()
        and (best_seasons["profit"] > 0.0).sum() >= 2
    )
    concentration_ok = bool(best["top3_home_bet_share"] <= 0.45 and best["top3_away_bet_share"] <= 0.45 and best["home_hhi_bets"] <= 0.18)
    if clv_ok and robust_ok and concentration_ok and best["z_score"] >= 2.0 and best["bets"] >= 150:
        return "confirmed edge", f"{best['strategy']} clears CLV, robustness, concentration, z-score, and sample gates."
    if clv_ok and robust_ok and concentration_ok:
        return "paper candidate", f"{best['strategy']} is positive and passes CLV/robustness/concentration gates, but does not clear the stricter confirmed-edge bar."
    return "research only", f"{best['strategy']} is historically positive, but CLV, robustness, and concentration do not all support promotion."


def write_outputs(dataframe, by_year, bets, candidates, sensitivity, strategies):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "nested_bets.csv", index=False)
    candidates.to_csv(DETAIL_DIR / "nested_candidates.csv", index=False)
    sensitivity.to_csv(DETAIL_DIR / "nearby_threshold_sensitivity.csv", index=False)

    grouped = []
    for strategy in strategies:
        if len(bets) and "strategy" in bets.columns:
            group = bets[bets["strategy"] == strategy].copy()
        else:
            group = pd.DataFrame(columns=dataframe.columns)
        grouped.append((strategy, group))
    overall = pd.DataFrame([overall_row(strategy, group) for strategy, group in grouped])
    seasonal = pd.DataFrame([row for strategy, group in grouped if len(group) for row in seasonal_rows(strategy, group)])
    best_season = bets.groupby("season_end_year")["profit"].sum().idxmax() if len(bets) else pd.NA
    exclude = pd.DataFrame([row for strategy, group in grouped if len(group) for row in exclude_rows(strategy, group, best_season)])
    classification, rationale = classify(overall, exclude, seasonal)
    overall["classification"] = classification
    overall.to_csv(SUMMARY_PATH, index=False)
    overall.to_csv(DETAIL_DIR / "overall.csv", index=False)
    seasonal.to_csv(DETAIL_DIR / "seasonal.csv", index=False)
    exclude.to_csv(DETAIL_DIR / "exclude_each_season.csv", index=False)

    selected = by_year[
        ["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"]
    ].copy()
    feature_columns = available_memory_features(dataframe)
    weather_rows = int(dataframe["has_weather"].sum()) if "has_weather" in dataframe.columns else 0
    shock_rows = int(dataframe["has_weather_shock"].sum()) if "has_weather_shock" in dataframe.columns else 0

    lines = [
        f"# {LEAGUE} Away AH Contextual Memory Review",
        "",
        f"Scope: controlled {LEAGUE_NAME} (`{LEAGUE}`) Away Asian Handicap review for seasons 2020-2021 through 2024-2025.",
        "",
        "Method: nested temporal validation only. AH thresholds and strategy filters are selected from prior validation seasons only. Memory kNN uses prior seasons only, and memory scalers are fit only on prior seasons. Main/open AH odds are used for selection; closing AH odds are used only for CLV diagnostics. External ClubElo is not used.",
        "",
        f"Input scope after bet-time-safe AH cleanup: {len(dataframe)} rows. Cached weather rows: {weather_rows}; climate-shock rows: {shock_rows}. Memory feature count: {len(feature_columns)}.",
        "",
        "## Overall Results",
        "",
        markdown_table(
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
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Top3 away", "Home HHI", "Away HHI"],
        ),
        "",
        "## Season By Season",
        "",
        markdown_table(
            seasonal,
            ["strategy", "season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"],
            ["Strategy", "Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate"],
        ),
        "",
        "## Exclude Each Season",
        "",
        markdown_table(
            exclude,
            ["strategy", "exclusion_reason", "excluded_season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp"],
            ["Strategy", "Reason", "Excluded season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp"],
        ),
        "",
        "## Nearby AH Threshold Sensitivity",
        "",
        markdown_table(
            sensitivity,
            ["strategy", "fixed_threshold", "bets", "profit", "roi", "z_score", "max_drawdown"],
            ["Strategy", "Fixed AH threshold", "Bets", "Profit", "ROI", "z", "Max DD"],
        ),
        "",
        "## Selected Filters",
        "",
        markdown_table(
            selected,
            ["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"],
            ["Strategy", "Test season", "AH threshold", "Prior-validation filter", "Validation ROI", "Min validation season ROI"],
        ),
        "",
        "## Controls",
        "",
        "- No broad model search was run; only the seven requested strategy families were evaluated.",
        "- Raw data was not edited.",
        "- Closing odds are absent from selection features and appear only in CLV diagnostics.",
        "- Weather/climate filters were evaluated against the local cache only; no API fetch or backfill was attempted.",
        f"- Internal Elo is computed chronologically from local processed {LEAGUE} results before each match update.",
        "",
        "## Final Classification",
        "",
        f"**{classification}**",
        "",
        f"Rationale: {rationale}",
        "",
    ]
    if classification != "confirmed edge":
        lines.append("Do not call this a confirmed edge.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataframe = add_knn_profit_memory(prepare_data())
    strategies = strategy_defs()
    by_year_frames = []
    bet_frames = []
    candidate_frames = []
    for strategy_name, strategy_def in strategies.items():
        by_year, bets, candidates = run_nested_strategy(dataframe, strategy_name, strategy_def)
        by_year_frames.append(by_year)
        if len(bets):
            bet_frames.append(bets)
        if len(candidates):
            candidate_frames.append(candidates)
    by_year = pd.concat(by_year_frames, ignore_index=True)
    bets = pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame()
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    sensitivity = nearby_threshold_sensitivity(dataframe, by_year, strategies)
    write_outputs(dataframe, by_year, bets, candidates, sensitivity, strategies.keys())
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
