import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.e0_away_ah_weather_internal_elo_review import concentration
from src.experiments.e0_away_ah_weather_internal_elo_review import prepare_data
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import calculate_max_drawdown
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import calculate_z_score
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/e0_away_ah_hopfield_memory_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_hopfield_memory_summary.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/hopfield_memory_review")

MIN_VALIDATION_YEARS = 2
MIN_VALIDATION_BETS = 40
MIN_POSITIVE_VALIDATION_YEARS = 2
MEMORY_QUANTILES = [0.50, 0.60, 0.70, 0.80]
KNN_K = 25

FEATURE_COLUMNS = [
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


def available_feature_columns(dataframe):
    return [column for column in FEATURE_COLUMNS if column in dataframe.columns]


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


def stable_softmax(values):
    if len(values) == 0:
        return np.array([])
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if total == 0.0 or not np.isfinite(total):
        return np.full(len(values), 1.0 / len(values))
    return exp_values / total


def retrieve_memory_value(memory_vectors, memory_values, query_vector, method, beta=1.0, k=KNN_K):
    if len(memory_values) == 0:
        return np.nan
    similarities = memory_vectors @ query_vector
    if method == "knn":
        count = min(int(k), len(memory_values))
        indices = np.argsort(similarities)[-count:]
        return float(np.nanmean(memory_values[indices])) if count else np.nan
    weights = stable_softmax(float(beta) * similarities)
    return float(np.sum(weights * memory_values))


def add_memory_value_columns(dataframe):
    output = dataframe.copy()
    output["memory_value_profit"] = pd.to_numeric(output["profit"], errors="coerce")
    output["memory_value_binary_cover"] = (pd.to_numeric(output["profit"], errors="coerce") > 0.0).astype(float)
    output["memory_value_market_residual"] = output["memory_value_binary_cover"] - pd.to_numeric(
        output["away_market_probability"], errors="coerce"
    )
    return output


def memory_variant_specs():
    specs = []
    value_columns = {
        "profit": "memory_value_profit",
        "market_residual": "memory_value_market_residual",
        "binary_cover": "memory_value_binary_cover",
    }
    for value_name, value_column in value_columns.items():
        specs.append({"name": f"knn_{value_name}", "method": "knn", "value_column": value_column, "beta": None})
        for beta in [1, 2, 5, 10]:
            specs.append(
                {
                    "name": f"hopfield_beta_{beta}_{value_name}",
                    "method": "hopfield",
                    "value_column": value_column,
                    "beta": beta,
                }
            )
    return specs


def compute_memory_scores_for_year(dataframe, target_year, variant, feature_columns):
    memory = dataframe[dataframe["season_end_year"] < target_year].copy()
    query = dataframe[dataframe["season_end_year"] == target_year].copy()
    scores = pd.Series(np.nan, index=query.index, dtype=float)
    if len(memory) == 0 or len(query) == 0:
        return scores, None

    scaler = fit_memory_scaler(memory, feature_columns)
    memory_matrix = normalize_rows(transform_with_scaler(memory, scaler))
    query_matrix = normalize_rows(transform_with_scaler(query, scaler))
    memory_values = pd.to_numeric(memory[variant["value_column"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    values = []
    for query_vector in query_matrix:
        values.append(
            retrieve_memory_value(
                memory_matrix,
                memory_values,
                query_vector,
                method=variant["method"],
                beta=variant.get("beta") or 1.0,
                k=KNN_K,
            )
        )
    scores.loc[query.index] = values
    return scores, scaler


def add_all_memory_scores(dataframe):
    output = add_memory_value_columns(dataframe)
    feature_columns = available_feature_columns(output)
    for variant in memory_variant_specs():
        column = f"memory_score_{variant['name']}"
        output[column] = np.nan
        for year in sorted(output["season_end_year"].unique()):
            scores, _ = compute_memory_scores_for_year(output, year, variant, feature_columns)
            output.loc[scores.index, column] = scores
    return output


def filter_none(dataframe):
    return pd.Series(True, index=dataframe.index)


def static_strategy_defs():
    return {
        "original_nested": {"kind": "static", "filters": [("none", filter_none)]},
        "away_odds_ge_1_85": {
            "kind": "static",
            "filters": [("away_odds_ge_1_85", lambda df: df["away_ah_odds"] >= 1.85)],
        },
        "climate_shock_features": {
            "kind": "static",
            "filters": [
                (
                    f"temp_shock_abs_le_{temp}_wind_shock_le_{wind}",
                    lambda df, temp=temp, wind=wind: (
                        df["has_weather_shock"]
                        & (pd.to_numeric(df["away_temperature_shock_c"], errors="coerce").abs() <= temp)
                        & (pd.to_numeric(df["away_wind_speed_shock_kph"], errors="coerce") <= wind)
                    ),
                )
                for temp, wind in [(6, 18), (8, 22), (10, 25), (12, 30)]
            ],
        },
        "market_internal_elo_disagreement": {
            "kind": "static",
            "filters": [
                (
                    f"market_home_minus_elo_le_{cutoff}",
                    lambda df, cutoff=cutoff: pd.to_numeric(df["market_home_prob_minus_internal_elo_prob"], errors="coerce")
                    <= cutoff,
                )
                for cutoff in [-0.05, 0.0, 0.05, 0.10]
            ],
        },
    }


def memory_strategy_defs():
    return {
        f"memory_{variant['name']}": {
            "kind": "memory",
            "score_column": f"memory_score_{variant['name']}",
        }
        for variant in memory_variant_specs()
    }


def all_strategy_defs():
    strategies = static_strategy_defs()
    strategies.update(memory_strategy_defs())
    return strategies


def apply_strategy(dataframe, threshold, strategy_def, selected_filter):
    mask = dataframe["ah_line"] <= threshold
    if strategy_def["kind"] == "static":
        func = next(func for name, func in strategy_def["filters"] if name == selected_filter)
        mask &= func(dataframe)
    else:
        score_threshold = float(selected_filter.split(">=")[1])
        mask &= pd.to_numeric(dataframe[strategy_def["score_column"]], errors="coerce") >= score_threshold
    return dataframe[mask].copy()


def candidate_filters(validation, strategy_def):
    if strategy_def["kind"] == "static":
        return strategy_def["filters"]
    scores = pd.to_numeric(validation[strategy_def["score_column"]], errors="coerce").dropna()
    if len(scores) == 0:
        return []
    thresholds = sorted(set(float(scores.quantile(q)) for q in MEMORY_QUANTILES))
    return [(f"{strategy_def['score_column']}>={threshold:.6f}", lambda df, threshold=threshold: df[strategy_def["score_column"]] >= threshold) for threshold in thresholds]


def evaluate_candidate(validation, ah_threshold, filter_name, filter_func):
    selected = validation[(validation["ah_line"] <= ah_threshold) & filter_func(validation)].copy()
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
        ["validation_positive_years", "validation_min_year_roi", "validation_roi", "validation_bets"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    frame["validation_rank"] = frame.index + 1
    return frame.iloc[0].to_dict(), frame


def run_nested_strategy(dataframe, strategy_name, strategy_def):
    years = sorted(dataframe["season_end_year"].unique())
    by_year_rows = []
    bets = []
    candidates = []
    for test_year in years:
        validation_years = [year for year in years if year < test_year]
        if len(validation_years) < MIN_VALIDATION_YEARS:
            continue
        validation = dataframe[dataframe["season_end_year"].isin(validation_years)].copy()
        test = dataframe[dataframe["season_end_year"] == test_year].copy()
        selected, candidate_frame = select_strategy(validation, strategy_def)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": strategy_name,
                    "test_year": test_year,
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
        selected_test = apply_strategy(test, float(selected["selected_threshold"]), strategy_def, selected["selected_filter"])
        summary = summarize(selected_test)
        by_year_rows.append(
            {
                "strategy": strategy_name,
                "test_year": test_year,
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
        if len(selected_test) > 0:
            selected_test = selected_test.copy()
            selected_test["strategy"] = strategy_name
            selected_test["nested_test_year"] = test_year
            selected_test["selected_threshold"] = selected["selected_threshold"]
            selected_test["selected_filter"] = selected["selected_filter"]
            bets.append(selected_test)
        if len(candidate_frame) > 0:
            candidate_frame = candidate_frame.copy()
            candidate_frame["strategy"] = strategy_name
            candidate_frame["test_year"] = test_year
            candidates.append(candidate_frame)
    return (
        pd.DataFrame(by_year_rows),
        pd.concat(bets, ignore_index=True) if bets else pd.DataFrame(),
        pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(),
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


def season_rows(strategy, dataframe):
    rows = []
    for season, group in dataframe.groupby("season_end_year"):
        row = overall_row(strategy, group)
        row["season"] = int(season)
        rows.append(row)
    return rows


def summarize_without_season(strategy, dataframe, season, reason):
    filtered = dataframe[dataframe["season_end_year"] != season].copy()
    row = overall_row(strategy, filtered)
    row["exclusion_reason"] = reason
    row["excluded_season"] = int(season)
    return row


def fixed_threshold_sensitivity(dataframe, strategy_name, strategy_def, by_year):
    rows = []
    selected_filters = by_year[by_year["strategy"] == strategy_name][["test_year", "selected_filter"]].dropna()
    for threshold in THRESHOLDS:
        chunks = []
        for _, row in selected_filters.iterrows():
            test = dataframe[dataframe["season_end_year"] == int(row["test_year"])]
            if len(test) == 0 or row["selected_filter"] == "no_valid_validation_candidate":
                continue
            chunks.append(apply_strategy(test, threshold, strategy_def, row["selected_filter"]))
        selected = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        summary = summarize(selected)
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


def pct(value):
    if value is None or pd.isna(value):
        return ""
    return f"{100.0 * float(value):.2f}%"


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def final_decision(overall):
    original = overall[overall["strategy"] == "original_nested"].iloc[0]
    memory = overall[overall["strategy"].str.startswith("memory_")].copy()
    viable = memory[(memory["bets"] >= 150) & (memory["profit"] > 0.0) & (memory["roi"] > 0.0)]
    if len(viable) == 0:
        return "reject"
    best = viable.sort_values(["z_score", "roi"], ascending=[False, False]).iloc[0]
    clv_ok = best["avg_clv_pp"] > original["avg_clv_pp"] and best["avg_clv_pp"] > 0.0
    robust_ok = best["z_score"] > original["z_score"] and best["max_drawdown"] <= original["max_drawdown"]
    if clv_ok and robust_ok:
        return "live shadow candidate"
    return "paper trade only"


def write_report(dataframe, by_year, bets, candidates, sensitivity):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "nested_bets.csv", index=False)
    candidates.to_csv(DETAIL_DIR / "nested_candidates.csv", index=False)
    sensitivity.to_csv(DETAIL_DIR / "nearby_threshold_sensitivity.csv", index=False)

    overall = pd.DataFrame([overall_row(strategy, group) for strategy, group in bets.groupby("strategy", sort=False)])
    seasonal = pd.DataFrame([row for strategy, group in bets.groupby("strategy", sort=False) for row in season_rows(strategy, group)])
    best_by_profit_season = int(bets.groupby("season_end_year")["profit"].sum().idxmax()) if len(bets) else 2025
    exclude_rows = []
    for strategy, group in bets.groupby("strategy", sort=False):
        exclude_rows.append(summarize_without_season(strategy, group, best_by_profit_season, "best_profit_season"))
        exclude_rows.append(summarize_without_season(strategy, group, 2025, "season_2025"))
    exclude = pd.DataFrame(exclude_rows)
    overall.to_csv(SUMMARY_PATH, index=False)
    seasonal.to_csv(DETAIL_DIR / "seasonal.csv", index=False)
    exclude.to_csv(DETAIL_DIR / "exclude_season.csv", index=False)

    memory_overall = overall[overall["strategy"].str.startswith("memory_")].copy()
    top_memory = memory_overall.sort_values(["z_score", "roi"], ascending=[False, False]).head(8)
    decision = final_decision(overall)

    selected = by_year[["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"]].copy()

    lines = [
        "# E0 Away AH Hopfield Memory Review",
        "",
        "Scope: controlled Hopfield-style temporal memory experiment for E0 Away AH. External ClubElo is not used. Closing odds are diagnostic only and not bet-time-safe features.",
        "",
        "Memory method: standardized local pre-match feature vectors from prior seasons only. Query rows retrieve weighted past values from earlier seasons. Hopfield variants use softmax(beta * similarity); kNN uses top-25 cosine-similar prior rows. Memory values tested: away AH profit, market residual, and binary cover.",
        "",
        f"Feature scope: {len(dataframe)} locally weather-covered E0 AH rows across seasons {int(dataframe['season_end_year'].min())}-{int(dataframe['season_end_year'].max())}.",
        "",
        "## Overall Results",
        "",
        markdown_table(
            overall,
            ["strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share"],
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate", "Top3 home share", "Top3 away share"],
        ),
        "",
        "## Top Memory Variants",
        "",
        markdown_table(
            top_memory,
            ["strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share"],
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate", "Top3 home share"],
        ),
        "",
        "## Season By Season",
        "",
        markdown_table(
            seasonal,
            ["strategy", "season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"],
            ["Strategy", "Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate"],
        ),
        "",
        "## Exclude Season Diagnostics",
        "",
        f"Best aggregate profit season in selected bets: {best_by_profit_season}.",
        "",
        markdown_table(
            exclude,
            [
                "strategy",
                "exclusion_reason",
                "excluded_season",
                "bets",
                "profit",
                "roi",
                "z_score",
                "max_drawdown",
                "avg_clv_pp",
            ],
            ["Strategy", "Reason", "Excluded season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp"],
        ),
        "",
        "## Nearby Threshold Sensitivity",
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
            ["Strategy", "Test season", "AH threshold", "Selected prior-validation filter", "Validation ROI", "Min validation season ROI"],
        ),
        "",
        "## Interpretation",
        "",
        "- The memory experiment is controlled to specified kNN/Hopfield variants, beta values, and value signals; it is not a broad model search.",
        "- Memory and scalers are fit only on seasons before the target season. Validation-season scores also use only seasons earlier than each validation season.",
        "- Thresholds are selected only from prior validation seasons inside the nested loop.",
        "- CLV uses closing odds only after bet selection for diagnostics.",
        "",
        "## Final Decision",
        "",
        f"**{decision}**",
        "",
    ]
    if decision == "reject":
        lines.append("Rationale: memory variants did not produce enough nested out-of-sample improvement in profitability, robustness, and CLV.")
    elif decision == "paper trade only":
        lines.append("Rationale: at least one memory variant is historically positive, but robustness and CLV do not clearly improve enough for live shadow or confirmation.")
    else:
        lines.append("Rationale: best memory variant improves robustness and CLV versus original, but this still requires prospective paper tracking.")
    lines.append("")
    lines.append("Do not call this a confirmed edge.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataframe = add_all_memory_scores(prepare_data())
    strategies = all_strategy_defs()
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
    bets = pd.concat(bet_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    sensitivity_targets = ["original_nested", "away_odds_ge_1_85"]
    memory_bets = bets[bets["strategy"].str.startswith("memory_")]
    if len(memory_bets):
        memory_summary = pd.DataFrame([overall_row(name, group) for name, group in memory_bets.groupby("strategy")])
        sensitivity_targets.append(memory_summary.sort_values(["z_score", "roi"], ascending=[False, False]).iloc[0]["strategy"])
    sensitivity = pd.concat(
        [fixed_threshold_sensitivity(dataframe, strategy, strategies[strategy], by_year) for strategy in sensitivity_targets],
        ignore_index=True,
    )
    write_report(dataframe, by_year, bets, candidates, sensitivity)
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
