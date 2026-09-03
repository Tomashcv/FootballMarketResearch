from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.e0_away_ah_hopfield_memory_review import FEATURE_COLUMNS
from src.experiments.e0_away_ah_hopfield_memory_review import available_feature_columns
from src.experiments.e0_away_ah_hopfield_memory_review import clv_summary
from src.experiments.e0_away_ah_hopfield_memory_review import compute_memory_scores_for_year
from src.experiments.e0_away_ah_hopfield_memory_review import fit_memory_scaler
from src.experiments.e0_away_ah_hopfield_memory_review import fmt
from src.experiments.e0_away_ah_hopfield_memory_review import pct
from src.experiments.e0_away_ah_hopfield_memory_review import transform_with_scaler
from src.experiments.e0_away_ah_weather_internal_elo_review import concentration
from src.experiments.e0_away_ah_weather_internal_elo_review import prepare_data
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/e0_away_ah_memory_odds_combo_review.md")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/memory_odds_combo_review")

MIN_VALIDATION_YEARS = 2
MIN_VALIDATION_BETS = 40
MIN_POSITIVE_VALIDATION_YEARS = 2
MEMORY_COLUMN = "memory_score_knn_profit"
STANDARD_MEMORY_QUANTILES = [0.50, 0.60, 0.70, 0.80]
CONSERVATIVE_MEMORY_QUANTILES = [0.75, 0.80, 0.85, 0.90]


def add_knn_profit_memory_score(dataframe):
    output = dataframe.copy()
    output["memory_value_profit"] = pd.to_numeric(output["profit"], errors="coerce")
    output[MEMORY_COLUMN] = np.nan
    feature_columns = available_feature_columns(output)
    variant = {"name": "knn_profit", "method": "knn", "value_column": "memory_value_profit", "beta": None}
    for year in sorted(output["season_end_year"].unique()):
        scores, _ = compute_memory_scores_for_year(output, year, variant, feature_columns)
        output.loc[scores.index, MEMORY_COLUMN] = scores
    return output


def validate_memory_scaler_scope(dataframe):
    feature_columns = available_feature_columns(dataframe)
    rows = []
    for year in sorted(dataframe["season_end_year"].unique()):
        memory = dataframe[dataframe["season_end_year"] < year]
        if len(memory) == 0:
            continue
        scaler = fit_memory_scaler(memory, feature_columns)
        transformed = transform_with_scaler(dataframe[dataframe["season_end_year"] == year], scaler)
        rows.append(
            {
                "target_year": int(year),
                "fit_max_year": int(dataframe.loc[scaler["fit_index"], "season_end_year"].max()),
                "target_rows": int((dataframe["season_end_year"] == year).sum()),
                "transformed_rows": int(transformed.shape[0]),
                "feature_count": len(FEATURE_COLUMNS),
                "available_feature_count": len(feature_columns),
            }
        )
    return pd.DataFrame(rows)


def strategy_defs():
    return {
        "original_nested": {"kind": "base", "memory_quantiles": []},
        "away_odds_ge_1_85": {"kind": "odds", "memory_quantiles": []},
        "memory_knn_profit": {"kind": "memory", "memory_quantiles": STANDARD_MEMORY_QUANTILES},
        "away_odds_ge_1_85_plus_memory_knn_profit": {
            "kind": "odds_memory",
            "memory_quantiles": STANDARD_MEMORY_QUANTILES,
        },
        "away_odds_ge_1_85_plus_memory_knn_profit_conservative": {
            "kind": "odds_memory",
            "memory_quantiles": CONSERVATIVE_MEMORY_QUANTILES,
        },
    }


def candidate_filters(validation, strategy_def):
    kind = strategy_def["kind"]
    if kind == "base":
        return [("none", lambda df: pd.Series(True, index=df.index))]
    if kind == "odds":
        return [("away_odds_ge_1_85", lambda df: pd.to_numeric(df["away_ah_odds"], errors="coerce") >= 1.85)]

    scores = pd.to_numeric(validation[MEMORY_COLUMN], errors="coerce").dropna()
    if len(scores) == 0:
        return []
    thresholds = sorted(set(float(scores.quantile(q)) for q in strategy_def["memory_quantiles"]))
    filters = []
    for threshold in thresholds:
        if kind == "memory":
            name = f"{MEMORY_COLUMN}>={threshold:.6f}"
            filters.append((name, lambda df, threshold=threshold: pd.to_numeric(df[MEMORY_COLUMN], errors="coerce") >= threshold))
        else:
            name = f"away_odds_ge_1_85_and_{MEMORY_COLUMN}>={threshold:.6f}"
            filters.append(
                (
                    name,
                    lambda df, threshold=threshold: (pd.to_numeric(df["away_ah_odds"], errors="coerce") >= 1.85)
                    & (pd.to_numeric(df[MEMORY_COLUMN], errors="coerce") >= threshold),
                )
            )
    return filters


def evaluate_candidate(validation, ah_threshold, filter_name, filter_func):
    selected = validation[(pd.to_numeric(validation["ah_line"], errors="coerce") <= ah_threshold) & filter_func(validation)]
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


def selected_filter_mask(dataframe, filter_name):
    if filter_name == "none":
        return pd.Series(True, index=dataframe.index)
    if filter_name == "away_odds_ge_1_85":
        return pd.to_numeric(dataframe["away_ah_odds"], errors="coerce") >= 1.85
    if filter_name.startswith(f"{MEMORY_COLUMN}>="):
        threshold = float(filter_name.split(">=")[1])
        return pd.to_numeric(dataframe[MEMORY_COLUMN], errors="coerce") >= threshold
    if filter_name.startswith(f"away_odds_ge_1_85_and_{MEMORY_COLUMN}>="):
        threshold = float(filter_name.split(">=")[1])
        return (pd.to_numeric(dataframe["away_ah_odds"], errors="coerce") >= 1.85) & (
            pd.to_numeric(dataframe[MEMORY_COLUMN], errors="coerce") >= threshold
        )
    raise ValueError(f"Unknown selected filter: {filter_name}")


def apply_filter(dataframe, threshold, filter_name, strategy_def):
    mask = pd.to_numeric(dataframe["ah_line"], errors="coerce") <= threshold
    return dataframe[mask & selected_filter_mask(dataframe, filter_name)].copy()


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

        selected_test = apply_filter(test, float(selected["selected_threshold"]), selected["selected_filter"], strategy_def)
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
    return [dict(overall_row(strategy, group), season=int(season)) for season, group in dataframe.groupby("season_end_year")]


def exclude_2025_rows(bets):
    return [dict(overall_row(strategy, group[group["season_end_year"] != 2025]), excluded_season=2025) for strategy, group in bets.groupby("strategy", sort=False)]


def fixed_threshold_sensitivity(dataframe, strategies, by_year):
    rows = []
    for strategy_name, strategy_def in strategies.items():
        selected_filters = by_year[by_year["strategy"] == strategy_name][["test_year", "selected_filter"]].dropna()
        for threshold in THRESHOLDS:
            chunks = []
            for _, row in selected_filters.iterrows():
                if row["selected_filter"] == "no_valid_validation_candidate":
                    continue
                test = dataframe[dataframe["season_end_year"] == int(row["test_year"])]
                chunks.append(apply_filter(test, threshold, row["selected_filter"], strategy_def))
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


def final_decision(overall):
    away = overall[overall["strategy"] == "away_odds_ge_1_85"].iloc[0]
    combo_rows = overall[overall["strategy"].str.contains("plus_memory_knn_profit")].copy()
    viable = combo_rows[(combo_rows["bets"] >= 100) & (combo_rows["profit"] > 0.0) & (combo_rows["roi"] > 0.0)]
    if len(viable) == 0:
        positive = combo_rows[(combo_rows["bets"] > 0) & (combo_rows["profit"] > 0.0) & (combo_rows["roi"] > 0.0)]
        if len(positive):
            return "research only"
        return "reject"

    best = viable.sort_values(["z_score", "roi"], ascending=[False, False]).iloc[0]
    clv_ok = best["avg_clv_pp"] > 0.0 and best["avg_clv_pp"] >= away["avg_clv_pp"]
    robustness_ok = best["z_score"] >= away["z_score"] and best["max_drawdown"] <= away["max_drawdown"]
    concentration_ok = (
        best["top3_home_bet_share"] < away["top3_home_bet_share"]
        and best["top3_away_bet_share"] <= away["top3_away_bet_share"]
        and best["home_hhi_bets"] < away["home_hhi_bets"]
    )
    if clv_ok and robustness_ok and concentration_ok:
        return "live shadow candidate"
    if best["profit"] > 0.0 and best["avg_clv_pp"] > 0.0:
        return "paper trade only"
    return "research only"


def write_report(dataframe, strategies, by_year, bets, candidates, sensitivity, scaler_scope):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "nested_bets.csv", index=False)
    candidates.to_csv(DETAIL_DIR / "nested_candidates.csv", index=False)
    sensitivity.to_csv(DETAIL_DIR / "nearby_threshold_sensitivity.csv", index=False)
    scaler_scope.to_csv(DETAIL_DIR / "memory_scaler_scope.csv", index=False)

    overall_rows = []
    for strategy in strategies:
        group = bets[bets["strategy"] == strategy] if len(bets) else dataframe.iloc[0:0].copy()
        overall_rows.append(overall_row(strategy, group if len(group) else dataframe.iloc[0:0].copy()))
    overall = pd.DataFrame(overall_rows)
    seasonal = pd.DataFrame([row for strategy, group in bets.groupby("strategy", sort=False) for row in season_rows(strategy, group)])
    exclude_2025 = pd.DataFrame(exclude_2025_rows(bets))
    selected = by_year[["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"]]
    decision = final_decision(overall)

    lines = [
        "# E0 Away AH Memory + Odds Combo Review",
        "",
        "Scope: focused challenger combination review for E0 Away AH. No new model family is tested. Closing odds are used only after selection for CLV diagnostics.",
        "",
        "Memory signal: `memory_knn_profit`, computed from standardized local pre-match vectors using only prior seasons. Scalers are fitted only on prior seasons.",
        "",
        "Strategies compared: original nested, away odds >= 1.85, memory kNN profit, away odds >= 1.85 + memory kNN profit, and a conservative combo using higher prior-validation memory quantiles.",
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
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate", "Top3 home", "Top3 away", "Home HHI", "Away HHI"],
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
        "## Exclude 2025",
        "",
        markdown_table(
            exclude_2025,
            ["strategy", "excluded_season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share"],
            ["Strategy", "Excluded season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate", "Top3 home", "Top3 away"],
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
        "## Selected Prior-Validation Filters",
        "",
        markdown_table(
            selected,
            ["strategy", "test_year", "selected_threshold", "selected_filter", "validation_roi", "validation_min_year_roi"],
            ["Strategy", "Test season", "AH threshold", "Selected filter", "Validation ROI", "Min validation season ROI"],
        ),
        "",
        "## Leak Controls",
        "",
        "- Nested temporal validation only.",
        "- AH and memory thresholds are selected only on prior validation seasons.",
        "- `memory_knn_profit` memory rows use seasons before the scored season only.",
        "- Scaler scope checks are written to `outputs/E0/asian_handicap_big_home_favorite_away/memory_odds_combo_review/memory_scaler_scope.csv`.",
        "- Closing odds are not bet-time-safe features here; they are used only for CLV diagnostics after bet selection.",
        "",
        "## Final Decision",
        "",
        f"**{decision}**",
        "",
    ]
    if decision == "live shadow candidate":
        lines.append("Rationale: the combo improves CLV, robustness, and concentration versus away odds >= 1.85, but still needs prospective shadow tracking.")
    elif decision == "paper trade only":
        lines.append("Rationale: a combo is profitable with positive CLV, but it does not improve CLV, robustness, and concentration together.")
    elif decision == "research only":
        lines.append(
            "Rationale: combo results remain historically interesting, but they do not preserve the away-odds CLV level while improving robustness and concentration together."
        )
    else:
        lines.append("Rationale: combo variants do not provide enough nested out-of-sample evidence.")
    lines.append("")
    lines.append("Do not call this a confirmed edge.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataframe = add_knn_profit_memory_score(prepare_data())
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
    bets = pd.concat(bet_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    sensitivity = fixed_threshold_sensitivity(dataframe, strategies, by_year)
    scaler_scope = validate_memory_scaler_scope(dataframe)
    write_report(dataframe, strategies, by_year, bets, candidates, sensitivity, scaler_scope)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
