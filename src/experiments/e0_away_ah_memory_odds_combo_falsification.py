from pathlib import Path

import pandas as pd

from src.experiments.e0_away_ah_hopfield_memory_review import clv_summary
from src.experiments.e0_away_ah_hopfield_memory_review import fmt
from src.experiments.e0_away_ah_hopfield_memory_review import pct
from src.experiments.e0_away_ah_memory_odds_combo_review import add_knn_profit_memory_score
from src.experiments.e0_away_ah_memory_odds_combo_review import apply_filter
from src.experiments.e0_away_ah_memory_odds_combo_review import candidate_filters
from src.experiments.e0_away_ah_memory_odds_combo_review import evaluate_candidate
from src.experiments.e0_away_ah_memory_odds_combo_review import overall_row
from src.experiments.e0_away_ah_memory_odds_combo_review import run_nested_strategy
from src.experiments.e0_away_ah_memory_odds_combo_review import strategy_defs
from src.experiments.e0_away_ah_weather_internal_elo_review import concentration
from src.experiments.e0_away_ah_weather_internal_elo_review import prepare_data
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/e0_away_ah_memory_odds_combo_falsification.md")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/memory_odds_combo_falsification")
FOCUS = "away_odds_ge_1_85_plus_memory_knn_profit"
BASELINE = "away_odds_ge_1_85"
FIXED_THRESHOLDS = [-1.25, -1.50, -1.75]
MIN_SAMPLE_SIZE = 100


def bet_key(dataframe):
    return (
        dataframe["Date"].astype(str)
        + "|"
        + dataframe["HomeTeam"].astype(str)
        + "|"
        + dataframe["AwayTeam"].astype(str)
    )


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


def audit_row(label, dataframe):
    row = {"group": label}
    summary = summarize(dataframe)
    row.update(
        {
            "bets": summary["bets"],
            "profit": summary["profit"],
            "roi": summary["roi"],
            "z_score": summary["z_score"],
            "max_drawdown": summary["max_drawdown"],
        }
    )
    row.update(clv_summary(dataframe))
    row.update(concentration(dataframe))
    return row


def overlap_groups(away_bets, combo_bets):
    away = away_bets.copy()
    combo = combo_bets.copy()
    away["_bet_key"] = bet_key(away)
    combo["_bet_key"] = bet_key(combo)
    away_keys = set(away["_bet_key"])
    combo_keys = set(combo["_bet_key"])
    common = away[away["_bet_key"].isin(combo_keys)].copy()
    removed = away[~away["_bet_key"].isin(combo_keys)].copy()
    combo_only = combo[~combo["_bet_key"].isin(away_keys)].copy()
    return pd.DataFrame(
        [
            audit_row("common_to_away_and_combo", common),
            audit_row("removed_from_away_by_memory_or_threshold", removed),
            audit_row("kept_by_memory_filter", common),
            audit_row("combo_only_due_to_nested_threshold_difference", combo_only),
        ]
    )


def bucket_breakdowns(combo_bets):
    output = combo_bets.copy()
    output["ah_line_bucket"] = pd.to_numeric(output["ah_line"], errors="coerce").map(lambda value: f"{value:.2f}")
    output["away_odds_bucket"] = pd.cut(
        pd.to_numeric(output["away_ah_odds"], errors="coerce"),
        bins=[0.0, 1.85, 1.95, 2.05, 2.20, 99.0],
        labels=["<1.85", "1.85-1.95", "1.95-2.05", "2.05-2.20", ">2.20"],
        right=False,
    ).astype(str)
    specs = [
        ("ah_line_bucket", "ah_line_bucket"),
        ("away_odds_bucket", "away_odds_bucket"),
        ("home_team", "HomeTeam"),
        ("away_team", "AwayTeam"),
        ("season", "season_end_year"),
    ]
    frames = []
    for section, column in specs:
        for value, group in output.groupby(column, dropna=False):
            row = audit_row(str(value), group)
            row["breakdown"] = section
            frames.append(row)
    return pd.DataFrame(frames)


def selected_nested_frames(dataframe):
    strategies = strategy_defs()
    by_year_frames = []
    bet_frames = []
    candidate_frames = []
    for strategy_name in [BASELINE, FOCUS]:
        by_year, bets, candidates = run_nested_strategy(dataframe, strategy_name, strategies[strategy_name])
        by_year_frames.append(by_year)
        if len(bets):
            bet_frames.append(bets)
        if len(candidates):
            candidate_frames.append(candidates)
    return (
        pd.concat(by_year_frames, ignore_index=True),
        pd.concat(bet_frames, ignore_index=True),
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
    )


def fixed_threshold_audit(dataframe, combo_by_year):
    strategies = strategy_defs()
    combo_def = strategies[FOCUS]
    selected_filters = combo_by_year[["test_year", "selected_filter"]].dropna()
    rows = []
    for threshold in FIXED_THRESHOLDS:
        chunks = []
        for _, row in selected_filters.iterrows():
            if row["selected_filter"] == "no_valid_validation_candidate":
                continue
            test = dataframe[dataframe["season_end_year"] == int(row["test_year"])]
            chunks.append(apply_filter(test, threshold, row["selected_filter"], combo_def))
        selected = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        audit = audit_row(f"fixed_{threshold:.2f}", selected)
        audit["fixed_threshold"] = threshold
        rows.append(audit)
    return pd.DataFrame(rows)


def exclude_each_season(bets):
    rows = []
    seasons = sorted(bets["season_end_year"].dropna().unique())
    for strategy, group in bets.groupby("strategy", sort=False):
        for season in seasons:
            row = audit_row(strategy, group[group["season_end_year"] != season].copy())
            row["excluded_season"] = int(season)
            rows.append(row)
    return pd.DataFrame(rows)


def investigate_2022_no_candidate(dataframe):
    strategies = strategy_defs()
    combo_def = strategies[FOCUS]
    validation = dataframe[dataframe["season_end_year"].isin([2020, 2021])].copy()
    rows = []
    for ah_threshold in [-1.00, -1.25, -1.50, -1.75, -2.00]:
        for filter_name, filter_func in candidate_filters(validation, combo_def):
            result = evaluate_candidate(validation, ah_threshold, filter_name, filter_func)
            if result is None:
                continue
            row = result.copy()
            row["passes_min_bets_40"] = row["validation_bets"] >= 40
            row["passes_positive_roi"] = row["validation_roi"] > 0.0
            row["passes_two_positive_years"] = row["validation_positive_years"] >= 2
            row["passes_positive_min_year_roi"] = row["validation_min_year_roi"] > 0.0
            row["passes_all_gates"] = (
                row["passes_min_bets_40"]
                and row["passes_positive_roi"]
                and row["passes_two_positive_years"]
                and row["passes_positive_min_year_roi"]
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["passes_all_gates", "validation_roi", "validation_bets"], ascending=[False, False, False]
    )


def metric_comparison(overall):
    away = overall[overall["strategy"] == BASELINE].iloc[0]
    combo = overall[overall["strategy"] == FOCUS].iloc[0]
    rows = [
        {"metric": "z_score", "away_odds_ge_1_85": away["z_score"], "combo": combo["z_score"], "combo_beats": combo["z_score"] > away["z_score"]},
        {
            "metric": "max_drawdown",
            "away_odds_ge_1_85": away["max_drawdown"],
            "combo": combo["max_drawdown"],
            "combo_beats": combo["max_drawdown"] < away["max_drawdown"],
        },
        {
            "metric": "avg_clv_pp",
            "away_odds_ge_1_85": away["avg_clv_pp"],
            "combo": combo["avg_clv_pp"],
            "combo_beats": combo["avg_clv_pp"] >= away["avg_clv_pp"],
        },
        {
            "metric": "top3_home_bet_share",
            "away_odds_ge_1_85": away["top3_home_bet_share"],
            "combo": combo["top3_home_bet_share"],
            "combo_beats": combo["top3_home_bet_share"] < away["top3_home_bet_share"],
        },
        {
            "metric": "sample_size_adequacy",
            "away_odds_ge_1_85": away["bets"],
            "combo": combo["bets"],
            "combo_beats": combo["bets"] >= MIN_SAMPLE_SIZE,
        },
    ]
    return pd.DataFrame(rows)


def classify(overall, fixed, comparison):
    combo = overall[overall["strategy"] == FOCUS].iloc[0]
    positive_fixed = fixed[(fixed["bets"] >= 80) & (fixed["profit"] > 0.0) & (fixed["roi"] > 0.0)]
    required_metrics = comparison.set_index("metric")["combo_beats"]
    survives_fixed = len(positive_fixed) == len(FIXED_THRESHOLDS)
    concentration_ok = bool(required_metrics["top3_home_bet_share"])
    clv_ok = bool(required_metrics["avg_clv_pp"])
    robust_ok = bool(required_metrics["z_score"]) and bool(required_metrics["max_drawdown"])
    sample_ok = bool(required_metrics["sample_size_adequacy"])
    if combo["bets"] == 0 or combo["profit"] <= 0.0:
        return "reject"
    if survives_fixed and concentration_ok and clv_ok and robust_ok and sample_ok:
        return "paper challenger"
    return "research only"


def write_report(overall, overlap, breakdowns, fixed, dynamic_combo, no_2022, exclude, comparison, classification):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(DETAIL_DIR / "overall.csv", index=False)
    overlap.to_csv(DETAIL_DIR / "overlap.csv", index=False)
    breakdowns.to_csv(DETAIL_DIR / "breakdowns.csv", index=False)
    fixed.to_csv(DETAIL_DIR / "fixed_thresholds.csv", index=False)
    no_2022.to_csv(DETAIL_DIR / "no_2022_validation_candidate.csv", index=False)
    exclude.to_csv(DETAIL_DIR / "exclude_each_season.csv", index=False)
    comparison.to_csv(DETAIL_DIR / "combo_vs_away_odds_checks.csv", index=False)

    top_home = breakdowns[breakdowns["breakdown"] == "home_team"].sort_values("bets", ascending=False).head(20)
    top_away = breakdowns[breakdowns["breakdown"] == "away_team"].sort_values("bets", ascending=False).head(20)
    dynamic_fixed = pd.concat([dynamic_combo.assign(group="nested_dynamic"), fixed], ignore_index=True, sort=False)
    failed_reasons = no_2022[
        [
            "selected_threshold",
            "selected_filter",
            "validation_bets",
            "validation_roi",
            "validation_positive_years",
            "validation_min_year_roi",
            "passes_min_bets_40",
            "passes_positive_roi",
            "passes_two_positive_years",
            "passes_positive_min_year_roi",
            "passes_all_gates",
        ]
    ].head(20)

    lines = [
        "# E0 Away AH Memory + Odds Combo Falsification",
        "",
        "Scope: final falsification audit for `away_odds_ge_1_85_plus_memory_knn_profit` only. No raw data edits, no broad search, no new model family. Closing odds are diagnostic only for CLV.",
        "",
        "## Overall",
        "",
        markdown_table(
            overall,
            ["strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share", "home_hhi_bets", "away_hhi_bets"],
            ["Strategy", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate", "Top3 home", "Top3 away", "Home HHI", "Away HHI"],
        ),
        "",
        "## Overlap With Away Odds >= 1.85",
        "",
        markdown_table(
            overlap,
            ["group", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"],
            ["Group", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate"],
        ),
        "",
        "Note: `combo_only_due_to_nested_threshold_difference` appears because the combo and away-odds baseline select AH thresholds independently on prior validation seasons.",
        "",
        "## Combo Breakdowns",
        "",
        "### AH Line Bucket",
        "",
        markdown_table(
            breakdowns[breakdowns["breakdown"] == "ah_line_bucket"].sort_values("group"),
            ["group", "bets", "profit", "roi", "avg_clv_pp", "clv_positive_rate"],
            ["AH line", "Bets", "Profit", "ROI", "Avg CLV pp", "CLV + rate"],
        ),
        "",
        "### Away Odds Bucket",
        "",
        markdown_table(
            breakdowns[breakdowns["breakdown"] == "away_odds_bucket"].sort_values("group"),
            ["group", "bets", "profit", "roi", "avg_clv_pp", "clv_positive_rate"],
            ["Away odds", "Bets", "Profit", "ROI", "Avg CLV pp", "CLV + rate"],
        ),
        "",
        "### Season",
        "",
        markdown_table(
            breakdowns[breakdowns["breakdown"] == "season"].sort_values("group"),
            ["group", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp"],
            ["Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp"],
        ),
        "",
        "### Home Team Top 20 By Bets",
        "",
        markdown_table(top_home, ["group", "bets", "profit", "roi", "avg_clv_pp"], ["Home team", "Bets", "Profit", "ROI", "Avg CLV pp"]),
        "",
        "### Away Team Top 20 By Bets",
        "",
        markdown_table(top_away, ["group", "bets", "profit", "roi", "avg_clv_pp"], ["Away team", "Bets", "Profit", "ROI", "Avg CLV pp"]),
        "",
        "## Fixed AH Threshold Falsification",
        "",
        markdown_table(
            dynamic_fixed,
            ["group", "fixed_threshold", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"],
            ["Run", "Fixed AH threshold", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate"],
        ),
        "",
        "## Why 2022 Has No Combo Candidate",
        "",
        "For the 2022 test season, validation is limited to 2020 and 2021. No combo candidate passed all gates: minimum 40 bets, positive aggregate validation ROI, at least two positive validation years, and positive minimum validation-season ROI.",
        "",
        markdown_table(
            failed_reasons,
            [
                "selected_threshold",
                "selected_filter",
                "validation_bets",
                "validation_roi",
                "validation_positive_years",
                "validation_min_year_roi",
                "passes_min_bets_40",
                "passes_positive_roi",
                "passes_two_positive_years",
                "passes_positive_min_year_roi",
                "passes_all_gates",
            ],
            ["AH threshold", "Filter", "Val bets", "Val ROI", "Positive years", "Min year ROI", "Bets gate", "ROI gate", "Years gate", "Min year gate", "All gates"],
        ),
        "",
        "## Exclude Each Season",
        "",
        markdown_table(
            exclude,
            ["group", "excluded_season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"],
            ["Strategy", "Excluded season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV + rate"],
        ),
        "",
        "## Combo Beats Away Odds Checks",
        "",
        markdown_table(
            comparison,
            ["metric", "away_odds_ge_1_85", "combo", "combo_beats"],
            ["Metric", "Away odds >=1.85", "Combo", "Combo beats"],
        ),
        "",
        "## Final Classification",
        "",
        f"**{classification}**",
        "",
    ]
    if classification == "paper challenger":
        lines.append("Rationale: the combo survived fixed-threshold, concentration, CLV, robustness, and sample-size checks.")
    elif classification == "research only":
        lines.append("Rationale: the combo is profitable and improves z/max drawdown/top3 home concentration, but it fails the CLV preservation and sample-size checks versus away odds >= 1.85.")
    else:
        lines.append("Rationale: the combo fails basic profitability or validation evidence.")
    lines.append("")
    lines.append("Do not call this a confirmed edge.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataframe = add_knn_profit_memory_score(prepare_data())
    by_year, bets, _ = selected_nested_frames(dataframe)
    away_bets = bets[bets["strategy"] == BASELINE].copy()
    combo_bets = bets[bets["strategy"] == FOCUS].copy()
    overall = pd.DataFrame([overall_row(BASELINE, away_bets), overall_row(FOCUS, combo_bets)])
    overlap = overlap_groups(away_bets, combo_bets)
    breakdowns = bucket_breakdowns(combo_bets)
    combo_by_year = by_year[by_year["strategy"] == FOCUS].copy()
    fixed = fixed_threshold_audit(dataframe, combo_by_year)
    dynamic_combo = pd.DataFrame([audit_row("nested_dynamic", combo_bets)])
    no_2022 = investigate_2022_no_candidate(dataframe)
    exclude = exclude_each_season(bets[bets["strategy"].isin([BASELINE, FOCUS])].copy())
    comparison = metric_comparison(overall)
    classification = classify(overall, fixed, comparison)
    write_report(overall, overlap, breakdowns, fixed, dynamic_combo, no_2022, exclude, comparison, classification)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
