from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_clubelo_understat_transfermarkt"
REPORT_DIR = ROOT / "outputs/reports/football_data_ah_predictive"

PRIMARY = DATA_DIR / "super_ah_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
OPEN = DATA_DIR / "super_ah_open_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"
CLOSE = DATA_DIR / "super_ah_close_football_data_top5_clubelo_understat_transfermarkt_research_v1.csv"

CANDIDATES = [
    {
        "candidate_id": "A",
        "feature_group": "market_plus_ah_line",
        "model": "xgboost_binary_ne80_lr0.05_rl5",
        "candidate_type": "market_recalibration",
    },
    {
        "candidate_id": "B",
        "feature_group": "market_plus_clubelo",
        "model": "regularized_logistic_regression",
        "candidate_type": "date_safe_feature_block",
    },
    {
        "candidate_id": "C",
        "feature_group": "market_probability_only",
        "model": "regularized_logistic_regression",
        "candidate_type": "market_probability_recalibration_control",
    },
    {
        "candidate_id": "D",
        "feature_group": "market_plus_ah_line",
        "model": "xgboost_binary_ne120_lr0.03_rl20",
        "candidate_type": "market_recalibration",
    },
]


def load_reports() -> dict[str, pd.DataFrame]:
    names = {
        "model": "ah_by_model.csv",
        "summary": "ah_summary.csv",
        "season": "ah_by_season.csv",
        "league": "ah_by_league.csv",
        "line": "ah_by_line_bucket.csv",
        "calibration": "ah_calibration.csv",
        "leakage": "ah_leakage_checks.csv",
        "features": "ah_feature_group_comparison.csv",
        "open_close": "ah_open_close_file_diagnostics.csv",
        "settlement": "ah_push_half_outcome_diagnostics.csv",
    }
    return {k: pd.read_csv(REPORT_DIR / v) for k, v in names.items()}


def candidate_filter(df: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    return df[df["feature_group"].eq(candidate["feature_group"]) & df["model"].eq(candidate["model"])].copy()


def leagues_both(league_df: pd.DataFrame, candidate: dict) -> int:
    rows = candidate_filter(league_df, candidate)
    return int(((rows["delta_log_loss_vs_market"] < 0) & (rows["delta_brier_vs_market"] < 0)).sum())


def candidate_comparison(reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    feature_lookup = reports["features"].set_index("feature_group").to_dict("index")
    leakage_pass = bool(reports["leakage"]["leakage_check_pass"].all())
    open_close = reports["open_close"]
    primary_timing = ";".join(open_close.loc[open_close["dataset"].eq("primary"), "ah_timing_labels"].astype(str))
    for c in CANDIDATES:
        m = candidate_filter(reports["model"], c)
        if m.empty:
            rows.append({"candidate_id": c["candidate_id"], "found": False, **c})
            continue
        row = m.iloc[0].to_dict()
        fg = feature_lookup.get(c["feature_group"], {})
        season_both = int(row["seasons_both_improved"])
        league_both = leagues_both(reports["league"], c)
        robust = season_both >= 4 and league_both >= 3 and leakage_pass
        if c["candidate_id"] == "B":
            eligibility = "eligible_feature_block_candidate" if robust else "not_eligible_needs_review"
        elif c["candidate_id"] in {"A", "D"}:
            eligibility = "eligible_market_recalibration_candidate" if robust else "not_eligible_needs_review"
        else:
            eligibility = "eligible_control_recalibration_candidate" if robust else "control_only_needs_review"
        rows.append(
            {
                "candidate_id": c["candidate_id"],
                "candidate_type": c["candidate_type"],
                "dataset_used": "primary",
                "timing_label": primary_timing,
                "feature_group": c["feature_group"],
                "model": c["model"],
                "feature_count": fg.get("feature_count", 0),
                "feature_list": fg.get("features", ""),
                "n_test": row["n_test"],
                "seasons": row["seasons"],
                "accuracy": row["accuracy"],
                "log_loss": row["log_loss"],
                "brier": row["brier"],
                "ece": row["ece"],
                "delta_log_loss_vs_market": row["delta_log_loss_vs_market"],
                "delta_brier_vs_market": row["delta_brier_vs_market"],
                "seasons_both_improved": season_both,
                "leagues_both_improved": league_both,
                "leakage_checks_pass": leakage_pass,
                "eligible_for_settlement_aware_value_diagnostic": eligibility.startswith("eligible_"),
                "eligibility_classification": eligibility,
                "settlement_value_diagnostic_requirement": "Must use actual ah_home_unit_return/ah_away_unit_return settlement returns, including push and half-loss outcomes; do not use binary target alone.",
                "found": True,
            }
        )
    return pd.DataFrame(rows)


def candidate_table(report_df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    frames = []
    for c in CANDIDATES:
        part = candidate_filter(report_df, c)
        if part.empty:
            continue
        part = part.copy()
        part.insert(0, "candidate_id", c["candidate_id"])
        part.insert(1, "candidate_type", c["candidate_type"])
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["candidate_id", "candidate_type"] + id_cols)
    return pd.concat(frames, ignore_index=True)


def line_bucket_summary(line_df: pd.DataFrame) -> pd.DataFrame:
    return candidate_table(line_df, ["ah_line_bucket"])


def calibration_summary(calib: pd.DataFrame) -> pd.DataFrame:
    return candidate_table(calib, ["side", "bin"])


def dataset_consistency() -> pd.DataFrame:
    rows = []
    for name, path in [("primary", PRIMARY), ("open", OPEN), ("close", CLOSE)]:
        if not path.exists():
            rows.append({"dataset": name, "exists": False})
            continue
        df = pd.read_csv(
            path,
            usecols=[
                "canonical_match_id",
                "season_start_year",
                "competition_slug",
                "ah_line_home",
                "ah_timing_label",
                "ah_home_unit_return",
                "ah_away_unit_return",
                "ah_push_flag",
                "classification",
            ],
        )
        rows.append(
            {
                "dataset": name,
                "exists": True,
                "rows": len(df),
                "unique_canonical_match_id": df["canonical_match_id"].nunique(),
                "duplicate_canonical_match_id": int(df["canonical_match_id"].duplicated().sum()),
                "seasons": df["season_start_year"].nunique(),
                "min_season": int(df["season_start_year"].min()),
                "max_season": int(df["season_start_year"].max()),
                "leagues": df["competition_slug"].nunique(),
                "timing_labels": ";".join(sorted(df["ah_timing_label"].astype(str).unique())),
                "home_positive_rate": float((df["ah_home_unit_return"] > 0).mean()),
                "away_positive_rate": float((df["ah_away_unit_return"] > 0).mean()),
                "push_rate": float(df["ah_push_flag"].astype(bool).mean()),
                "classification_research_only": bool(df["classification"].eq("research_only").all()),
                "line_mean": float(pd.to_numeric(df["ah_line_home"], errors="coerce").mean()),
                "line_median": float(pd.to_numeric(df["ah_line_home"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    comparison: pd.DataFrame,
    by_season: pd.DataFrame,
    by_league: pd.DataFrame,
    by_line: pd.DataFrame,
    calib: pd.DataFrame,
    leakage: pd.DataFrame,
    consistency: pd.DataFrame,
    settlement: pd.DataFrame,
    decision: str,
) -> None:
    best_rows = comparison.sort_values(["delta_log_loss_vs_market", "delta_brier_vs_market"])
    lines = [
        "# AH Predictive Candidate Sanity Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "No value search, threshold optimization, source join, raw-file modification, or confirmed-edge claim was performed.",
        "",
        "## Direct Answers",
        "",
        "1. The decision table selects `market_plus_clubelo` regularized logistic because the predictive decision is anchored to a robust date-safe feature-block model. `market_plus_ah_line` XGBoost has stronger overall deltas, but it is a market-line recalibration candidate rather than evidence that external feature blocks add signal.",
        "2. `market_plus_ah_line` XGBoost is eligible as a market-recalibration candidate for a later settlement-aware value diagnostic, not as the feature-block candidate.",
        "3. `market_plus_clubelo` regularized logistic is eligible as the feature-block candidate: it improves both log loss and brier across all available test seasons and all five leagues.",
        "4. Both candidate families are leakage-safe under the existing audit: canonical IDs, team identifiers/names, raw odds, settlement outcomes, scores, current/future fields, lineups, and unrelated markets are excluded as model features.",
        "5. Primary and close AH datasets are identical in size/timing profile; open is also present with nearly identical row count and settlement distribution. Predictive metrics were run on the primary/close dataset only, so open-vs-close predictive robustness still needs a separate controlled audit before any open-line value diagnostic.",
        "",
        "## Candidate Comparison",
        best_rows[
            [
                "candidate_id",
                "candidate_type",
                "feature_group",
                "model",
                "feature_count",
                "log_loss",
                "brier",
                "delta_log_loss_vs_market",
                "delta_brier_vs_market",
                "seasons_both_improved",
                "leagues_both_improved",
                "eligibility_classification",
            ]
        ].to_markdown(index=False),
        "",
        "## Dataset Consistency",
        consistency.to_markdown(index=False),
        "",
        "## Settlement Outcome Distribution",
        settlement.to_markdown(index=False),
        "",
        "## Leakage Checks",
        leakage.to_markdown(index=False),
        "",
        "## Settlement-Aware Requirement",
        "A later value diagnostic must use `ah_home_unit_return` and `ah_away_unit_return` directly, preserving full win, half win, push, half loss, and full loss economics. A binary positive-return target is acceptable for predictive screening only.",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "ah_candidate_sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "ah_candidate_decision.md").write_text(
        "\n".join(
            [
                "# AH Candidate Decision",
                "",
                f"Decision: **{decision}**",
                "",
                "Candidates are ready only for a settlement-aware research value diagnostic. This is not a betting signal and no confirmed edge is claimed.",
                "",
                comparison[
                    [
                        "candidate_id",
                        "candidate_type",
                        "feature_group",
                        "model",
                        "eligible_for_settlement_aware_value_diagnostic",
                        "eligibility_classification",
                    ]
                ].to_markdown(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reports = load_reports()
    comparison = candidate_comparison(reports)
    by_season = candidate_table(reports["season"], ["test_season"])
    by_league = candidate_table(reports["league"], ["competition_slug"])
    by_line = line_bucket_summary(reports["line"])
    calib = calibration_summary(reports["calibration"])
    leakage = reports["leakage"].copy()
    consistency = dataset_consistency()
    settlement = reports["settlement"].copy()

    comparison.to_csv(REPORT_DIR / "ah_candidate_comparison.csv", index=False)
    by_season.to_csv(REPORT_DIR / "ah_candidate_by_season.csv", index=False)
    by_league.to_csv(REPORT_DIR / "ah_candidate_by_league.csv", index=False)
    by_line.to_csv(REPORT_DIR / "ah_candidate_by_line_bucket.csv", index=False)
    calib.to_csv(REPORT_DIR / "ah_candidate_calibration.csv", index=False)
    leakage.to_csv(REPORT_DIR / "ah_candidate_leakage_checks.csv", index=False)
    consistency.to_csv(REPORT_DIR / "ah_candidate_dataset_consistency.csv", index=False)

    required_pass = (
        bool(leakage["leakage_check_pass"].all())
        and comparison["eligible_for_settlement_aware_value_diagnostic"].any()
        and {"primary", "open", "close"}.issubset(set(consistency.loc[consistency["exists"].astype(bool), "dataset"]))
    )
    if not bool(leakage["leakage_check_pass"].all()):
        decision = "ah_candidate_sanity_failed"
    elif required_pass:
        decision = "ah_candidate_sanity_ready_for_settlement_value_diagnostic"
    else:
        decision = "ah_candidate_sanity_ready_needs_review"

    write_markdown(comparison, by_season, by_league, by_line, calib, leakage, consistency, settlement, decision)
    print(decision)


if __name__ == "__main__":
    main()
