from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments import transfermarkt_proxy_predictive_audit as audit
from src.features.contextual_features import build_contextual_features


LEAGUE = "P1"
REPORT_PATH = Path("outputs/reports/p1_transfermarkt_proxy_predictive_audit.md")
SUMMARY_PATH = Path("outputs/reports/p1_transfermarkt_proxy_predictive_summary.csv")
COVERAGE_PATH = Path("outputs/reports/p1_transfermarkt_proxy_target_coverage.csv")
NEGATIVE_PATH = Path("outputs/reports/p1_transfermarkt_proxy_negative_controls.csv")

FEATURE_GROUPS = {
    "market_baseline",
    "tm_proxy_only",
    "market_plus_tm_365d",
    "market_plus_tm_180d_365d",
}


def load_p1_dataset() -> pd.DataFrame:
    matches = pd.read_csv("data/processed/P1/P1_matches.csv", low_memory=False)
    matches["league"] = LEAGUE
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce").dt.normalize()
    matches["season_end_year"] = pd.to_numeric(matches["season_end_year"], errors="coerce")
    contextual = build_contextual_features(matches)
    proxy = pd.read_csv(audit.PROXY_PATH, low_memory=False)
    proxy["Date"] = pd.to_datetime(proxy["Date"], errors="coerce").dt.normalize()
    proxy = proxy[proxy["league"].eq(LEAGUE)].copy()
    keep_proxy = ["league", "Date", "HomeTeam", "AwayTeam"] + [
        column for column in proxy.columns if "_tm_" in column and "mapped_club_name" not in column
    ]
    output = contextual.merge(proxy[keep_proxy], on=["league", "Date", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one")
    return audit.add_targets(output.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True))


def target_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required_1x2 = ["AvgH", "AvgD", "AvgA", "FTR"]
    required_ah = ["AHh", "AvgAHH", "AvgAHA", "FTHG", "FTAG"]
    closing_ah = ["AHCh", "AvgCAHH", "AvgCAHA"]
    for season, group in frame.groupby("season_end_year", dropna=False):
        missing_required = []
        for column in sorted(set(required_1x2 + required_ah)):
            if column not in group.columns:
                missing_required.append(column)
        row = {
            "league": LEAGUE,
            "season_end_year": int(season) if pd.notna(season) else pd.NA,
            "matches": len(group),
            "one_x_two_odds_coverage": float(group[["AvgH", "AvgD", "AvgA"]].notna().all(axis=1).mean())
            if {"AvgH", "AvgD", "AvgA"}.issubset(group.columns)
            else 0.0,
            "ah_odds_coverage": float(group[["AHh", "AvgAHH", "AvgAHA"]].notna().all(axis=1).mean())
            if {"AHh", "AvgAHH", "AvgAHA"}.issubset(group.columns)
            else 0.0,
            "closing_ah_coverage_diagnostic_only": float(group[closing_ah].notna().all(axis=1).mean())
            if set(closing_ah).issubset(group.columns)
            else 0.0,
            "usable_ah_target_rows": int(group.dropna(subset=[c for c in required_ah if c in group.columns] + ["target_ah_home_cover"]).shape[0])
            if set(required_ah).issubset(group.columns)
            else 0,
            "usable_1x2_target_rows": int(group.dropna(subset=[c for c in required_1x2 if c in group.columns] + ["target_1x2"]).shape[0])
            if set(required_1x2).issubset(group.columns)
            else 0,
            "away_ah_big_home_favourite_rows": int(group["subset_away_ah_big_home_favourite"].fillna(False).sum()),
            "missing_required_columns": ",".join(missing_required),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("season_end_year").reset_index(drop=True)


def run_p1_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    negative_rows = []
    years = sorted(pd.to_numeric(frame["season_end_year"], errors="coerce").dropna().astype(int).unique())
    for target in ["ah_home_cover", "outcome_1x2"]:
        target_column = "target_ah_home_cover" if target == "ah_home_cover" else "target_1x2"
        models = ["logistic_l2", "logistic_elasticnet"]
        subsets = ["subset_all"] + (["subset_away_ah_big_home_favourite"] if target == "ah_home_cover" else [])
        for subset in subsets:
            for test_year in years:
                train, validation, test = audit.prepare_fold_data(frame, LEAGUE, target, subset, test_year)
                if len(train) < 300 or len(validation) < 50 or len(test) < 50:
                    continue
                groups = {name: cols for name, cols in audit.feature_groups(frame, target).items() if name in FEATURE_GROUPS}
                market_p = audit.market_probabilities(test, target)
                y_test = test[target_column].astype(int).to_numpy()
                market_metrics = audit.metrics(y_test, market_p, target)
                for group_name, features in groups.items():
                    available = [column for column in features if column in train.columns]
                    if not available:
                        continue
                    missing = train[available].isna().mean()
                    for model_name in models:
                        try:
                            probabilities, _ = audit.fit_predict(train, test, available, target_column, target, model_name)
                        except Exception:
                            continue
                        result = audit.metrics(y_test, probabilities, target)
                        rows.append(
                            {
                                "league": LEAGUE,
                                "target": target,
                                "subset": subset,
                                "feature_group": group_name,
                                "model": model_name,
                                "test_year": test_year,
                                "validation_year": test_year - 1,
                                "train_rows": len(train),
                                "validation_rows": len(validation),
                                "rows": len(test),
                                "feature_count": len(available),
                                "mean_train_feature_missing_rate": float(missing.mean()),
                                "max_train_feature_missing_rate": float(missing.max()),
                                "tm_365d_both_coverage_train": float(
                                    train[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()
                                ),
                                "tm_365d_both_coverage_validation": float(
                                    validation[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()
                                ),
                                "tm_365d_both_coverage_test": float(
                                    test[["home_tm_value_total_365d", "away_tm_value_total_365d"]].notna().all(axis=1).mean()
                                ),
                                "accuracy": result["accuracy"],
                                "log_loss": result["log_loss"],
                                "brier": result["brier"],
                                "ece": result["ece"],
                                "market_log_loss": market_metrics["log_loss"],
                                "market_brier": market_metrics["brier"],
                                "market_ece": market_metrics["ece"],
                                "delta_log_loss_vs_market_baseline": result["log_loss"] - market_metrics["log_loss"],
                                "delta_brier_vs_market_baseline": result["brier"] - market_metrics["brier"],
                                "delta_ece_vs_market_baseline": result["ece"] - market_metrics["ece"],
                            }
                        )
                for group_name in ["market_plus_tm_365d", "market_plus_tm_180d_365d"]:
                    features = groups.get(group_name, [])
                    if not features:
                        continue
                    for control in [
                        "permute_tm_within_season",
                        "random_noise_same_shape",
                        "shuffled_train_labels",
                        "market_baseline_without_tm",
                    ]:
                        if control == "market_baseline_without_tm":
                            control_features = groups["market_baseline"]
                            train_c, test_c = train, test
                        else:
                            control_features = features
                            train_c, test_c = audit.apply_negative_control(train, test, control_features, control, test_year, target_column)
                        try:
                            probabilities, _ = audit.fit_predict(train_c, test_c, control_features, target_column, target, "logistic_l2")
                        except Exception:
                            continue
                        result = audit.metrics(y_test, probabilities, target)
                        negative_rows.append(
                            {
                                "league": LEAGUE,
                                "target": target,
                                "subset": subset,
                                "feature_group": group_name,
                                "control": control,
                                "model": "logistic_l2",
                                "test_year": test_year,
                                "rows": len(test),
                                "log_loss": result["log_loss"],
                                "brier": result["brier"],
                                "ece": result["ece"],
                                "delta_log_loss_vs_market_baseline": result["log_loss"] - market_metrics["log_loss"],
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(negative_rows)


def classify(summary: pd.DataFrame, negatives: pd.DataFrame) -> str:
    if summary.empty:
        return "p1_reject"
    primary = summary[
        summary["target"].eq("ah_home_cover")
        & summary["subset"].eq("subset_all")
        & summary["feature_group"].isin(["market_plus_tm_365d", "market_plus_tm_180d_365d"])
        & summary["model"].eq("logistic_l2")
    ].copy()
    if primary.empty:
        return "p1_predictive_diagnostic_only"
    grouped = primary.groupby("feature_group").agg(
        seasons=("test_year", "nunique"),
        improved=("delta_log_loss_vs_market_baseline", lambda s: int((s < 0).sum())),
        mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
        mean_delta_brier=("delta_brier_vs_market_baseline", "mean"),
        mean_delta_ece=("delta_ece_vs_market_baseline", "mean"),
    )
    candidate = grouped[
        (grouped["improved"] > 1)
        & (grouped["mean_delta_log_loss"] < 0)
        & (grouped["mean_delta_brier"] <= 0.002)
        & (grouped["mean_delta_ece"] <= 0.02)
    ]
    if candidate.empty:
        return "p1_predictive_diagnostic_only"
    real_best = primary["delta_log_loss_vs_market_baseline"].mean()
    neg_best = negatives["delta_log_loss_vs_market_baseline"].min() if len(negatives) else np.nan
    if pd.notna(neg_best) and neg_best <= real_best:
        return "p1_predictive_diagnostic_only"
    return "p1_predictive_signal_candidate"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(max_rows).fillna("").to_markdown(index=False)


def write_report(summary: pd.DataFrame, coverage: pd.DataFrame, negatives: pd.DataFrame, classification: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    coverage.to_csv(COVERAGE_PATH, index=False)
    negatives.to_csv(NEGATIVE_PATH, index=False)
    aggregate = (
        summary.groupby(["target", "subset", "feature_group", "model"])
        .agg(
            seasons=("test_year", "nunique"),
            rows=("rows", "sum"),
            mean_delta_log_loss=("delta_log_loss_vs_market_baseline", "mean"),
            improved_seasons=("delta_log_loss_vs_market_baseline", lambda s: int((s < 0).sum())),
            mean_delta_brier=("delta_brier_vs_market_baseline", "mean"),
            mean_delta_ece=("delta_ece_vs_market_baseline", "mean"),
        )
        .reset_index()
        .sort_values(["target", "subset", "mean_delta_log_loss"])
        if len(summary)
        else pd.DataFrame()
    )
    lines = [
        "# P1 Transfermarkt Proxy Predictive Audit",
        "",
        f"Final classification: **{classification}**",
        "",
        "No betting strategies, value searches, threshold optimization, closing-odds features, diagnostic-only club history, `players.current_club_*`, or lineups were used.",
        "Closing AH coverage is reported only as a diagnostic availability field.",
        "",
        "## Target And Odds Coverage",
        markdown_table(
            coverage,
            [
                "season_end_year",
                "matches",
                "one_x_two_odds_coverage",
                "ah_odds_coverage",
                "closing_ah_coverage_diagnostic_only",
                "usable_ah_target_rows",
                "usable_1x2_target_rows",
                "away_ah_big_home_favourite_rows",
                "missing_required_columns",
            ],
            max_rows=30,
        ),
        "",
        "## Predictive Aggregate",
        markdown_table(
            aggregate,
            [
                "target",
                "subset",
                "feature_group",
                "model",
                "seasons",
                "rows",
                "mean_delta_log_loss",
                "improved_seasons",
                "mean_delta_brier",
                "mean_delta_ece",
            ],
            max_rows=60,
        ),
        "",
        "## Season Metrics",
        markdown_table(
            summary.sort_values(["target", "subset", "test_year", "feature_group", "model"]),
            [
                "target",
                "subset",
                "feature_group",
                "model",
                "test_year",
                "rows",
                "accuracy",
                "log_loss",
                "brier",
                "ece",
                "delta_log_loss_vs_market_baseline",
            ],
            max_rows=80,
        ),
        "",
        "## Negative Controls",
        markdown_table(
            negatives.sort_values(["target", "subset", "test_year", "feature_group", "control"]),
            ["target", "subset", "feature_group", "control", "test_year", "rows", "log_loss", "delta_log_loss_vs_market_baseline"],
            max_rows=80,
        ),
        "",
        "Conclusion: P1 is usable for predictive diagnostics where AH and 1X2 rows exist, but the success gate is not met unless the classification above says otherwise. No confirmed edge is claimed.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frame = load_p1_dataset()
    coverage = target_coverage(frame)
    summary, negatives = run_p1_audit(frame)
    classification = classify(summary, negatives)
    write_report(summary, coverage, negatives, classification)
    print(f"coverage_rows: {len(coverage)}")
    print(f"summary_rows: {len(summary)}")
    print(f"negative_control_rows: {len(negatives)}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
