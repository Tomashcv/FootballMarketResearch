from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import (
    TEST_YEARS,
    VALUE_CONTROLS_CSV,
    VALUE_FIXED_CSV,
    VALUE_MD,
    VALUE_NESTED_CSV,
    VALUE_ROBUSTNESS_CSV,
    annual_predictions,
    feature_groups,
    load_data,
    scope_mask,
)


SUMMARY_CSV = Path("outputs/reports/feature_matrix_v2_tm_1x2_predictive_summary.csv")
REPORT_MD = Path("outputs/reports/feature_matrix_v2_tm_1x2_predictive_audit.md")

RULE_GRID = [
    (0.01, 1.50),
    (0.015, 1.50),
    (0.02, 1.50),
    (0.03, 1.50),
    (0.04, 1.50),
    (0.05, 1.50),
    (0.02, 1.80),
    (0.03, 1.80),
    (0.04, 1.80),
    (0.05, 1.80),
    (0.02, 2.00),
    (0.03, 2.00),
    (0.04, 2.00),
    (0.05, 2.00),
    (0.02, 2.50),
    (0.03, 2.50),
    (0.04, 2.50),
    (0.05, 2.50),
]


def add_value_columns(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "league",
        "season_start_year",
        "target_y",
        "x1x2_avg_prob_home",
        "x1x2_avg_prob_draw",
        "x1x2_avg_prob_away",
        "x1x2_avg_odds_home",
        "x1x2_avg_odds_draw",
        "x1x2_avg_odds_away",
    ]
    out = pred.merge(df[cols], on=["match_id", "league", "season_start_year", "target_y"], how="left")
    for side, cls in [("home", 0), ("draw", 1), ("away", 2)]:
        out[f"{side}_edge"] = out[f"prob_{side}"] - out[f"x1x2_avg_prob_{side}"]
        out[f"{side}_profit"] = np.where(out["target_y"].eq(cls), out[f"x1x2_avg_odds_{side}"] - 1.0, -1.0)
    out["test_year"] = out["season_start_year"].astype(int)
    return out


def select_rule(frame: pd.DataFrame, side: str, edge_threshold: float, min_odds: float) -> pd.DataFrame:
    selected = frame[
        frame[f"{side}_edge"].ge(edge_threshold)
        & frame[f"x1x2_avg_odds_{side}"].ge(min_odds)
    ].copy()
    selected["profit"] = selected[f"{side}_profit"]
    selected["side"] = side
    selected["rule_name"] = f"{side}_edge_{edge_threshold:g}_odds_{min_odds:g}"
    return selected


def bet_summary(selected: pd.DataFrame, label: str, rule_name: str) -> dict[str, object]:
    bets = int(len(selected))
    profit = float(selected["profit"].sum()) if bets else 0.0
    roi = profit / bets if bets else 0.0
    std = float(selected["profit"].std(ddof=1)) if bets > 1 else 0.0
    z = profit / (std * math.sqrt(bets)) if bets > 1 and std > 0 else 0.0
    return {
        "label": label,
        "rule_name": rule_name,
        "bets": bets,
        "profit": profit,
        "roi": roi,
        "z_score": z,
        "leagues": int(selected["league"].nunique()) if bets else 0,
        "years": int(selected["test_year"].nunique()) if bets else 0,
    }


def fixed_value_rules(test: pd.DataFrame, scope: str, model: str, feature_group: str) -> pd.DataFrame:
    rows = []
    for side in ["home", "draw", "away"]:
        for edge, odds in RULE_GRID:
            selected = select_rule(test, side, edge, odds)
            rule_name = selected["rule_name"].iloc[0] if len(selected) else f"{side}_edge_{edge:g}_odds_{odds:g}"
            row = bet_summary(selected, "fixed_rule", rule_name)
            row.update(
                {
                    "scope": scope,
                    "model": model,
                    "feature_group": feature_group,
                    "side": side,
                    "edge_threshold": edge,
                    "min_odds": odds,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def validation_rule_passes(selected: pd.DataFrame) -> bool:
    if len(selected) < 150:
        return False
    stats = bet_summary(selected, "validation", selected["rule_name"].iloc[0])
    if stats["roi"] <= 0 or stats["z_score"] <= 0.75 or stats["profit"] <= 0:
        return False
    if selected["league"].nunique() < 4:
        return False
    if selected["league"].value_counts(normalize=True).max() > 0.35:
        return False
    return True


def nested_selection(test: pd.DataFrame, scope: str, model: str, feature_group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bets = []
    for year in TEST_YEARS:
        prior = test[test["test_year"].lt(year)].copy()
        current = test[test["test_year"].eq(year)].copy()
        candidates = []
        if len(prior):
            for side in ["home", "draw", "away"]:
                for edge, odds in RULE_GRID:
                    selected = select_rule(prior, side, edge, odds)
                    if validation_rule_passes(selected):
                        stats = bet_summary(selected, "validation", selected["rule_name"].iloc[0])
                        stats.update({"side": side, "edge_threshold": edge, "min_odds": odds})
                        candidates.append(stats)
        if not candidates:
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "feature_group": feature_group,
                    "test_year": year,
                    "selected_rule": "",
                    "selection_status": "no_prior_rule_passed",
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "test_z": 0.0,
                }
            )
            continue
        chosen = pd.DataFrame(candidates).sort_values(["z_score", "profit", "bets"], ascending=[False, False, False]).iloc[0]
        selected_test = select_rule(current, str(chosen["side"]), float(chosen["edge_threshold"]), float(chosen["min_odds"]))
        stats = bet_summary(selected_test, "test", str(chosen["rule_name"]))
        rows.append(
            {
                "scope": scope,
                "model": model,
                "feature_group": feature_group,
                "test_year": year,
                "selected_rule": chosen["rule_name"],
                "selection_status": "selected_prior_out_of_sample_only",
                "test_bets": stats["bets"],
                "test_profit": stats["profit"],
                "test_roi": stats["roi"],
                "test_z": stats["z_score"],
            }
        )
        bets.append(selected_test.assign(selected_rule=str(chosen["rule_name"])))
    return pd.DataFrame(rows), pd.concat(bets, ignore_index=True, sort=False) if bets else pd.DataFrame()


def value_controls(test: pd.DataFrame, selected: pd.DataFrame, scope: str, model: str, feature_group: str, rule_name: str) -> pd.DataFrame:
    rows = [
        {
            "scope": scope,
            "model": model,
            "feature_group": feature_group,
            "control": "selected_rule",
            **bet_summary(selected, "control", rule_name),
        }
    ]
    if len(selected):
        rng = np.random.default_rng(20260701)
        sample = test.sample(n=min(len(selected), len(test)), replace=False, random_state=20260701).copy()
        choices = rng.integers(0, 3, len(sample))
        sample["profit"] = np.select(
            [choices == 0, choices == 1, choices == 2],
            [sample["home_profit"], sample["draw_profit"], sample["away_profit"]],
        )
        rows.append(
            {
                "scope": scope,
                "model": model,
                "feature_group": feature_group,
                "control": "random_same_bet_count",
                **bet_summary(sample, "control", "random_same_bet_count"),
            }
        )
    return pd.DataFrame(rows)


def value_robustness(selected: pd.DataFrame, scope: str, model: str, feature_group: str, portfolio: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            [
                {
                    "scope": scope,
                    "model": model,
                    "feature_group": feature_group,
                    "portfolio": portfolio,
                    "robustness": "empty",
                    "bets": 0,
                    "profit": 0.0,
                    "roi": 0.0,
                    "z_score": 0.0,
                }
            ]
        )
    best_year = selected.groupby("test_year")["profit"].sum().sort_values(ascending=False).index[0]
    best_league = selected.groupby("league")["profit"].sum().sort_values(ascending=False).index[0]
    rows = []
    for name, frame in [
        ("all", selected),
        ("exclude_best_profit_season", selected[selected["test_year"].ne(best_year)]),
        ("exclude_best_profit_league", selected[~selected["league"].eq(best_league)]),
    ]:
        rows.append(
            {
                "scope": scope,
                "model": model,
                "feature_group": feature_group,
                "portfolio": portfolio,
                "robustness": name,
                **bet_summary(frame, "robustness", portfolio),
            }
        )
    return pd.DataFrame(rows)


def passed_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    tm_groups = {
        "x1_market_plus_tm_all",
        "x1_full_safe_v2",
        "x1_market_plus_tm_valuation",
        "x1_market_plus_transfer_churn",
    }
    out = summary[
        summary["control"].eq("none")
        & summary["feature_group"].isin(tm_groups)
        & summary["delta_log_loss_vs_raw_market"].lt(0)
        & summary["delta_brier_vs_raw_market"].lt(0)
        & summary["delta_log_loss_vs_v1_1_residual"].lt(0)
        & summary["delta_ece_vs_raw_market"].le(0.005)
    ].copy()
    return out.sort_values("delta_log_loss_vs_raw_market")


def classify_value(nested_bets: pd.DataFrame, controls: pd.DataFrame, robust: pd.DataFrame) -> str:
    if nested_bets.empty:
        return "predictive_only_no_value"
    stats = bet_summary(nested_bets, "nested", "nested")
    season_profit = nested_bets.groupby("test_year")["profit"].sum()
    league_profit = nested_bets.groupby("league")["profit"].sum()
    positive_profit = max(stats["profit"], 0.0)
    no_majority_season = positive_profit > 0 and season_profit.max() <= 0.5 * positive_profit
    no_majority_league = positive_profit > 0 and league_profit.max() <= 0.5 * positive_profit
    robust_row = robust[robust["robustness"].eq("exclude_best_profit_season")]
    robust_positive = len(robust_row) > 0 and float(robust_row["profit"].iloc[0]) > 0
    control_best = controls[~controls["control"].eq("selected_rule")]["profit"].max() if len(controls) else 0.0
    controls_fail = pd.isna(control_best) or float(control_best) < stats["profit"]
    if (
        stats["profit"] > 0
        and stats["roi"] > 0.02
        and stats["z_score"] >= 1.0
        and stats["years"] >= 4
        and no_majority_season
        and no_majority_league
        and robust_positive
        and controls_fail
    ):
        return "forward_paper_candidate"
    return "research_only" if stats["bets"] > 0 else "predictive_only_no_value"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[c for c in columns if c in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    df = load_data()
    groups = feature_groups(df)
    summary = pd.read_csv(SUMMARY_CSV)
    passed = passed_candidates(summary)
    fixed_parts = []
    nested_parts = []
    control_parts = []
    robust_parts = []
    classes = []
    for row in passed.itertuples(index=False):
        scope = str(row.scope)
        fg = str(row.feature_group)
        model = str(row.model)
        print(f"value_review scope={scope} feature_group={fg} model={model}", flush=True)
        pred, _ = annual_predictions(df, scope, fg, model, groups[fg])
        pred = add_value_columns(pred, df[scope_mask(df, scope)].copy())
        fixed = fixed_value_rules(pred, scope, model, fg)
        nested, nested_bets = nested_selection(pred, scope, model, fg)
        fixed_parts.append(fixed)
        nested_parts.append(nested)
        if len(fixed):
            best = fixed.sort_values(["profit", "z_score"], ascending=[False, False]).iloc[0]
            best_bets = select_rule(pred, str(best["side"]), float(best["edge_threshold"]), float(best["min_odds"]))
            control_parts.append(value_controls(pred, best_bets, scope, model, fg, str(best["rule_name"])))
            robust_parts.append(value_robustness(best_bets, scope, model, fg, "best_fixed_rule"))
        if len(nested_bets):
            vc = value_controls(pred, nested_bets, scope, model, fg, "nested_portfolio")
            vr = value_robustness(nested_bets, scope, model, fg, "nested_portfolio")
            control_parts.append(vc)
            robust_parts.append(vr)
            classes.append(classify_value(nested_bets, vc, vr))
        else:
            classes.append("predictive_only_no_value")
    value_fixed = pd.concat(fixed_parts, ignore_index=True, sort=False) if fixed_parts else pd.DataFrame()
    value_nested = pd.concat(nested_parts, ignore_index=True, sort=False) if nested_parts else pd.DataFrame()
    value_controls_df = pd.concat(control_parts, ignore_index=True, sort=False) if control_parts else pd.DataFrame()
    value_robustness_df = pd.concat(robust_parts, ignore_index=True, sort=False) if robust_parts else pd.DataFrame()
    value_fixed.to_csv(VALUE_FIXED_CSV, index=False)
    value_nested.to_csv(VALUE_NESTED_CSV, index=False)
    value_controls_df.to_csv(VALUE_CONTROLS_CSV, index=False)
    value_robustness_df.to_csv(VALUE_ROBUSTNESS_CSV, index=False)
    final_class = "forward_paper_candidate" if "forward_paper_candidate" in classes else ("research_only" if "research_only" in classes else "predictive_only_no_value")
    if final_class == "research_only":
        final_class = "predictive_only_no_value"
    VALUE_MD.write_text(
        "\n".join(
            [
                "# 1X2 locked value review",
                "",
                f"Final classification: `{final_class}`",
                "",
                "Predictive gate passed for at least one Transfermarkt candidate. Fixed rules used the predeclared prior 1X2 edge/odds grid; nested selection used only prior out-of-sample years. No post-test threshold mining, live betting, closing-odds selection, or confirmed-edge claim was used.",
                "",
                "## Fixed Rules",
                "",
                markdown_table(value_fixed.sort_values(["profit", "z_score"], ascending=[False, False]) if len(value_fixed) else value_fixed, ["scope", "model", "feature_group", "rule_name", "bets", "profit", "roi", "z_score", "leagues", "years"], 50),
                "",
                "## Nested Temporal Selection",
                "",
                markdown_table(value_nested, ["scope", "model", "feature_group", "test_year", "selected_rule", "selection_status", "test_bets", "test_profit", "test_roi", "test_z"], 80),
                "",
                "No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if REPORT_MD.exists():
        text = REPORT_MD.read_text(encoding="utf-8")
        text = text.replace("Final classification: `predictive_only_no_value`", f"Final classification: `{final_class}`")
        text = text.replace("Locked value review run: `True`", "Locked value review run: `True`")
        REPORT_MD.write_text(text, encoding="utf-8")
    print({"passed_candidates": len(passed), "value_fixed_rows": len(value_fixed), "value_nested_rows": len(value_nested), "classification": final_class})


if __name__ == "__main__":
    main()
