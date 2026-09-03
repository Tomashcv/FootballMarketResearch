from __future__ import annotations

from pathlib import Path
import math
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import (
    CLASS_TO_INT,
    MARKET_BASELINE,
    TEST_YEARS,
    brier_multi,
    ece_multi,
    feature_groups,
    model_predict,
    normalize_probs,
    scope_mask,
)
from src.experiments.feature_matrix_v2_tm_1x2_value_review import add_value_columns, nested_selection
from src.experiments.feature_matrix_v3_clubelo_vs_v2_comparison import (
    CLUBELO_CORE,
    CLUBELO_STALENESS,
    LOCKED_RULE_SCHEDULE,
    MODEL,
    SCOPE,
    bet_summary,
    breakdown,
    edge_buckets,
    md_table,
    select_locked_schedule,
    status,
    z_score,
)


REPORT_DIR = Path("outputs/reports")
V3_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
V4_MATRIX = Path("data/processed/features/football_feature_matrix_v4_1_understat_partial_v2.csv")
V4_LEAKAGE = REPORT_DIR / "feature_matrix_v4_1_understat_v2_leakage_checks.csv"

REPORT_MD = REPORT_DIR / "feature_matrix_v4_1_understat_vs_v3_comparison.md"
PREDICTIVE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_predictive_summary.csv"
VALUE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_fixed_rule_value_replay.csv"
YEAR_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_year_breakdown.csv"
LEAGUE_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_league_breakdown.csv"
NESTED_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_nested_diagnostic.csv"
STALENESS_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_staleness_diagnostics.csv"
EDGE_BUCKET_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_edge_bucket_calibration.csv"
BUG_CHECKS_CSV = REPORT_DIR / "feature_matrix_v4_1_understat_bug_checks.csv"
DECISION_MD = REPORT_DIR / "feature_matrix_v4_1_understat_decision.md"

TOP5 = {"E0", "D1", "SP1", "I1", "F1"}
SOURCE_END = pd.Timestamp("2024-09-29")
BASELINE = "v3_baseline"
CORE = "v4_1_understat_core"
FULL = "v4_1_understat_full"
SCOPES = ["top5_only", "top5_understat_available", "full_locked_scope_diagnostic"]


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").astype("Int64")
    df["target_y"] = df["target_outcome_1x2"].map(CLASS_TO_INT)
    for col in MARKET_BASELINE:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = (
        df["target_1x2_available"].fillna(0).astype(bool)
        & df["target_y"].notna()
        & df[MARKET_BASELINE[:3]].notna().all(axis=1)
        & df[["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]].notna().all(axis=1)
    )
    out = df[valid].copy()
    out["target_y"] = out["target_y"].astype(int)
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def scope_filter(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "top5_only":
        return df["league"].isin(TOP5)
    if name == "top5_understat_available":
        return df["league"].isin(TOP5) & df["understat_both_available_flag"].fillna(False).astype(bool)
    if name == "full_locked_scope_diagnostic":
        return scope_mask(df, SCOPE)
    raise ValueError(name)


def understat_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    under = [c for c in df.columns if "understat" in c and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    flags = ["understat_home_available_flag", "understat_away_available_flag", "understat_both_available_flag"]
    core_tokens = ["xg_for", "xg_against", "npxg_for", "npxg_against", "xg_diff", "npxg_diff"]
    core = [c for c in under if any(tok in c for tok in core_tokens) or c in flags]
    full = under
    return sorted(set(core)), sorted(set(full))


def feature_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    tm_cols = feature_groups(df)["x1_market_plus_tm_all"]
    baseline = sorted(set(tm_cols + CLUBELO_CORE + CLUBELO_STALENESS))
    under_core, under_full = understat_feature_columns(df)
    return {
        BASELINE: baseline,
        CORE: sorted(set(baseline + under_core)),
        FULL: sorted(set(baseline + under_full)),
    }


def latest_understat_days(frame: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            pd.to_numeric(frame.get("understat_home_latest_days_ago", pd.Series(np.nan, index=frame.index)), errors="coerce"),
            pd.to_numeric(frame.get("understat_away_latest_days_ago", pd.Series(np.nan, index=frame.index)), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)


def annual_predictions(df: pd.DataFrame, scope_name: str, label: str, cols: list[str]) -> pd.DataFrame:
    scoped = df[scope_filter(df, scope_name)].copy()
    preds = []
    rng = np.random.default_rng(20260702)
    for year in TEST_YEARS:
        train = scoped[scoped["season_start_year"].astype(int).lt(year)].copy()
        test = scoped[scoped["season_start_year"].astype(int).eq(year)].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        print(f"understat_compare scope={scope_name} feature_group={label} test_year={year} rows={len(test)}", flush=True)
        prob = model_predict(MODEL, train, test, cols, rng)
        pred = test[["match_id", "match_date", "league", "season_start_year", "target_y"]].copy()
        pred[["prob_home", "prob_draw", "prob_away"]] = prob
        pred["scope"] = scope_name
        pred["feature_group"] = label
        pred["model"] = MODEL
        preds.append(pred)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def add_understat_context(value_pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "league",
        "season_start_year",
        "understat_home_available_flag",
        "understat_away_available_flag",
        "understat_both_available_flag",
        "understat_home_latest_days_ago",
        "understat_away_latest_days_ago",
        "understat_home_history_count",
        "understat_away_history_count",
    ]
    present = [c for c in cols if c in df.columns]
    return value_pred.merge(df[present], on=["match_id", "league", "season_start_year"], how="left", validate="many_to_one")


def metrics_for_predictions(value_pred: pd.DataFrame, scope_name: str, label: str) -> dict[str, object]:
    y = value_pred["target_y"].to_numpy(dtype=int)
    prob = normalize_probs(value_pred[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float))
    latest_days = latest_understat_days(value_pred)
    both_avail = value_pred.get("understat_both_available_flag", pd.Series(False, index=value_pred.index)).fillna(False).astype(bool)
    after_source_end = pd.to_datetime(value_pred["match_date"]).gt(SOURCE_END)
    return {
        "scope": scope_name,
        "feature_group": label,
        "rows": int(len(value_pred)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1, 2])),
        "brier": brier_multi(y, prob),
        "ece": ece_multi(y, prob),
        "class_log_loss_home": float(log_loss((y == 0).astype(int), prob[:, 0], labels=[0, 1])),
        "class_log_loss_draw": float(log_loss((y == 1).astype(int), prob[:, 1], labels=[0, 1])),
        "class_log_loss_away": float(log_loss((y == 2).astype(int), prob[:, 2], labels=[0, 1])),
        "away_mean_pred": float(prob[:, 2].mean()),
        "away_observed_rate": float((y == 2).mean()),
        "away_calibration_error": float(prob[:, 2].mean() - (y == 2).mean()),
        "away_brier": float(np.mean((prob[:, 2] - (y == 2).astype(float)) ** 2)),
        "both_understat_available_rows": int(both_avail.sum()),
        "rows_after_understat_source_end": int(after_source_end.sum()),
        "latest_days_gt_180_rows": int(latest_days.gt(180).sum()),
        "latest_days_gt_365_rows": int(latest_days.gt(365).sum()),
        "latest_days_gt_730_rows": int(latest_days.gt(730).sum()),
    }


def scoped_edge_buckets(value_pred: pd.DataFrame, scope_name: str, label: str) -> pd.DataFrame:
    out = edge_buckets(value_pred, label)
    out.insert(0, "scope", scope_name)
    return out


def selected_summary(selected: pd.DataFrame, scope_name: str, label: str) -> dict[str, object]:
    row = bet_summary(selected, label)
    row["scope"] = scope_name
    return row


def selected_breakdown(selected: pd.DataFrame, scope_name: str, label: str, group_col: str) -> pd.DataFrame:
    out = breakdown(selected, label, group_col)
    if out.empty:
        return pd.DataFrame(columns=["scope", "feature_group", group_col, "bets", "profit", "roi", "z"])
    out.insert(0, "scope", scope_name)
    return out


def staleness_metrics(value_pred: pd.DataFrame, scope_name: str, label: str) -> pd.DataFrame:
    latest_days = latest_understat_days(value_pred)
    both = value_pred.get("understat_both_available_flag", pd.Series(False, index=value_pred.index)).fillna(False).astype(bool)
    after_source_end = pd.to_datetime(value_pred["match_date"]).gt(SOURCE_END)
    masks = {
        "all_2020_2025": pd.Series(True, index=value_pred.index),
        "exclude_2025": value_pred["season_start_year"].astype(int).ne(2025),
        "exclude_after_2024_09_29": ~after_source_end,
        "both_understat_available": both,
        "latest_days_le_180": latest_days.le(180),
        "latest_days_le_365": latest_days.le(365),
        "latest_days_le_730": latest_days.le(730),
    }
    rows = []
    for segment, mask in masks.items():
        g = value_pred[mask].copy()
        if g.empty:
            rows.append({"scope": scope_name, "feature_group": label, "segment": segment, "rows": 0})
        else:
            rows.append({"segment": segment, **metrics_for_predictions(g, scope_name, label)})
    return pd.DataFrame(rows)


def nested_diagnostic(value_pred: pd.DataFrame, scope_name: str, label: str) -> pd.DataFrame:
    nested, _ = nested_selection(value_pred, scope_name, MODEL, label)
    nested["diagnostic_only"] = True
    return nested


def bug_checks(v3_raw: pd.DataFrame, v4_raw: pd.DataFrame, feature_cols: dict[str, list[str]], selected_by_key: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    unchanged_bad = []
    for col in v3_raw.columns:
        left = v3_raw[col]
        right = v4_raw[col]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            ok = np.allclose(
                pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            ok = left.astype("string").fillna("<NA>").equals(right.astype("string").fillna("<NA>"))
        if not ok:
            unchanged_bad.append(col)
    leakage = pd.read_csv(V4_LEAKAGE) if V4_LEAKAGE.exists() else pd.DataFrame(columns=["check", "status", "count"])
    leak_pass = bool(len(leakage) and leakage["status"].eq("pass").all())
    all_cols = sorted(set(sum(feature_cols.values(), [])))
    bad_names = [c for c in all_cols if c in v4_raw.columns and pd.api.types.is_object_dtype(v4_raw[c]) and re.search(r"team|club|name", c, re.I)]
    bad_current = [c for c in all_cols if re.search(r"current|scored|missed|result", c, re.I)]
    bad_xg_direct = [c for c in all_cols if re.fullmatch(r".*(^|_)xg($|_current).*", c, re.I)]
    closing = [c for c in all_cols if re.search(r"closing|_close|close_", c, re.I)]
    duplicate_selected = 0
    non_away = 0
    profit_bad = 0
    odds_bad = 0
    for selected in selected_by_key.values():
        duplicate_selected += int(selected.duplicated(["match_id"]).sum()) if len(selected) else 0
        non_away += int((~selected.get("side", pd.Series("away", index=selected.index)).eq("away")).sum()) if len(selected) else 0
        if len(selected):
            expected = np.where(selected["target_y"].eq(2), selected["x1x2_avg_odds_away"] - 1.0, -1.0)
            profit_bad += int((np.round(selected["profit"].to_numpy(dtype=float), 12) != np.round(expected, 12)).sum())
            odds_bad += int((~selected["x1x2_avg_odds_away"].gt(1.0)).sum())
    checks = [
        ("v4_1_row_count_equals_v3", len(v3_raw) == len(v4_raw), len(v4_raw), f"v3={len(v3_raw)} v4={len(v4_raw)}"),
        ("original_v3_columns_unchanged", len(unchanged_bad) == 0, len(unchanged_bad), "|".join(unchanged_bad[:20])),
        ("understat_v2_leakage_checks_all_passed", leak_pass, int(leakage["status"].ne("pass").sum()) if len(leakage) else -1, ""),
        ("no_current_match_understat_result_xg_scored_missed_direct_features", len(bad_current) == 0 and len(bad_xg_direct) == 0, len(bad_current) + len(bad_xg_direct), "|".join((bad_current + bad_xg_direct)[:20])),
        ("no_team_name_string_features", len(bad_names) == 0, len(bad_names), "|".join(bad_names)),
        ("no_closing_odds_used_for_selection", len(closing) == 0, len(closing), "|".join(closing)),
        ("selected_away_bets_use_away_side", non_away == 0, non_away, ""),
        ("profit_formula_correct", profit_bad == 0, profit_bad, ""),
        ("no_impossible_selected_odds", odds_bad == 0, odds_bad, ""),
        ("no_duplicate_selected_matches_within_group", duplicate_selected == 0, duplicate_selected, ""),
    ]
    # Surface the concrete date-safety rows from the v4.1 build audit as explicit checks.
    for name in ["all_contributing_understat_rows_strictly_before_match_date", "no_same_day_understat_joins", "no_future_understat_joins"]:
        row = leakage[leakage["check"].eq(name)]
        checks.append((name, bool(len(row) and row["status"].iloc[0] == "pass"), int(row["count"].iloc[0]) if len(row) else -1, "from v4.1 build leakage audit"))
    return pd.DataFrame([{"check": n, "status": status(bool(ok)), "count": int(count), "detail": detail} for n, ok, count, detail in checks])


def add_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for scope_name, g in out.groupby("scope"):
        base = g[g["feature_group"].eq(BASELINE)].iloc[0]
        idx = out["scope"].eq(scope_name)
        for metric in ["log_loss", "brier", "ece", "class_log_loss_away", "away_brier"]:
            out.loc[idx, f"delta_{metric}_vs_v3"] = out.loc[idx, metric] - float(base[metric])
    return out


def add_value_deltas(value: pd.DataFrame) -> pd.DataFrame:
    out = value.copy()
    for scope_name, g in out.groupby("scope"):
        base = g[g["feature_group"].eq(BASELINE)].iloc[0]
        idx = out["scope"].eq(scope_name)
        for metric in ["bets", "profit", "roi", "z"]:
            out.loc[idx, f"delta_{metric}_vs_v3"] = out.loc[idx, metric] - float(base[metric])
    return out


def classify_decision(summary: pd.DataFrame, value: pd.DataFrame, checks: pd.DataFrame, stale: pd.DataFrame) -> str:
    if not checks["status"].eq("pass").all():
        return "v4_1_rejected_bug_or_leakage"

    def scope_ok(scope_name: str) -> tuple[bool, bool, bool, bool]:
        pred = summary[(summary["scope"].eq(scope_name)) & (summary["feature_group"].isin([CORE, FULL]))].copy()
        val = value[(value["scope"].eq(scope_name)) & (value["feature_group"].isin([CORE, FULL]))].copy()
        if pred.empty or val.empty:
            return False, False, False, False
        best_pred = pred.sort_values(["delta_log_loss_vs_v3", "delta_brier_vs_v3"]).iloc[0]
        best_val = val[val["feature_group"].eq(best_pred["feature_group"])].iloc[0]
        pred_gain = bool(best_pred["delta_log_loss_vs_v3"] < 0 and best_pred["delta_brier_vs_v3"] < 0)
        value_non_degrade = bool(float(best_val["profit"]) >= float(value[(value["scope"].eq(scope_name)) & (value["feature_group"].eq(BASELINE))]["profit"].iloc[0]) - 1.0)
        concentration_ok = bool(
            pd.isna(best_val["best_year_share"])
            or (float(best_val["best_year_share"]) < 0.60 and float(best_val["best_league_share"]) < 0.60)
        )
        stale_rows = stale[
            stale["scope"].eq(scope_name)
            & stale["feature_group"].eq(best_pred["feature_group"])
            & stale["segment"].isin(["latest_days_le_365", "exclude_after_2024_09_29"])
        ]
        stale_ok = bool(len(stale_rows) and stale_rows["rows"].max() >= 500)
        return pred_gain, value_non_degrade, concentration_ok, stale_ok

    top_pred, top_val, top_conc, top_stale = scope_ok("top5_only")
    full_pred, full_val, full_conc, full_stale = scope_ok("full_locked_scope_diagnostic")
    any_pred_gain = bool((summary[summary["feature_group"].isin([CORE, FULL])]["delta_log_loss_vs_v3"] < 0).any())
    if full_pred and full_val and full_conc and full_stale:
        return "v4_1_improves_general_locked_candidate"
    if top_pred and top_val and top_conc and top_stale:
        return "v4_1_improves_top5_branch"
    if any_pred_gain:
        return "v4_1_improves_predictive_only"
    return "v4_1_rejected_no_predictive_gain"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v3_raw = pd.read_csv(V3_MATRIX, low_memory=False)
    v4_raw = pd.read_csv(V4_MATRIX, low_memory=False)
    v4 = load_matrix(V4_MATRIX)
    group_cols = feature_column_groups(v4)

    value_preds: dict[tuple[str, str], pd.DataFrame] = {}
    for scope_name in SCOPES:
        base_ids: set[object] | None = None
        for label, cols in group_cols.items():
            pred = annual_predictions(v4, scope_name, label, cols)
            vp = add_understat_context(add_value_columns(pred, v4[scope_filter(v4, scope_name)].copy()), v4)
            if base_ids is None:
                base_ids = set(vp["match_id"])
            elif set(vp["match_id"]) != base_ids:
                raise RuntimeError(f"same-row mismatch scope={scope_name} feature_group={label}")
            value_preds[(scope_name, label)] = vp

    summary = pd.DataFrame([metrics_for_predictions(vp, scope_name, label) for (scope_name, label), vp in value_preds.items()])
    summary = add_deltas(summary)
    summary.to_csv(PREDICTIVE_CSV, index=False)

    edge = pd.concat([scoped_edge_buckets(vp, scope_name, label) for (scope_name, label), vp in value_preds.items()], ignore_index=True)
    edge.to_csv(EDGE_BUCKET_CSV, index=False)

    selected_by_key = {(scope_name, label): select_locked_schedule(vp, label) for (scope_name, label), vp in value_preds.items()}
    value = pd.DataFrame([selected_summary(sel, scope_name, label) for (scope_name, label), sel in selected_by_key.items()])
    value = add_value_deltas(value)
    value.to_csv(VALUE_CSV, index=False)

    year = pd.concat([selected_breakdown(sel, scope_name, label, "season_start_year") for (scope_name, label), sel in selected_by_key.items()], ignore_index=True)
    league = pd.concat([selected_breakdown(sel, scope_name, label, "league") for (scope_name, label), sel in selected_by_key.items()], ignore_index=True)
    year.to_csv(YEAR_CSV, index=False)
    league.to_csv(LEAGUE_CSV, index=False)

    nested = pd.concat(
        [nested_diagnostic(vp, scope_name, label) for (scope_name, label), vp in value_preds.items() if label in {CORE, FULL}],
        ignore_index=True,
        sort=False,
    )
    nested.to_csv(NESTED_CSV, index=False)

    stale = pd.concat(
        [staleness_metrics(vp, scope_name, label) for (scope_name, label), vp in value_preds.items() if label in {CORE, FULL}],
        ignore_index=True,
        sort=False,
    )
    stale.to_csv(STALENESS_CSV, index=False)

    checks = bug_checks(v3_raw, v4_raw, group_cols, selected_by_key)
    checks.to_csv(BUG_CHECKS_CSV, index=False)
    decision = classify_decision(summary, value, checks, stale)

    key_summary = summary[
        [
            "scope",
            "feature_group",
            "rows",
            "log_loss",
            "brier",
            "ece",
            "delta_log_loss_vs_v3",
            "delta_brier_vs_v3",
            "delta_ece_vs_v3",
            "both_understat_available_rows",
            "rows_after_understat_source_end",
            "latest_days_gt_365_rows",
            "latest_days_gt_730_rows",
        ]
    ]
    key_value = value[
        [
            "scope",
            "feature_group",
            "bets",
            "profit",
            "roi",
            "z",
            "delta_profit_vs_v3",
            "best_year_share",
            "best_league_share",
            "profit_ex_best_year",
            "profit_ex_best_league",
        ]
    ]
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V4.1 Understat vs V3 Same-Row Comparison",
                "",
                f"Decision: `{decision}`",
                "",
                "No broad model search, threshold search, value search, FBref data, closing-odds selection, current-match Understat feature use, or locked v3 candidate change was run. No confirmed edge is claimed.",
                "",
                "## Predictive Summary",
                md_table(key_summary, 40),
                "",
                "## Fixed Locked-Rule Value Replay",
                md_table(key_value, 40),
                "",
                "## Staleness Diagnostics",
                md_table(stale[["scope", "feature_group", "segment", "rows", "log_loss", "brier", "ece", "away_calibration_error"]], 60),
                "",
                "## Bug Checks",
                md_table(checks, 80),
                "",
            ]
        ),
        encoding="utf-8",
    )
    DECISION_MD.write_text(
        "\n".join(
            [
                "# V4.1 Understat Decision",
                "",
                f"Decision: `{decision}`",
                "",
                "This is a same-row diagnostic/top-five branch comparison against the locked v3 feature definition and fixed locked-rule schedule.",
                "No threshold search or confirmed edge claim was made.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    best_rows = value[value["feature_group"].isin([CORE, FULL])].sort_values("profit", ascending=False).head(3)
    print(
        {
            "decision": decision,
            "failed_checks": int(checks["status"].ne("pass").sum()),
            "top5_baseline_profit": round(float(value[(value["scope"].eq("top5_only")) & (value["feature_group"].eq(BASELINE))]["profit"].iloc[0]), 2),
            "top5_best_understat_profit": round(float(best_rows[best_rows["scope"].eq("top5_only")]["profit"].max()), 2),
            "full_baseline_profit": round(float(value[(value["scope"].eq("full_locked_scope_diagnostic")) & (value["feature_group"].eq(BASELINE))]["profit"].iloc[0]), 2),
            "full_best_understat_profit": round(float(best_rows[best_rows["scope"].eq("full_locked_scope_diagnostic")]["profit"].max()), 2),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
