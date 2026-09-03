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
from src.experiments.feature_matrix_v2_tm_1x2_value_review import (
    RULE_GRID,
    add_value_columns,
    nested_selection,
)


REPORT_DIR = Path("outputs/reports")
V2_MATRIX = Path("data/processed/features/football_feature_matrix_v2_transfermarkt_partial.csv")
V3_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
V2_LOCKED_PREDS = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_row_predictions.csv"
V2_LOCKED_BETS = REPORT_DIR / "feature_matrix_v2_tm_1x2_locked_selected_bets.csv"

REPORT_MD = REPORT_DIR / "feature_matrix_v3_clubelo_vs_v2_comparison.md"
PREDICTIVE_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_predictive_summary.csv"
VALUE_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_locked_rule_value_replay.csv"
YEAR_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_locked_rule_year_breakdown.csv"
LEAGUE_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_locked_rule_league_breakdown.csv"
NESTED_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_nested_diagnostic.csv"
STALENESS_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_staleness_diagnostics.csv"
EDGE_BUCKET_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_edge_bucket_calibration.csv"
BUG_CHECKS_CSV = REPORT_DIR / "feature_matrix_v3_clubelo_bug_checks.csv"
DECISION_MD = REPORT_DIR / "feature_matrix_v3_clubelo_decision.md"

SCOPE = "scope_C_top_divisions_ex_e1_e2_e3"
MODEL = "xgboost_market_residual_multiclass"
BASE_GROUP = "v2_baseline_tm_all"
FEATURE_GROUP = "x1_market_plus_tm_all"
LOCKED_RULE_SCHEDULE = {
    2021: ("away_edge_0.01_odds_1.5", 0.01, 1.5),
    2022: ("away_edge_0.01_odds_1.5", 0.01, 1.5),
    2023: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
    2024: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
    2025: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
}

CLUBELO_CORE = [
    "clubelo_home_rating",
    "clubelo_away_rating",
    "clubelo_diff",
    "clubelo_abs_diff",
    "clubelo_missing_home",
    "clubelo_missing_away",
    "clubelo_missing_both",
]
CLUBELO_STALENESS = [
    "clubelo_staleness_home",
    "clubelo_staleness_away",
    "clubelo_diff_minus_internal_elo_diff",
]


def z_score(profit: pd.Series) -> float:
    n = int(len(profit))
    if n <= 1:
        return 0.0
    sd = float(profit.std(ddof=1))
    return float(profit.sum() / (sd * math.sqrt(n))) if sd > 0 else 0.0


def md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def status(ok: bool) -> str:
    return "pass" if ok else "fail"


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


def filtered365_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    home_bad = out["clubelo_staleness_home"].gt(365) | out["clubelo_home_rating"].isna()
    away_bad = out["clubelo_staleness_away"].gt(365) | out["clubelo_away_rating"].isna()
    out.loc[home_bad, ["clubelo_home_rating", "clubelo_home_minus_internal_elo", "clubelo_staleness_home"]] = np.nan
    out.loc[away_bad, ["clubelo_away_rating", "clubelo_away_minus_internal_elo", "clubelo_staleness_away"]] = np.nan
    both_bad = home_bad | away_bad
    out.loc[both_bad, ["clubelo_diff", "clubelo_abs_diff", "clubelo_diff_minus_internal_elo_diff"]] = np.nan
    out["clubelo_missing_home"] = home_bad
    out["clubelo_missing_away"] = away_bad
    out["clubelo_missing_both"] = both_bad
    out["clubelo_both_ratings_available_flag"] = ~both_bad
    return out


def annual_predictions_locked(df: pd.DataFrame, label: str, cols: list[str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    scoped = df[scope_mask(df, SCOPE)].copy()
    preds = []
    yearly = []
    rng = np.random.default_rng(20260701)
    for year in TEST_YEARS:
        train = scoped[scoped["season_start_year"].astype(int).lt(year)].copy()
        test = scoped[scoped["season_start_year"].astype(int).eq(year)].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        print(f"locked_compare feature_group={label} test_year={year} rows={len(test)}", flush=True)
        prob = model_predict(MODEL, train, test, cols, rng)
        pred = test[["match_id", "match_date", "league", "season_start_year", "target_y"]].copy()
        pred[["prob_home", "prob_draw", "prob_away"]] = prob
        pred["scope"] = SCOPE
        pred["feature_group"] = label
        pred["model"] = MODEL
        pred["control"] = "none"
        preds.append(pred)
        yearly.append({"feature_group": label, "test_year": year, "rows": len(test)})
    return pd.concat(preds, ignore_index=True), yearly


def v2_predictions_as_value(df: pd.DataFrame) -> pd.DataFrame:
    locked = pd.read_csv(V2_LOCKED_PREDS)
    pred = locked.rename(
        columns={
            "model_prob_home": "prob_home",
            "model_prob_draw": "prob_draw",
            "model_prob_away": "prob_away",
            "fold_test_year": "test_year",
        }
    )[["match_id", "match_date", "league", "season_start_year", "target_y" if False else "target_outcome_1x2", "prob_home", "prob_draw", "prob_away"]].copy()
    pred["match_date"] = pd.to_datetime(pred["match_date"], errors="coerce")
    pred["target_y"] = pred["target_outcome_1x2"].map(CLASS_TO_INT).astype(int)
    pred = pred.drop(columns=["target_outcome_1x2"])
    pred["scope"] = SCOPE
    pred["feature_group"] = BASE_GROUP
    pred["model"] = MODEL
    pred["control"] = "none"
    return add_value_columns(pred, df[scope_mask(df, SCOPE)].copy())


def metrics_for_predictions(value_pred: pd.DataFrame, label: str) -> dict[str, object]:
    y = value_pred["target_y"].to_numpy(dtype=int)
    prob = normalize_probs(value_pred[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float))
    out: dict[str, object] = {
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
        "both_clubelo_ratings_rows": int(value_pred.get("clubelo_both_ratings_available_flag", pd.Series(False, index=value_pred.index)).fillna(False).astype(bool).sum()),
        "stale_gt_365_rows": int(((value_pred.get("clubelo_staleness_home", pd.Series(np.nan, index=value_pred.index)).gt(365)) | (value_pred.get("clubelo_staleness_away", pd.Series(np.nan, index=value_pred.index)).gt(365))).sum()),
        "stale_gt_730_rows": int(((value_pred.get("clubelo_staleness_home", pd.Series(np.nan, index=value_pred.index)).gt(730)) | (value_pred.get("clubelo_staleness_away", pd.Series(np.nan, index=value_pred.index)).gt(730))).sum()),
    }
    return out


def add_clubelo_context(value_pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "league",
        "season_start_year",
        "clubelo_home_rating",
        "clubelo_away_rating",
        "clubelo_staleness_home",
        "clubelo_staleness_away",
        "clubelo_both_ratings_available_flag",
        "clubelo_home_rating_date",
        "clubelo_away_rating_date",
        "target_outcome_1x2",
        "target_away_win",
    ]
    present = [c for c in cols if c in df.columns]
    return value_pred.merge(df[present], on=["match_id", "league", "season_start_year"], how="left", validate="many_to_one")


def edge_buckets(value_pred: pd.DataFrame, label: str) -> pd.DataFrame:
    out = value_pred.copy()
    model_p = out[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    market_p = out[["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].to_numpy(dtype=float)
    edge = model_p - market_p
    out["max_abs_edge"] = np.abs(edge).max(axis=1)
    bins = [-np.inf, 0.005, 0.01, 0.02, 0.04, np.inf]
    labels = ["<=0.5pp", "0.5-1pp", "1-2pp", "2-4pp", ">4pp"]
    out["edge_bucket"] = pd.cut(out["max_abs_edge"], bins=bins, labels=labels)
    rows = []
    for bucket, g in out.groupby("edge_bucket", dropna=False):
        prob = normalize_probs(g[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float))
        y = g["target_y"].to_numpy(dtype=int)
        rows.append(
            {
                "feature_group": label,
                "edge_bucket": str(bucket),
                "rows": int(len(g)),
                "mean_max_abs_edge": float(g["max_abs_edge"].mean()) if len(g) else np.nan,
                "accuracy": float((prob.argmax(axis=1) == y).mean()) if len(g) else np.nan,
                "log_loss": float(log_loss(y, prob, labels=[0, 1, 2])) if len(g) else np.nan,
                "away_mean_edge": float((g["prob_away"] - g["x1x2_avg_prob_away"]).mean()) if len(g) else np.nan,
                "away_observed_rate": float((g["target_y"].eq(2)).mean()) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_locked_schedule(value_pred: pd.DataFrame, label: str) -> pd.DataFrame:
    parts = []
    for year, (rule, edge, min_odds) in LOCKED_RULE_SCHEDULE.items():
        current = value_pred[value_pred["season_start_year"].astype(int).eq(year)].copy()
        selected = current[current["away_edge"].ge(edge) & current["x1x2_avg_odds_away"].ge(min_odds)].copy()
        selected["selected_rule"] = rule
        selected["side"] = "away"
        selected["profit"] = selected["away_profit"]
        selected["feature_group"] = label
        parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def bet_summary(selected: pd.DataFrame, label: str) -> dict[str, object]:
    bets = int(len(selected))
    profit = float(selected["profit"].sum()) if bets else 0.0
    best_year_profit = selected.groupby("season_start_year")["profit"].sum().max() if bets else 0.0
    best_league_profit = selected.groupby("league")["profit"].sum().max() if bets else 0.0
    best_year = selected.groupby("season_start_year")["profit"].sum().idxmax() if bets else ""
    best_league = selected.groupby("league")["profit"].sum().idxmax() if bets else ""
    return {
        "feature_group": label,
        "bets": bets,
        "profit": profit,
        "roi": profit / bets if bets else 0.0,
        "z": z_score(selected["profit"]) if bets else 0.0,
        "best_year": best_year,
        "best_league": best_league,
        "best_year_share": float(best_year_profit / profit) if profit > 0 else np.nan,
        "best_league_share": float(best_league_profit / profit) if profit > 0 else np.nan,
        "profit_ex_best_year": float(selected[selected["season_start_year"].ne(best_year)]["profit"].sum()) if bets else 0.0,
        "profit_ex_best_league": float(selected[~selected["league"].eq(best_league)]["profit"].sum()) if bets else 0.0,
    }


def breakdown(selected: pd.DataFrame, label: str, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in selected.groupby(group_col, dropna=False):
        rows.append(
            {
                "feature_group": label,
                group_col: key,
                "bets": int(len(g)),
                "profit": float(g["profit"].sum()),
                "roi": float(g["profit"].sum() / len(g)) if len(g) else 0.0,
                "z": z_score(g["profit"]),
            }
        )
    return pd.DataFrame(rows)


def staleness_metrics(value_pred: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    masks = {
        "all_2020_2025": pd.Series(True, index=value_pred.index),
        "exclude_2025": value_pred["season_start_year"].astype(int).ne(2025),
        "both_ratings_available": value_pred["clubelo_both_ratings_available_flag"].fillna(False).astype(bool) if "clubelo_both_ratings_available_flag" in value_pred else pd.Series(False, index=value_pred.index),
        "both_staleness_le_365": value_pred["clubelo_staleness_home"].le(365) & value_pred["clubelo_staleness_away"].le(365) if "clubelo_staleness_home" in value_pred else pd.Series(False, index=value_pred.index),
        "both_staleness_le_730": value_pred["clubelo_staleness_home"].le(730) & value_pred["clubelo_staleness_away"].le(730) if "clubelo_staleness_home" in value_pred else pd.Series(False, index=value_pred.index),
    }
    for segment, mask in masks.items():
        g = value_pred[mask].copy()
        if g.empty:
            rows.append({"feature_group": label, "segment": segment, "rows": 0})
        else:
            m = metrics_for_predictions(g, label)
            rows.append({"feature_group": label, "segment": segment, **m})
    return pd.DataFrame(rows)


def bug_checks(v2_raw: pd.DataFrame, v3_raw: pd.DataFrame, feature_cols: dict[str, list[str]], selected_by_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    original_cols = list(v2_raw.columns)
    unchanged_bad = []
    for col in original_cols:
        left = v2_raw[col]
        right = v3_raw[col]
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
    common_unchanged = len(unchanged_bad) == 0
    home_date = pd.to_datetime(v3_raw["clubelo_home_rating_date"], errors="coerce")
    away_date = pd.to_datetime(v3_raw["clubelo_away_rating_date"], errors="coerce")
    match_date = pd.to_datetime(v3_raw["match_date"], errors="coerce")
    home_exists = v3_raw["clubelo_home_rating"].notna()
    away_exists = v3_raw["clubelo_away_rating"].notna()
    all_cols = sorted(set(sum(feature_cols.values(), [])))
    bad_names = [c for c in all_cols if pd.api.types.is_object_dtype(v3_raw[c]) and re.search(r"team|club|name", c, re.I)]
    bad_post = [c for c in all_cols if re.search(r"FTHome|FTAway|FTResult|HomeShots|AwayShots|HomeTarget|AwayTarget|HTHome|HTAway|HTResult", c, re.I)]
    closing = [c for c in all_cols if re.search(r"closing|_close|close_", c, re.I)]
    duplicate_selected = sum(int(s.duplicated(["match_id"]).sum()) for s in selected_by_group.values())
    non_away = sum(int((~s.get("side", pd.Series("away", index=s.index)).eq("away")).sum()) for s in selected_by_group.values())
    profit_bad = 0
    odds_bad = 0
    for s in selected_by_group.values():
        expected = np.where(s["target_y"].eq(2), s["x1x2_avg_odds_away"] - 1.0, -1.0)
        profit_bad += int((np.round(s["profit"].to_numpy(dtype=float), 12) != np.round(expected, 12)).sum())
        odds_bad += int((~s["x1x2_avg_odds_away"].gt(1.0)).sum())
    rows = [
        ("v3_row_count_equals_v2", len(v2_raw) == len(v3_raw), len(v3_raw), f"v2={len(v2_raw)} v3={len(v3_raw)}"),
        ("original_v2_columns_unchanged", common_unchanged, len(unchanged_bad), "|".join(unchanged_bad[:20])),
        ("no_clubelo_rating_date_gte_match_date", bool((home_date[home_exists] < match_date[home_exists]).all() and (away_date[away_exists] < match_date[away_exists]).all()), int(home_exists.sum() + away_exists.sum()), ""),
        ("no_same_day_clubelo_joins", int(((home_date == match_date) & home_exists).sum() + ((away_date == match_date) & away_exists).sum()) == 0, int(((home_date == match_date) & home_exists).sum() + ((away_date == match_date) & away_exists).sum()), ""),
        ("no_clubelo_matches_postmatch_columns_used", len(bad_post) == 0, len(bad_post), "|".join(bad_post)),
        ("no_team_club_name_string_features_used", len(bad_names) == 0, len(bad_names), "|".join(bad_names)),
        ("no_closing_odds_used_for_selection", len(closing) == 0, len(closing), "|".join(closing)),
        ("selected_away_bets_use_away_side", non_away == 0, non_away, ""),
        ("profit_formula_correct", profit_bad == 0, profit_bad, ""),
        ("no_impossible_selected_odds", odds_bad == 0, odds_bad, ""),
        ("no_duplicate_selected_matches_within_group", duplicate_selected == 0, duplicate_selected, ""),
    ]
    return pd.DataFrame([{"check": n, "status": status(bool(ok)), "count": int(count), "detail": detail} for n, ok, count, detail in rows])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v2_raw_full = pd.read_csv(V2_MATRIX, low_memory=False)
    v3_raw_full = pd.read_csv(V3_MATRIX, low_memory=False)
    v2 = load_matrix(V2_MATRIX)
    v3 = load_matrix(V3_MATRIX)
    v3_filtered = filtered365_frame(v3)
    tm_cols = feature_groups(v3)[FEATURE_GROUP]
    group_cols = {
        BASE_GROUP: tm_cols,
        "v3_tm_plus_clubelo_core": sorted(set(tm_cols + CLUBELO_CORE)),
        "v3_tm_plus_clubelo_core_staleness": sorted(set(tm_cols + CLUBELO_CORE + CLUBELO_STALENESS)),
        "v3_tm_plus_clubelo_filtered365_diagnostic": sorted(set(tm_cols + CLUBELO_CORE + CLUBELO_STALENESS)),
    }

    value_preds: dict[str, pd.DataFrame] = {}
    value_preds[BASE_GROUP] = v2_predictions_as_value(v2)
    for label in ["v3_tm_plus_clubelo_core", "v3_tm_plus_clubelo_core_staleness"]:
        pred, _ = annual_predictions_locked(v3, label, group_cols[label])
        value_preds[label] = add_clubelo_context(add_value_columns(pred, v3[scope_mask(v3, SCOPE)].copy()), v3)
    label = "v3_tm_plus_clubelo_filtered365_diagnostic"
    pred, _ = annual_predictions_locked(v3_filtered, label, group_cols[label])
    value_preds[label] = add_clubelo_context(add_value_columns(pred, v3_filtered[scope_mask(v3_filtered, SCOPE)].copy()), v3_filtered)
    value_preds[BASE_GROUP] = add_clubelo_context(value_preds[BASE_GROUP], v3)

    base_ids = set(value_preds[BASE_GROUP]["match_id"])
    for label, vp in value_preds.items():
        if set(vp["match_id"]) != base_ids:
            raise RuntimeError(f"same-row universe mismatch for {label}")

    summary = pd.DataFrame([metrics_for_predictions(vp, label) for label, vp in value_preds.items()])
    base = summary[summary["feature_group"].eq(BASE_GROUP)].iloc[0]
    for metric in ["log_loss", "brier", "ece", "class_log_loss_away", "away_brier"]:
        summary[f"delta_{metric}_vs_v2"] = summary[metric] - float(base[metric])
    summary.to_csv(PREDICTIVE_CSV, index=False)

    edge = pd.concat([edge_buckets(vp, label) for label, vp in value_preds.items()], ignore_index=True)
    edge.to_csv(EDGE_BUCKET_CSV, index=False)

    selected_by_group = {label: select_locked_schedule(vp, label) for label, vp in value_preds.items()}
    value = pd.DataFrame([bet_summary(sel, label) for label, sel in selected_by_group.items()])
    base_value = value[value["feature_group"].eq(BASE_GROUP)].iloc[0]
    for metric in ["bets", "profit", "roi", "z"]:
        value[f"delta_{metric}_vs_v2"] = value[metric] - float(base_value[metric])
    value.to_csv(VALUE_CSV, index=False)
    year = pd.concat([breakdown(sel, label, "season_start_year") for label, sel in selected_by_group.items()], ignore_index=True)
    league = pd.concat([breakdown(sel, label, "league") for label, sel in selected_by_group.items()], ignore_index=True)
    year.to_csv(YEAR_CSV, index=False)
    league.to_csv(LEAGUE_CSV, index=False)

    nested_rows = []
    for label in ["v3_tm_plus_clubelo_core", "v3_tm_plus_clubelo_core_staleness", "v3_tm_plus_clubelo_filtered365_diagnostic"]:
        nested, _ = nested_selection(value_preds[label], SCOPE, MODEL, label)
        nested_rows.append(nested.assign(diagnostic_only=True))
    nested_diag = pd.concat(nested_rows, ignore_index=True, sort=False)
    nested_diag.to_csv(NESTED_CSV, index=False)

    stale = pd.concat(
        [staleness_metrics(value_preds[label], label) for label in ["v3_tm_plus_clubelo_core", "v3_tm_plus_clubelo_core_staleness", "v3_tm_plus_clubelo_filtered365_diagnostic"]],
        ignore_index=True,
        sort=False,
    )
    stale.to_csv(STALENESS_CSV, index=False)

    checks = bug_checks(v2_raw_full, v3_raw_full, group_cols, selected_by_group)
    checks.to_csv(BUG_CHECKS_CSV, index=False)

    decision_eligible = value[
        value["feature_group"].isin(["v3_tm_plus_clubelo_core", "v3_tm_plus_clubelo_core_staleness"])
    ].copy()
    best_v3 = decision_eligible.sort_values("profit", ascending=False).iloc[0]
    best_v3_any = value[value["feature_group"].ne(BASE_GROUP)].sort_values("profit", ascending=False).iloc[0]
    pred_best = summary[summary["feature_group"].eq(best_v3["feature_group"])].iloc[0]
    bug_ok = checks["status"].eq("pass").all()
    predictive_gain = bool(pred_best["delta_log_loss_vs_v2"] < 0 and pred_best["delta_brier_vs_v2"] < 0)
    value_non_degrade = bool(float(best_v3["profit"]) >= float(base_value["profit"]) - 1.0)
    value_gain = bool(float(best_v3["profit"]) > float(base_value["profit"]))
    concentration_ok = bool(float(best_v3["best_year_share"]) < 0.6 and float(best_v3["best_league_share"]) < 0.6)
    stale_best = stale[(stale["feature_group"].eq(best_v3["feature_group"])) & (stale["segment"].eq("both_staleness_le_365"))]
    stale_ok = len(stale_best) > 0 and int(stale_best["rows"].iloc[0]) > 1000
    if not bug_ok:
        decision = "v3_rejected_bug_or_leakage"
    elif not predictive_gain:
        decision = "v3_rejected_no_predictive_gain"
    elif value_gain and value_non_degrade and concentration_ok and stale_ok:
        decision = "v3_improves_locked_candidate"
    elif predictive_gain:
        decision = "v3_improves_predictive_only" if value_non_degrade else "v3_diagnostic_only"
    else:
        decision = "v3_diagnostic_only"

    report_lines = [
        "# V3 ClubElo vs V2 Locked Candidate Comparison",
        "",
        f"Decision: `{decision}`",
        "",
        "No broad model search, threshold search, value search, closing-odds selection, or locked v2 candidate change was run. No confirmed edge is claimed.",
        "",
        "## Predictive Summary",
        md_table(summary[["feature_group", "rows", "log_loss", "brier", "ece", "delta_log_loss_vs_v2", "delta_brier_vs_v2", "delta_ece_vs_v2", "both_clubelo_ratings_rows", "stale_gt_365_rows", "stale_gt_730_rows"]], 10),
        "",
        "## Locked Rule Schedule Value Replay",
        md_table(value[["feature_group", "bets", "profit", "roi", "z", "delta_profit_vs_v2", "best_year_share", "best_league_share", "profit_ex_best_year", "profit_ex_best_league"]], 10),
        "",
        "## Staleness Diagnostics",
        md_table(stale[["feature_group", "segment", "rows", "log_loss", "brier", "ece", "away_calibration_error"]], 30),
        "",
        "## Bug Checks",
        md_table(checks, 80),
        "",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
    DECISION_MD.write_text(
        "\n".join(
            [
                "# V3 ClubElo Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"Best decision-eligible v3 locked-rule group by profit: `{best_v3['feature_group']}`.",
                f"Best decision-eligible v3 profit: {float(best_v3['profit']):.2f}u vs v2 {float(base_value['profit']):.2f}u.",
                f"Best diagnostic-only v3 profit: {float(best_v3_any['profit']):.2f}u from `{best_v3_any['feature_group']}`.",
                f"Best v3 predictive delta log loss vs v2: {float(pred_best['delta_log_loss_vs_v2']):.6f}.",
                "",
                "No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        {
            "decision": decision,
            "best_v3_group": str(best_v3["feature_group"]),
            "best_v3_any_group": str(best_v3_any["feature_group"]),
            "v2_profit": round(float(base_value["profit"]), 2),
            "best_v3_profit": round(float(best_v3["profit"]), 2),
            "best_v3_delta_log_loss": round(float(pred_best["delta_log_loss_vs_v2"]), 8),
            "failed_checks": int(checks["status"].ne("pass").sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
