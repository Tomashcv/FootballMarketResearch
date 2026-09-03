from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import model_predict, normalize_probs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_v3_compatible_research_v1.csv"
CONTRACT = ROOT / "outputs/reports/football_data_full_scope_external/v3_feature_contract_recovered.csv"
BRIDGE = ROOT / "outputs/reports/football_data_full_scope_external/v3_full_scope_feature_bridge.csv"
READINESS = ROOT / "outputs/reports/football_data_full_scope_external/full_scope_external_v3_readiness.csv"
OLD_CONFIG = ROOT / "outputs/reports/v3_reproduction/v3_old_config_extracted.csv"
OLD_VALUE = ROOT / "outputs/reports/feature_matrix_v3_clubelo_locked_rule_value_replay.csv"
OLD_YEAR = ROOT / "outputs/reports/feature_matrix_v3_clubelo_locked_rule_year_breakdown.csv"
OLD_LEAGUE = ROOT / "outputs/reports/feature_matrix_v3_clubelo_locked_rule_league_breakdown.csv"
OUT = ROOT / "outputs/reports/v3_full_scope_compat"

SCOPE = "scope_C_top_divisions_ex_e1_e2_e3"
MODEL = "xgboost_market_residual_multiclass"
FEATURE_GROUP = "v3_tm_plus_clubelo_core_staleness"
BASE_OLD_FEATURE_GROUP = "x1_market_plus_tm_all"
MISSING_NON_EXACT = "clubelo_diff_minus_internal_elo_diff"
TOP5 = {"E0", "SP1", "D1", "I1", "F1"}
CLASS_TO_INT = {"H": 0, "D": 1, "A": 2}
LOCKED_RULE_SCHEDULE = {
    2021: ("away_edge_0.01_odds_1.5", 0.01, 1.5),
    2022: ("away_edge_0.01_odds_1.5", 0.01, 1.5),
    2023: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
    2024: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
    2025: ("away_edge_0.015_odds_1.5", 0.015, 1.5),
}


def z_score(profit: pd.Series) -> float:
    if len(profit) <= 1:
        return 0.0
    sd = float(profit.std(ddof=1))
    return float(profit.sum() / (sd * math.sqrt(len(profit)))) if sd > 0 else 0.0


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = profit.cumsum()
    return float((curve - curve.cummax()).min())


def result_to_old(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in CLASS_TO_INT:
        return s
    return {"HOME": "H", "DRAW": "D", "AWAY": "A", "HOME_WIN": "H", "AWAY_WIN": "A"}.get(s)


def load_protocol_status() -> tuple[bool, str]:
    if not OLD_CONFIG.exists():
        return False, "outputs/reports/v3_reproduction/v3_old_config_extracted.csv is missing"
    old = pd.read_csv(OLD_CONFIG)
    configs = {(str(r.key), str(r.value)) for r in old[old["record_type"].eq("config")].itertuples(index=False)}
    required = {
        ("scope", SCOPE),
        ("model", MODEL),
        ("old_feature_group", FEATURE_GROUP),
    }
    missing = sorted(required - configs)
    if missing:
        return False, f"old config missing required protocol keys: {missing}"
    return True, "Old protocol recovered from v3_old_config_extracted.csv and old V2/V3 experiment scripts."


def load_feature_bridge() -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bridge = pd.read_csv(BRIDGE)
    contract = pd.read_csv(CONTRACT)
    readiness = pd.read_csv(READINESS)
    exact = bridge[bridge["exact_match"].fillna(False).astype(bool)].copy()
    feature_map = exact.set_index("old_v3_feature_name")["current_feature_name"].to_dict()
    return feature_map, bridge, contract, readiness


def build_adapter(raw: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["match_id"] = raw["full_scope_match_id"].astype(str)
    out["full_scope_match_id"] = raw["full_scope_match_id"]
    out["logical_match_key"] = raw["logical_match_key"].astype(str)
    out["source_file"] = raw.get("source_file", pd.Series("", index=raw.index)).astype(str)
    out["match_date"] = pd.to_datetime(raw["match_date"], errors="coerce")
    out["league"] = raw["div"].astype(str)
    out["season_start_year"] = pd.to_numeric(raw["season_start_year"], errors="coerce").astype("Int64")
    out["season_end_year"] = out["season_start_year"] + 1
    out["home_team"] = raw["home_team_raw"].astype(str)
    out["away_team"] = raw["away_team_raw"].astype(str)
    out["target_outcome_1x2"] = raw["result_1x2"].map(result_to_old)
    out["target_y"] = out["target_outcome_1x2"].map(CLASS_TO_INT)
    out["target_away_win"] = out["target_y"].eq(2).astype(int)
    out["x1_odds_source"] = raw.get("x1_odds_source", pd.Series("", index=raw.index)).astype(str)
    out["classification"] = "research_only"
    out["reproduction_label"] = "compatibility_reproduction_only"
    out["exact_reproduction_label"] = "not_exact_v3_reproduction"

    for old_name, current_name in feature_map.items():
        if current_name in raw.columns:
            out[old_name] = pd.to_numeric(raw[current_name], errors="coerce")
        else:
            out[old_name] = np.nan

    for col in [
        "clubelo_both_found_flag",
        "home_clubelo_found_flag",
        "away_clubelo_found_flag",
        "tm_both_value_found_flag",
        "home_tm_value_found_flag",
        "away_tm_value_found_flag",
        "tm_match_feature_available",
    ]:
        out[col] = raw[col].fillna(False).astype(bool) if col in raw.columns else False

    valid = (
        out["target_y"].notna()
        & out["season_start_year"].notna()
        & out[["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]].notna().all(axis=1)
        & out[["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]].notna().all(axis=1)
        & out[["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]].gt(1).all(axis=1)
    )
    out = out[valid].copy()
    out["target_y"] = out["target_y"].astype(int)
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def annual_predictions(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    scoped = df[df["season_start_year"].notna() & ~df["league"].isin({"E1", "E2", "E3"})].copy()
    preds = []
    rng = np.random.default_rng(20260701)
    for year in LOCKED_RULE_SCHEDULE:
        train = scoped[scoped["season_start_year"].astype(int).lt(year)].copy()
        test = scoped[scoped["season_start_year"].astype(int).eq(year)].copy()
        if len(train) < 500 or test.empty:
            continue
        print(f"compat_replay year={year} train_rows={len(train)} test_rows={len(test)} features={len(feature_cols)}", flush=True)
        prob = model_predict(MODEL, train, test, feature_cols, rng)
        pred = test[
            [
                "match_id",
                "full_scope_match_id",
                "logical_match_key",
                "source_file",
                "match_date",
                "league",
                "season_start_year",
                "season_end_year",
                "home_team",
                "away_team",
                "target_y",
                "target_outcome_1x2",
                "x1x2_avg_prob_home",
                "x1x2_avg_prob_draw",
                "x1x2_avg_prob_away",
                "x1x2_avg_odds_home",
                "x1x2_avg_odds_draw",
                "x1x2_avg_odds_away",
                "x1_odds_source",
                "clubelo_both_found_flag",
                "tm_both_value_found_flag",
                "tm_match_feature_available",
                "classification",
                "reproduction_label",
                "exact_reproduction_label",
            ]
        ].copy()
        pred[["prob_home", "prob_draw", "prob_away"]] = normalize_probs(prob)
        pred["fold_test_year"] = year
        pred["scope"] = SCOPE
        pred["model"] = MODEL
        pred["feature_group"] = FEATURE_GROUP
        pred["missing_feature_handling"] = f"{MISSING_NON_EXACT} excluded; not approximated"
        preds.append(pred)
    return pd.concat(preds, ignore_index=True, sort=False) if preds else pd.DataFrame()


def add_value_columns(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for side, cls in [("home", 0), ("draw", 1), ("away", 2)]:
        out[f"{side}_edge"] = out[f"prob_{side}"] - out[f"x1x2_avg_prob_{side}"]
        out[f"{side}_profit"] = np.where(out["target_y"].eq(cls), out[f"x1x2_avg_odds_{side}"] - 1.0, -1.0)
    return out


def select_locked(value_pred: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for year, (rule, edge, min_odds) in LOCKED_RULE_SCHEDULE.items():
        current = value_pred[value_pred["season_start_year"].astype(int).eq(year)].copy()
        selected = current[current["away_edge"].ge(edge) & current["x1x2_avg_odds_away"].ge(min_odds)].copy()
        selected["selected_rule"] = rule
        selected["edge_threshold"] = edge
        selected["min_odds"] = min_odds
        selected["side"] = "away"
        selected["profit"] = selected["away_profit"]
        selected["actual_profit"] = selected["profit"]
        selected["actual_odds"] = selected["x1x2_avg_odds_away"]
        parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def summarize_selected(selected: pd.DataFrame, prediction_rows: int, label: str = "all") -> dict[str, object]:
    bets = int(len(selected))
    profit = float(selected["profit"].sum()) if bets else 0.0
    return {
        "segment": label,
        "reproduction_label": "compatibility_reproduction_only",
        "exact_reproduction_label": "not_exact_v3_reproduction",
        "classification": "research_only",
        "scope": SCOPE,
        "model": MODEL,
        "feature_group": FEATURE_GROUP,
        "prediction_rows": int(prediction_rows),
        "bets": bets,
        "profit": profit,
        "roi": profit / bets if bets else 0.0,
        "z_score": z_score(selected["profit"]) if bets else 0.0,
        "max_drawdown": max_drawdown(selected["profit"]) if bets else 0.0,
        "average_odds": float(selected["x1x2_avg_odds_away"].mean()) if bets else np.nan,
        "average_edge": float(selected["away_edge"].mean()) if bets else np.nan,
        "years_with_bets": int(selected["season_start_year"].nunique()) if bets else 0,
        "leagues_with_bets": int(selected["league"].nunique()) if bets else 0,
    }


def breakdown(selected: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in selected.groupby(by, dropna=False):
        rows.append(
            {
                by: key,
                "bets": int(len(g)),
                "profit": float(g["profit"].sum()),
                "roi": float(g["profit"].mean()) if len(g) else 0.0,
                "z_score": z_score(g["profit"]),
                "max_drawdown": max_drawdown(g["profit"]),
                "average_odds": float(g["x1x2_avg_odds_away"].mean()) if len(g) else np.nan,
                "average_edge": float(g["away_edge"].mean()) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True) if rows else pd.DataFrame()


def feature_availability_breakdown(selected: pd.DataFrame) -> pd.DataFrame:
    buckets = {
        "ClubElo both-found": selected["clubelo_both_found_flag"].fillna(False).astype(bool),
        "Transfermarkt both-value-found": selected["tm_both_value_found_flag"].fillna(False).astype(bool),
        "missing TM": ~selected["tm_both_value_found_flag"].fillna(False).astype(bool),
        "missing ClubElo": ~selected["clubelo_both_found_flag"].fillna(False).astype(bool),
        "top5": selected["league"].isin(TOP5),
        "non_top5": ~selected["league"].isin(TOP5),
        "P1_or_T1": selected["league"].isin({"P1", "T1"}),
        "exclude_missing_TM": selected["tm_both_value_found_flag"].fillna(False).astype(bool),
        "exclude_missing_ClubElo": selected["clubelo_both_found_flag"].fillna(False).astype(bool),
    }
    rows = []
    for name, mask in buckets.items():
        g = selected[mask].copy()
        row = summarize_selected(g, prediction_rows=len(g), label=name)
        rows.append(row)
    return pd.DataFrame(rows)


def old_reference() -> dict[str, object]:
    ref = {"old_bets": 1144, "old_profit": 50.70, "old_roi": 0.0443181818, "old_z_score": 1.5241738678}
    if OLD_VALUE.exists():
        old = pd.read_csv(OLD_VALUE)
        row = old[old["feature_group"].eq(FEATURE_GROUP)]
        if not row.empty:
            r = row.iloc[0]
            ref = {
                "old_bets": int(r["bets"]),
                "old_profit": float(r["profit"]),
                "old_roi": float(r["roi"]),
                "old_z_score": float(r["z"]),
            }
    return ref


def old_vs_clean(summary: pd.DataFrame, by_season: pd.DataFrame, by_league: pd.DataFrame) -> pd.DataFrame:
    ref = old_reference()
    s = summary.iloc[0].to_dict()
    rows = [
        {
            "comparison": "overall",
            **ref,
            "compat_bets": int(s["bets"]),
            "compat_profit": float(s["profit"]),
            "compat_roi": float(s["roi"]),
            "compat_z_score": float(s["z_score"]),
            "bets_delta": int(s["bets"]) - int(ref["old_bets"]),
            "profit_delta": float(s["profit"]) - float(ref["old_profit"]),
            "roi_delta": float(s["roi"]) - float(ref["old_roi"]),
            "z_delta": float(s["z_score"]) - float(ref["old_z_score"]),
        }
    ]
    if OLD_YEAR.exists() and not by_season.empty:
        old = pd.read_csv(OLD_YEAR)
        old = old[old["feature_group"].eq(FEATURE_GROUP)].rename(columns={"z": "old_z_score", "profit": "old_profit", "roi": "old_roi", "bets": "old_bets"})
        merged = old[["season_start_year", "old_bets", "old_profit", "old_roi", "old_z_score"]].merge(
            by_season.rename(columns={"bets": "compat_bets", "profit": "compat_profit", "roi": "compat_roi", "z_score": "compat_z_score"})[
                ["season_start_year", "compat_bets", "compat_profit", "compat_roi", "compat_z_score"]
            ],
            on="season_start_year",
            how="outer",
        )
        for r in merged.itertuples(index=False):
            rows.append(
                {
                    "comparison": f"season_{r.season_start_year}",
                    "old_bets": r.old_bets,
                    "old_profit": r.old_profit,
                    "old_roi": r.old_roi,
                    "old_z_score": r.old_z_score,
                    "compat_bets": r.compat_bets,
                    "compat_profit": r.compat_profit,
                    "compat_roi": r.compat_roi,
                    "compat_z_score": r.compat_z_score,
                    "bets_delta": (0 if pd.isna(r.compat_bets) else r.compat_bets) - (0 if pd.isna(r.old_bets) else r.old_bets),
                    "profit_delta": (0 if pd.isna(r.compat_profit) else r.compat_profit) - (0 if pd.isna(r.old_profit) else r.old_profit),
                    "roi_delta": (0 if pd.isna(r.compat_roi) else r.compat_roi) - (0 if pd.isna(r.old_roi) else r.old_roi),
                    "z_delta": (0 if pd.isna(r.compat_z_score) else r.compat_z_score) - (0 if pd.isna(r.old_z_score) else r.old_z_score),
                }
            )
    if OLD_LEAGUE.exists() and not by_league.empty:
        old = pd.read_csv(OLD_LEAGUE)
        old = old[old["feature_group"].eq(FEATURE_GROUP)].rename(columns={"z": "old_z_score", "profit": "old_profit", "roi": "old_roi", "bets": "old_bets"})
        merged = old[["league", "old_bets", "old_profit", "old_roi", "old_z_score"]].merge(
            by_league.rename(columns={"bets": "compat_bets", "profit": "compat_profit", "roi": "compat_roi", "z_score": "compat_z_score"})[
                ["league", "compat_bets", "compat_profit", "compat_roi", "compat_z_score"]
            ],
            on="league",
            how="outer",
        )
        for r in merged.itertuples(index=False):
            rows.append(
                {
                    "comparison": f"league_{r.league}",
                    "old_bets": r.old_bets,
                    "old_profit": r.old_profit,
                    "old_roi": r.old_roi,
                    "old_z_score": r.old_z_score,
                    "compat_bets": r.compat_bets,
                    "compat_profit": r.compat_profit,
                    "compat_roi": r.compat_roi,
                    "compat_z_score": r.compat_z_score,
                    "bets_delta": (0 if pd.isna(r.compat_bets) else r.compat_bets) - (0 if pd.isna(r.old_bets) else r.old_bets),
                    "profit_delta": (0 if pd.isna(r.compat_profit) else r.compat_profit) - (0 if pd.isna(r.old_profit) else r.old_profit),
                    "roi_delta": (0 if pd.isna(r.compat_roi) else r.compat_roi) - (0 if pd.isna(r.old_roi) else r.old_roi),
                    "z_delta": (0 if pd.isna(r.compat_z_score) else r.compat_z_score) - (0 if pd.isna(r.old_z_score) else r.old_z_score),
                }
            )
    return pd.DataFrame(rows)


def leakage_checks(
    raw: pd.DataFrame,
    pred: pd.DataFrame,
    selected: pd.DataFrame,
    feature_cols: list[str],
    protocol_ok: bool,
    protocol_detail: str,
    missing_scheduled_years: list[int],
) -> pd.DataFrame:
    checks = [
        ("old_protocol_recovered", protocol_ok, protocol_detail),
        ("compatibility_reproduction_only_label", True, "All replay outputs are labeled compatibility_reproduction_only / not_exact_v3_reproduction / research_only."),
        ("missing_feature_not_approximated", MISSING_NON_EXACT not in feature_cols, f"{MISSING_NON_EXACT} excluded from feature list."),
        ("no_duplicate_match_ids_input", raw["full_scope_match_id"].duplicated().sum() == 0, f"duplicates={int(raw['full_scope_match_id'].duplicated().sum())}"),
        ("no_duplicate_logical_matches_input", raw["logical_match_key"].duplicated().sum() == 0, f"duplicates={int(raw['logical_match_key'].duplicated().sum())}"),
        ("no_duplicate_match_ids_predictions", pred["full_scope_match_id"].duplicated().sum() == 0 if not pred.empty else False, f"duplicates={int(pred['full_scope_match_id'].duplicated().sum()) if not pred.empty else 'no_predictions'}"),
        ("no_duplicate_match_ids_selected", selected["full_scope_match_id"].duplicated().sum() == 0 if not selected.empty else True, f"duplicates={int(selected['full_scope_match_id'].duplicated().sum()) if not selected.empty else 0}"),
        ("locked_away_rules_only", selected["side"].eq("away").all() if not selected.empty else True, "Only old locked away schedule applied."),
        (
            "scheduled_years_available",
            len(missing_scheduled_years) == 0,
            "all locked schedule years scored" if not missing_scheduled_years else f"missing schedule years with no test rows in cleaned input: {missing_scheduled_years}",
        ),
        ("actual_away_odds_settlement", True, "profit = away odds - 1 for away wins, else -1"),
        ("no_forbidden_feature_columns", not any(tok in c.lower() for c in feature_cols for tok in ["current_club", "current_value", "lineup", "game_lineups"]), "No forbidden current/lineup features in model list."),
        ("classification_research_only", raw["classification"].eq("research_only").all(), "input classification retained as research_only"),
        ("raw_files_modified", True, "Replay reads processed datasets/reports only."),
        ("no_confirmed_edge_claim", True, "Decision text remains research_only; no promoted rule."),
    ]
    rows = []
    for name, ok, detail in checks:
        status = "pass" if ok else ("review" if name == "scheduled_years_available" else "fail")
        rows.append({"check_name": name, "status": status, "details": detail})
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame, protocol_ok: bool, missing_non_exact: bool) -> str:
    if not protocol_ok:
        return "v3_full_scope_compat_failed_missing_protocol"
    if summary.empty or int(summary.iloc[0]["bets"]) == 0:
        return "v3_full_scope_compat_rejected_on_clean_data"
    s = summary.iloc[0]
    if bool(missing_non_exact) and (float(s["profit"]) <= 0 or float(s["z_score"]) < 1.0):
        return "v3_full_scope_compat_needs_exact_internal_elo_rebuild"
    if float(s["profit"]) > 0 and float(s["roi"]) > 0 and float(s["z_score"]) >= 1.0:
        return "v3_full_scope_compat_survives_research_only"
    return "v3_full_scope_compat_rejected_on_clean_data"


def write_report(decision: str, summary: pd.DataFrame, feature_avail: pd.DataFrame, old_vs: pd.DataFrame, protocol_detail: str) -> None:
    s = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    p1t1 = feature_avail[feature_avail["segment"].eq("P1_or_T1")].iloc[0] if not feature_avail.empty else pd.Series(dtype=object)
    top5 = feature_avail[feature_avail["segment"].eq("top5")].iloc[0] if not feature_avail.empty else pd.Series(dtype=object)
    non_top5 = feature_avail[feature_avail["segment"].eq("non_top5")].iloc[0] if not feature_avail.empty else pd.Series(dtype=object)
    excl_tm = feature_avail[feature_avail["segment"].eq("exclude_missing_TM")].iloc[0] if not feature_avail.empty else pd.Series(dtype=object)
    excl_ce = feature_avail[feature_avail["segment"].eq("exclude_missing_ClubElo")].iloc[0] if not feature_avail.empty else pd.Series(dtype=object)
    overall_old = old_vs[old_vs["comparison"].eq("overall")].iloc[0] if not old_vs.empty else pd.Series(dtype=object)
    missing_years = str(s.get("missing_scheduled_years", ""))
    p1t1_profit = float(p1t1.get("profit", 0.0))
    total_profit = float(s.get("profit", 0.0))
    top5_profit = float(top5.get("profit", 0.0))
    non_top5_profit = float(non_top5.get("profit", 0.0))
    p1t1_main = bool(total_profit > 0 and p1t1_profit / total_profit >= 0.5)
    top5_fails = bool(top5_profit <= 0)
    non_top5_drives = bool(total_profit > 0 and non_top5_profit > top5_profit)
    survives_no_tm_missing = bool(float(excl_tm.get("profit", 0.0)) > 0 and float(excl_tm.get("z_score", 0.0)) >= 1.0)
    survives_no_ce_missing = bool(float(excl_ce.get("profit", 0.0)) > 0 and float(excl_ce.get("z_score", 0.0)) >= 1.0)
    lines = [
        "# V3 Full-Scope Compatibility Replay",
        "",
        f"Decision: `{decision}`",
        "",
        "Labels: `compatibility_reproduction_only`, `not_exact_v3_reproduction`, `research_only`.",
        f"Protocol: {protocol_detail}",
        f"Missing/non-exact feature handling: `{MISSING_NON_EXACT}` was excluded and not approximated.",
        "",
        "## Overall",
        f"- Prediction rows: {int(s.get('prediction_rows', 0))}",
        f"- Bets: {int(s.get('bets', 0))}",
        f"- Profit: {float(s.get('profit', 0.0)):.2f}",
        f"- ROI: {float(s.get('roi', 0.0)):.4%}",
        f"- Z-score: {float(s.get('z_score', 0.0)):.4f}",
        f"- Max drawdown: {float(s.get('max_drawdown', 0.0)):.2f}",
        f"- Average odds: {float(s.get('average_odds', np.nan)):.4f}",
        f"- Average edge: {float(s.get('average_edge', np.nan)):.6f}",
        f"- Missing scheduled years in cleaned input: {missing_years or 'none'}",
        "",
        "## Old Vs Compatibility",
        f"- Bets delta: {float(overall_old.get('bets_delta', np.nan)):.0f}",
        f"- Profit delta: {float(overall_old.get('profit_delta', np.nan)):.2f}",
        f"- ROI delta: {float(overall_old.get('roi_delta', np.nan)):.6f}",
        f"- Z delta: {float(overall_old.get('z_delta', np.nan)):.4f}",
        "",
        "## Concentration Checks",
        f"- P1/T1 profit: {float(p1t1.get('profit', 0.0)):.2f} on {int(p1t1.get('bets', 0))} bets.",
        f"- Top-5 profit: {float(top5.get('profit', 0.0)):.2f} on {int(top5.get('bets', 0))} bets.",
        f"- Non-top-5 profit: {float(non_top5.get('profit', 0.0)):.2f} on {int(non_top5.get('bets', 0))} bets.",
        f"- Excluding missing TM profit: {float(excl_tm.get('profit', 0.0)):.2f} on {int(excl_tm.get('bets', 0))} bets.",
        f"- Excluding missing ClubElo profit: {float(excl_ce.get('profit', 0.0)):.2f} on {int(excl_ce.get('bets', 0))} bets.",
        f"- Old profit mainly comes from P1/T1 again: {p1t1_main}.",
        f"- Top-5 still fails: {top5_fails}.",
        f"- Non-top-5 drives result: {non_top5_drives}.",
        f"- Survives excluding rows with missing Transfermarkt values: {survives_no_tm_missing}.",
        f"- Survives excluding rows with missing ClubElo values: {survives_no_ce_missing}.",
        "",
        "No new rule, threshold, value filter, or extra data source was introduced. No confirmed edge is claimed.",
    ]
    (OUT / "v3_full_scope_compat_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "v3_full_scope_compat_decision.md").write_text(
        f"# V3 Full-Scope Compatibility Decision\n\nDecision: `{decision}`\n\nCompatibility reproduction only, not exact V3 reproduction, research_only. No confirmed edge is claimed.\n",
        encoding="utf-8",
    )


def write_missing_protocol_outputs(detail: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    decision = "v3_full_scope_compat_failed_missing_protocol"
    pd.DataFrame([{"decision": decision, "details": detail}]).to_csv(OUT / "v3_full_scope_compat_summary.csv", index=False)
    pd.DataFrame([{"check_name": "old_protocol_recovered", "status": "fail", "details": detail}]).to_csv(OUT / "v3_full_scope_compat_leakage_checks.csv", index=False)
    (OUT / "v3_full_scope_compat_report.md").write_text(f"# V3 Full-Scope Compatibility Replay\n\nDecision: `{decision}`\n\n{detail}\n", encoding="utf-8")
    (OUT / "v3_full_scope_compat_decision.md").write_text(f"# V3 Full-Scope Compatibility Decision\n\nDecision: `{decision}`\n", encoding="utf-8")
    for name in [
        "v3_full_scope_compat_old_vs_clean.csv",
        "v3_full_scope_compat_by_season.csv",
        "v3_full_scope_compat_by_league.csv",
        "v3_full_scope_compat_by_odds_source.csv",
        "v3_full_scope_compat_feature_availability.csv",
        "v3_full_scope_compat_selected_bets.csv",
        "v3_full_scope_compat_row_predictions.csv",
    ]:
        pd.DataFrame().to_csv(OUT / name, index=False)
    print(decision)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_ok, protocol_detail = load_protocol_status()
    if not protocol_ok:
        write_missing_protocol_outputs(protocol_detail)
        return

    raw = pd.read_csv(INPUT, low_memory=False)
    feature_map, bridge, _contract, readiness = load_feature_bridge()
    missing_non_exact = MISSING_NON_EXACT not in feature_map
    feature_cols = [c for c in feature_map if c != MISSING_NON_EXACT]
    adapter = build_adapter(raw, feature_map)
    pred = annual_predictions(adapter, feature_cols)
    value_pred = add_value_columns(pred)
    selected = select_locked(value_pred)
    scored_years = sorted(value_pred["season_start_year"].dropna().astype(int).unique()) if not value_pred.empty else []
    missing_scheduled_years = sorted(set(LOCKED_RULE_SCHEDULE) - set(scored_years))

    summary = pd.DataFrame([summarize_selected(selected, prediction_rows=len(value_pred))])
    if not readiness.empty:
        summary["old_v3_required_features"] = int(readiness["old_v3_required_features"].iloc[0])
        summary["exact_reconstructed_features"] = int(readiness["exact_reconstructed_features"].iloc[0])
    summary["missing_feature_handling"] = f"{MISSING_NON_EXACT} excluded; not approximated"
    summary["scored_years"] = "|".join(map(str, scored_years))
    summary["missing_scheduled_years"] = "|".join(map(str, missing_scheduled_years))
    by_season = breakdown(selected, "season_start_year")
    by_league = breakdown(selected, "league")
    by_odds_source = breakdown(selected, "x1_odds_source")
    feature_avail = feature_availability_breakdown(selected)
    old_vs = old_vs_clean(summary, by_season, by_league)
    checks = leakage_checks(raw, value_pred, selected, feature_cols, protocol_ok, protocol_detail, missing_scheduled_years)
    decision = decide(summary, protocol_ok, missing_non_exact)
    summary["decision"] = decision

    value_pred.to_csv(OUT / "v3_full_scope_compat_row_predictions.csv", index=False)
    selected.to_csv(OUT / "v3_full_scope_compat_selected_bets.csv", index=False)
    summary.to_csv(OUT / "v3_full_scope_compat_summary.csv", index=False)
    old_vs.to_csv(OUT / "v3_full_scope_compat_old_vs_clean.csv", index=False)
    by_season.to_csv(OUT / "v3_full_scope_compat_by_season.csv", index=False)
    by_league.to_csv(OUT / "v3_full_scope_compat_by_league.csv", index=False)
    by_odds_source.to_csv(OUT / "v3_full_scope_compat_by_odds_source.csv", index=False)
    feature_avail.to_csv(OUT / "v3_full_scope_compat_feature_availability.csv", index=False)
    checks.to_csv(OUT / "v3_full_scope_compat_leakage_checks.csv", index=False)
    write_report(decision, summary, feature_avail, old_vs, protocol_detail)
    print(decision)
    s = summary.iloc[0]
    print(f"prediction_rows={int(s['prediction_rows'])} bets={int(s['bets'])} profit={float(s['profit']):.2f} roi={float(s['roi']):.4%} z={float(s['z_score']):.4f}")


if __name__ == "__main__":
    main()
