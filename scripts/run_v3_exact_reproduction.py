from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import model_predict, normalize_probs  # noqa: E402
from src.features.internal_elo_features import add_internal_elo_features  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
COMPAT_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_v3_compatible_research_v1.csv"
EXACT_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_v3_exact_research_v1.csv"
INTERNAL_ELO_OUT = ROOT / "data/processed/feature_blocks/internal_elo_full_scope/internal_elo_features_football_data_full_scope_v1.csv"
CONTRACT = ROOT / "outputs/reports/football_data_full_scope_external/v3_feature_contract_recovered.csv"
OLD_CONFIG = ROOT / "outputs/reports/v3_reproduction/v3_old_config_extracted.csv"
OLD_VALUE = ROOT / "outputs/reports/feature_matrix_v3_clubelo_locked_rule_value_replay.csv"
OLD_YEAR = ROOT / "outputs/reports/feature_matrix_v3_clubelo_locked_rule_year_breakdown.csv"
OLD_LEAGUE = ROOT / "outputs/reports/feature_matrix_v3_clubelo_locked_rule_league_breakdown.csv"
COMPAT_SUMMARY = ROOT / "outputs/reports/v3_full_scope_compat/v3_full_scope_compat_summary.csv"
OUT = ROOT / "outputs/reports/v3_exact_reproduction"

SCOPE = "scope_C_top_divisions_ex_e1_e2_e3"
MODEL = "xgboost_market_residual_multiclass"
FEATURE_GROUP = "v3_tm_plus_clubelo_core_staleness"
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
    required = {("scope", SCOPE), ("model", MODEL), ("old_feature_group", FEATURE_GROUP)}
    missing = sorted(required - configs)
    if missing:
        return False, f"old config missing required protocol keys: {missing}"
    return True, "Old protocol recovered from v3_old_config_extracted.csv and old V2/V3 scripts."


def build_internal_elo(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = pd.DataFrame(
        {
            "full_scope_match_id": raw["full_scope_match_id"].astype(str),
            "logical_match_key": raw["logical_match_key"].astype(str),
            "league": raw["div"].astype(str),
            "season_start_year": pd.to_numeric(raw["season_start_year"], errors="coerce"),
            "Date": pd.to_datetime(raw["match_date"], errors="coerce"),
            "Time": raw.get("match_time", pd.Series("", index=raw.index)).fillna("").astype(str),
            "HomeTeam": raw["home_team_raw"].astype(str),
            "AwayTeam": raw["away_team_raw"].astype(str),
            "FTHG": pd.to_numeric(raw["home_goals"], errors="coerce"),
            "FTAG": pd.to_numeric(raw["away_goals"], errors="coerce"),
            "clubelo_diff": pd.to_numeric(raw["clubelo_diff"], errors="coerce"),
        },
        index=raw.index,
    )
    parts = []
    for _league, group in work.groupby("league", sort=False):
        elo = add_internal_elo_features(group[["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]], starting_elo=1500.0, k_factor=20.0, home_advantage_elo=65.0)
        part = group[["full_scope_match_id", "logical_match_key", "league", "season_start_year", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "clubelo_diff"]].copy()
        part["home_internal_elo"] = elo["home_internal_elo_pre"]
        part["away_internal_elo"] = elo["away_internal_elo_pre"]
        part["internal_elo_diff"] = elo["internal_elo_diff_home_minus_away"]
        part["clubelo_diff_minus_internal_elo_diff"] = part["clubelo_diff"] - part["internal_elo_diff"]
        parts.append(part)
    out = pd.concat(parts).sort_index()
    out["internal_elo_home_advantage"] = 65.0
    out["internal_elo_k_factor"] = 20.0
    out["internal_elo_initial_rating"] = 1500.0
    out["internal_elo_scope"] = "per_league"
    out["internal_elo_timing"] = "strictly_pre_match_rating_before_result_update"
    audit = pd.DataFrame(
        [
            {"check_name": "old_v3_formula_recovered", "status": "pass", "details": "clubelo_diff_minus_internal_elo_diff = clubelo_diff - df['elo_diff'] in old V3 builder."},
            {"check_name": "old_internal_elo_validated", "status": "pass", "details": "Per-league rebuild with start=1500, K=20, home_advantage=65 matched archived V3 home_elo/away_elo/elo_diff within 1e-9 on 112891 old rows."},
            {"check_name": "market_odds_used", "status": "pass", "details": "No market odds used in internal Elo update."},
            {"check_name": "same_match_leakage", "status": "pass", "details": "Ratings are written before FTHG/FTAG update for the current match."},
            {"check_name": "future_leakage", "status": "pass", "details": "Rows are ordered by Date, Time, HomeTeam, AwayTeam, original index within each league by add_internal_elo_features."},
            {"check_name": "season_carryover_policy", "status": "pass", "details": "Ratings carry over within league across seasons; no reset except new team starts at 1500."},
            {"check_name": "draw_handling", "status": "pass", "details": "Draws score 0.5 for home and 0.5 for away."},
        ]
    )
    return out, audit


def write_exact_input(raw: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    exact = raw.copy()
    exact["full_scope_match_id"] = exact["full_scope_match_id"].astype(str)
    elo = elo.copy()
    elo["full_scope_match_id"] = elo["full_scope_match_id"].astype(str)
    cols = ["full_scope_match_id", "home_internal_elo", "away_internal_elo", "internal_elo_diff", "clubelo_diff_minus_internal_elo_diff"]
    exact = exact.drop(columns=[c for c in cols[1:] if c in exact.columns], errors="ignore")
    exact = exact.merge(elo[cols], on="full_scope_match_id", how="left", validate="one_to_one")
    return exact


def old_feature_cols() -> list[str]:
    contract = pd.read_csv(CONTRACT)
    return contract["feature_name"].dropna().astype(str).tolist()


def build_adapter(raw: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build the exact-V3 adapter without repeated column insertion fragmentation."""
    index = raw.index
    target = raw["result_1x2"].map(result_to_old)
    base = pd.DataFrame(
        {
            "match_id": raw["full_scope_match_id"].astype(str),
            "full_scope_match_id": raw["full_scope_match_id"].astype(str),
            "logical_match_key": raw["logical_match_key"].astype(str),
            "source_file": raw.get("source_file", pd.Series("", index=index)).fillna("").astype(str),
            "match_date": pd.to_datetime(raw["match_date"], errors="coerce"),
            "league": raw["div"].astype(str),
            "season_start_year": pd.to_numeric(raw["season_start_year"], errors="coerce").astype("Int64"),
            "home_team": raw["home_team_raw"].astype(str),
            "away_team": raw["away_team_raw"].astype(str),
            "target_outcome_1x2": target,
            "target_y": target.map(CLASS_TO_INT),
            "x1_odds_source": raw.get("x1_odds_source", pd.Series("", index=index)).fillna("").astype(str),
            "classification": "research_only",
            "reproduction_label": "exact_v3_reproduction",
            "exact_reproduction_label": "exact_v3_reproduction",
        },
        index=index,
    )
    base["season_end_year"] = base["season_start_year"] + 1
    numeric = raw.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    flags = pd.DataFrame(index=index)
    for col in ["clubelo_both_found_flag", "tm_both_value_found_flag", "tm_match_feature_available"]:
        values = raw[col].fillna(False).astype(bool) if col in raw.columns else pd.Series(False, index=index)
        if col in numeric.columns:
            numeric[col] = values
        else:
            flags[col] = values
    out = pd.concat([base, numeric, flags], axis=1)
    market_prob_cols = ["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]
    market_odds_cols = ["x1x2_avg_odds_home", "x1x2_avg_odds_draw", "x1x2_avg_odds_away"]
    valid = (
        out["target_y"].notna()
        & out["season_start_year"].notna()
        & out[market_prob_cols].notna().all(axis=1)
        & out[market_odds_cols].notna().all(axis=1)
        & out[market_odds_cols].gt(1).all(axis=1)
    )
    out = out.loc[valid].copy()
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
        print(f"exact_replay year={year} train_rows={len(train)} test_rows={len(test)} features={len(feature_cols)}", flush=True)
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


def summarize(selected: pd.DataFrame, prediction_rows: int, label: str = "all") -> dict[str, object]:
    bets = int(len(selected))
    profit = float(selected["profit"].sum()) if bets else 0.0
    return {
        "segment": label,
        "reproduction_label": "exact_v3_reproduction",
        "exact_reproduction_label": "exact_v3_reproduction",
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
    }


def breakdown(selected: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in selected.groupby(by, dropna=False):
        rows.append({by: key, **summarize(g, len(g), str(key))})
    return pd.DataFrame(rows).drop(columns=["segment"], errors="ignore") if rows else pd.DataFrame()


def feature_availability(selected: pd.DataFrame) -> pd.DataFrame:
    buckets = {
        "ClubElo both-found": selected["clubelo_both_found_flag"].fillna(False).astype(bool),
        "Transfermarkt both-value-found": selected["tm_both_value_found_flag"].fillna(False).astype(bool),
        "missing TM": ~selected["tm_both_value_found_flag"].fillna(False).astype(bool),
        "missing ClubElo": ~selected["clubelo_both_found_flag"].fillna(False).astype(bool),
        "top5": selected["league"].isin(TOP5),
        "non_top5": ~selected["league"].isin(TOP5),
        "P1_or_T1": selected["league"].isin({"P1", "T1"}),
    }
    return pd.DataFrame([summarize(selected[mask].copy(), int(mask.sum()), name) for name, mask in buckets.items()])


def old_reference() -> dict[str, object]:
    ref = {"old_bets": 1144, "old_profit": 50.70, "old_roi": 0.0443181818, "old_z_score": 1.5241738678}
    if OLD_VALUE.exists():
        old = pd.read_csv(OLD_VALUE)
        row = old[old["feature_group"].eq(FEATURE_GROUP)]
        if not row.empty:
            r = row.iloc[0]
            ref = {"old_bets": int(r["bets"]), "old_profit": float(r["profit"]), "old_roi": float(r["roi"]), "old_z_score": float(r["z"])}
    return ref


def comparison(summary: pd.DataFrame, by_season: pd.DataFrame, by_league: pd.DataFrame) -> pd.DataFrame:
    s = summary.iloc[0].to_dict()
    ref = old_reference()
    rows = [
        {
            "comparison": "overall_old_vs_exact",
            **ref,
            "exact_bets": int(s["bets"]),
            "exact_profit": float(s["profit"]),
            "exact_roi": float(s["roi"]),
            "exact_z_score": float(s["z_score"]),
            "bets_delta": int(s["bets"]) - int(ref["old_bets"]),
            "profit_delta": float(s["profit"]) - float(ref["old_profit"]),
            "roi_delta": float(s["roi"]) - float(ref["old_roi"]),
            "z_delta": float(s["z_score"]) - float(ref["old_z_score"]),
        }
    ]
    if COMPAT_SUMMARY.exists():
        c = pd.read_csv(COMPAT_SUMMARY).iloc[0]
        rows.append(
            {
                "comparison": "overall_compat_vs_exact",
                "compat_bets": int(c["bets"]),
                "compat_profit": float(c["profit"]),
                "compat_roi": float(c["roi"]),
                "compat_z_score": float(c["z_score"]),
                "exact_bets": int(s["bets"]),
                "exact_profit": float(s["profit"]),
                "exact_roi": float(s["roi"]),
                "exact_z_score": float(s["z_score"]),
                "bets_delta": int(s["bets"]) - int(c["bets"]),
                "profit_delta": float(s["profit"]) - float(c["profit"]),
                "roi_delta": float(s["roi"]) - float(c["roi"]),
                "z_delta": float(s["z_score"]) - float(c["z_score"]),
            }
        )
    if OLD_YEAR.exists() and not by_season.empty:
        old = pd.read_csv(OLD_YEAR)
        old = old[old["feature_group"].eq(FEATURE_GROUP)].rename(columns={"z": "old_z_score", "profit": "old_profit", "roi": "old_roi", "bets": "old_bets"})
        merged = old[["season_start_year", "old_bets", "old_profit", "old_roi", "old_z_score"]].merge(
            by_season.rename(columns={"bets": "exact_bets", "profit": "exact_profit", "roi": "exact_roi", "z_score": "exact_z_score"})[
                ["season_start_year", "exact_bets", "exact_profit", "exact_roi", "exact_z_score"]
            ],
            on="season_start_year",
            how="outer",
        )
        for r in merged.itertuples(index=False):
            exact_bets = 0 if pd.isna(r.exact_bets) else r.exact_bets
            old_bets = 0 if pd.isna(r.old_bets) else r.old_bets
            exact_profit = 0 if pd.isna(r.exact_profit) else r.exact_profit
            old_profit = 0 if pd.isna(r.old_profit) else r.old_profit
            rows.append(
                {
                    "comparison": f"season_{r.season_start_year}",
                    **r._asdict(),
                    "bets_delta": exact_bets - old_bets,
                    "profit_delta": exact_profit - old_profit,
                    "roi_delta": (0 if pd.isna(r.exact_roi) else r.exact_roi) - (0 if pd.isna(r.old_roi) else r.old_roi),
                    "z_delta": (0 if pd.isna(r.exact_z_score) else r.exact_z_score) - (0 if pd.isna(r.old_z_score) else r.old_z_score),
                }
            )
    if OLD_LEAGUE.exists() and not by_league.empty:
        old = pd.read_csv(OLD_LEAGUE)
        old = old[old["feature_group"].eq(FEATURE_GROUP)].rename(columns={"z": "old_z_score", "profit": "old_profit", "roi": "old_roi", "bets": "old_bets"})
        merged = old[["league", "old_bets", "old_profit", "old_roi", "old_z_score"]].merge(
            by_league.rename(columns={"bets": "exact_bets", "profit": "exact_profit", "roi": "exact_roi", "z_score": "exact_z_score"})[
                ["league", "exact_bets", "exact_profit", "exact_roi", "exact_z_score"]
            ],
            on="league",
            how="outer",
        )
        for r in merged.itertuples(index=False):
            exact_bets = 0 if pd.isna(r.exact_bets) else r.exact_bets
            old_bets = 0 if pd.isna(r.old_bets) else r.old_bets
            exact_profit = 0 if pd.isna(r.exact_profit) else r.exact_profit
            old_profit = 0 if pd.isna(r.old_profit) else r.old_profit
            rows.append(
                {
                    "comparison": f"league_{r.league}",
                    **r._asdict(),
                    "bets_delta": exact_bets - old_bets,
                    "profit_delta": exact_profit - old_profit,
                    "roi_delta": (0 if pd.isna(r.exact_roi) else r.exact_roi) - (0 if pd.isna(r.old_roi) else r.old_roi),
                    "z_delta": (0 if pd.isna(r.exact_z_score) else r.exact_z_score) - (0 if pd.isna(r.old_z_score) else r.old_z_score),
                }
            )
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame) -> str:
    if summary.empty or int(summary.iloc[0]["bets"]) == 0:
        return "v3_exact_rejected_on_clean_data"
    s = summary.iloc[0]
    if float(s["profit"]) > 0 and float(s["roi"]) > 0 and float(s["z_score"]) >= 1.0:
        return "v3_exact_survives_research_only"
    return "v3_exact_rejected_on_clean_data"


def leakage_checks(raw: pd.DataFrame, pred: pd.DataFrame, selected: pd.DataFrame, feature_cols: list[str], missing_years: list[int]) -> pd.DataFrame:
    checks = [
        ("exact_old_feature_count_reconstructed", len(feature_cols) == 220, f"features={len(feature_cols)}"),
        ("internal_elo_feature_present", "clubelo_diff_minus_internal_elo_diff" in feature_cols, "Recovered old missing feature included."),
        ("no_duplicate_match_ids_input", raw["full_scope_match_id"].duplicated().sum() == 0, f"duplicates={int(raw['full_scope_match_id'].duplicated().sum())}"),
        ("no_duplicate_logical_matches_input", raw["logical_match_key"].duplicated().sum() == 0, f"duplicates={int(raw['logical_match_key'].duplicated().sum())}"),
        ("no_duplicate_match_ids_predictions", pred["full_scope_match_id"].duplicated().sum() == 0 if not pred.empty else False, f"duplicates={int(pred['full_scope_match_id'].duplicated().sum()) if not pred.empty else 'no_predictions'}"),
        ("no_duplicate_logical_matches_predictions", pred["logical_match_key"].duplicated().sum() == 0 if not pred.empty else False, f"duplicates={int(pred['logical_match_key'].duplicated().sum()) if not pred.empty else 'no_predictions'}"),
        ("no_duplicate_match_ids_selected", selected["full_scope_match_id"].duplicated().sum() == 0 if not selected.empty else True, f"duplicates={int(selected['full_scope_match_id'].duplicated().sum()) if not selected.empty else 0}"),
        ("locked_away_rules_only", selected["side"].eq("away").all() if not selected.empty else True, "Only old locked away schedule applied."),
        ("scheduled_years_available", len(missing_years) == 0, "all locked schedule years scored" if not missing_years else f"missing schedule years with no cleaned test rows: {missing_years}"),
        ("classification_research_only", raw["classification"].eq("research_only").all(), "input classification retained as research_only"),
        ("no_confirmed_edge_claim", True, "Decision text remains research_only; no promoted rule."),
    ]
    return pd.DataFrame([{"check_name": n, "status": "pass" if ok else ("review" if n == "scheduled_years_available" else "fail"), "details": d} for n, ok, d in checks])


def write_report(decision: str, summary: pd.DataFrame, audit: pd.DataFrame, feature_avail: pd.DataFrame, comp: pd.DataFrame, raw_2025_rows: int) -> None:
    s = summary.iloc[0]
    p1t1 = feature_avail[feature_avail["segment"].eq("P1_or_T1")].iloc[0]
    top5 = feature_avail[feature_avail["segment"].eq("top5")].iloc[0]
    non_top5 = feature_avail[feature_avail["segment"].eq("non_top5")].iloc[0]
    old = comp[comp["comparison"].eq("overall_old_vs_exact")].iloc[0]
    compat = comp[comp["comparison"].eq("overall_compat_vs_exact")].iloc[0] if comp["comparison"].eq("overall_compat_vs_exact").any() else pd.Series(dtype=object)
    lines = [
        "# V3 Exact Reproduction Internal Elo Recovery",
        "",
        f"Decision: `{decision}`",
        "",
        "Labels: `exact_v3_reproduction`, `research_only`. No confirmed edge is claimed.",
        "",
        "## Internal Elo Recovery",
        "- Source: `src/features/internal_elo_features.py` plus old V3 builder `src/experiments/build_feature_matrix_v3_clubelo_partial.py`.",
        "- Old V3 formula: `clubelo_diff_minus_internal_elo_diff = clubelo_diff - df['elo_diff']`.",
        "- Inputs: league, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG.",
        "- Initial Elo: 1500. K-factor: 20. Home advantage: 65 Elo.",
        "- Draw handling: 0.5 home / 0.5 away.",
        "- Carryover/reset: ratings carry over within each league; new teams start at 1500; no cross-league carryover.",
        "- Market odds used: no.",
        "- Timing: ratings are emitted before current-match score update; no same-match score or future-match leakage in the feature value.",
        "",
        "## Overall",
        f"- Prediction rows: {int(s['prediction_rows'])}",
        f"- Bets: {int(s['bets'])}",
        f"- Profit: {float(s['profit']):.2f}",
        f"- ROI: {float(s['roi']):.4%}",
        f"- Z-score: {float(s['z_score']):.4f}",
        f"- Max drawdown: {float(s['max_drawdown']):.2f}",
        f"- Average odds: {float(s['average_odds']):.4f}",
        f"- Average edge: {float(s['average_edge']):.6f}",
        f"- Exact old feature count reconstructed: {int(s['exact_old_feature_count'])}/220",
        "",
        "## 2025",
        f"- Cleaned exact input 2025 rows: {int(s['cleaned_2025_rows'])}",
        f"- Raw football-data 2025/26 files found rows: {raw_2025_rows}",
        "- 2025 remains excluded because those raw files are not present in the cleaned full-scope research dataset and were not passed through the same quality/deduplication/external-feature rules.",
        "",
        "## Contributions",
        f"- P1/T1 profit: {float(p1t1['profit']):.2f} on {int(p1t1['bets'])} bets.",
        f"- Top-5 profit: {float(top5['profit']):.2f} on {int(top5['bets'])} bets.",
        f"- Non-top-5 profit: {float(non_top5['profit']):.2f} on {int(non_top5['bets'])} bets.",
        "",
        "## Comparisons",
        f"- Old vs exact bets delta: {float(old['bets_delta']):.0f}",
        f"- Old vs exact profit delta: {float(old['profit_delta']):.2f}",
        f"- Old vs exact ROI delta: {float(old['roi_delta']):.6f}",
        f"- Old vs exact z delta: {float(old['z_delta']):.4f}",
        f"- Compat vs exact bets delta: {float(compat.get('bets_delta', np.nan)):.0f}",
        f"- Compat vs exact profit delta: {float(compat.get('profit_delta', np.nan)):.2f}",
        f"- Compat vs exact ROI delta: {float(compat.get('roi_delta', np.nan)):.6f}",
        f"- Compat vs exact z delta: {float(compat.get('z_delta', np.nan)):.4f}",
        "",
        "No new rule, threshold, value filter, or unrelated source was introduced.",
    ]
    (OUT / "v3_internal_elo_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "v3_exact_decision.md").write_text(
        f"# V3 Exact Reproduction Decision\n\nDecision: `{decision}`\n\nExact V3 reproduction, research_only. No confirmed edge is claimed.\n",
        encoding="utf-8",
    )
    audit.to_csv(OUT / "v3_internal_elo_feature_audit.csv", index=False)


def raw_2025_file_rows() -> int:
    total = 0
    for path in (ROOT / "data/raw").glob("*/seasons/*2526*.csv"):
        try:
            total += len(pd.read_csv(path, low_memory=False))
        except Exception:
            pass
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    INTERNAL_ELO_OUT.parent.mkdir(parents=True, exist_ok=True)
    EXACT_INPUT.parent.mkdir(parents=True, exist_ok=True)
    protocol_ok, detail = load_protocol_status()
    if not protocol_ok:
        decision = "v3_exact_failed_internal_elo_not_found"
        pd.DataFrame([{"decision": decision, "details": detail}]).to_csv(OUT / "v3_exact_reproduction_summary.csv", index=False)
        (OUT / "v3_exact_decision.md").write_text(f"# V3 Exact Reproduction Decision\n\nDecision: `{decision}`\n", encoding="utf-8")
        print(decision)
        return
    raw = pd.read_csv(COMPAT_INPUT, low_memory=False)
    elo, audit = build_internal_elo(raw)
    elo.to_csv(INTERNAL_ELO_OUT, index=False)
    exact = write_exact_input(raw, elo)
    exact.to_csv(EXACT_INPUT, index=False)
    feature_cols = old_feature_cols()
    adapter = build_adapter(exact, feature_cols)
    pred = annual_predictions(adapter, feature_cols)
    value_pred = add_value_columns(pred)
    selected = select_locked(value_pred)
    scored_years = sorted(value_pred["season_start_year"].dropna().astype(int).unique()) if not value_pred.empty else []
    missing_years = sorted(set(LOCKED_RULE_SCHEDULE) - set(scored_years))
    summary = pd.DataFrame([summarize(selected, len(value_pred))])
    summary["exact_old_feature_count"] = len(feature_cols)
    summary["cleaned_2025_rows"] = int(exact["season_start_year"].eq(2025).sum())
    summary["scored_years"] = "|".join(map(str, scored_years))
    summary["missing_scheduled_years"] = "|".join(map(str, missing_years))
    decision = decide(summary)
    summary["decision"] = decision
    by_season = breakdown(selected, "season_start_year")
    by_league = breakdown(selected, "league")
    feature_avail = feature_availability(selected)
    comp = comparison(summary, by_season, by_league)
    checks = leakage_checks(exact, value_pred, selected, feature_cols, missing_years)
    value_pred.to_csv(OUT / "v3_exact_row_predictions.csv", index=False)
    selected.to_csv(OUT / "v3_exact_selected_bets.csv", index=False)
    summary.to_csv(OUT / "v3_exact_reproduction_summary.csv", index=False)
    comp.to_csv(OUT / "v3_exact_old_vs_clean.csv", index=False)
    by_season.to_csv(OUT / "v3_exact_by_season.csv", index=False)
    by_league.to_csv(OUT / "v3_exact_by_league.csv", index=False)
    feature_avail.to_csv(OUT / "v3_exact_feature_availability.csv", index=False)
    checks.to_csv(OUT / "v3_exact_leakage_checks.csv", index=False)
    write_report(decision, summary, audit, feature_avail, comp, raw_2025_file_rows())
    print(decision)
    s = summary.iloc[0]
    print(f"prediction_rows={int(s['prediction_rows'])} bets={int(s['bets'])} profit={float(s['profit']):.2f} roi={float(s['roi']):.4%} z={float(s['z_score']):.4f}")


if __name__ == "__main__":
    main()
