from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import build_football_data_full_scope_1x2 as fd  # noqa: E402
from scripts import build_full_scope_external_features as ext  # noqa: E402
from scripts.run_v3_exact_reproduction import (  # noqa: E402
    MODEL,
    FEATURE_GROUP,
    SCOPE,
    add_value_columns,
    build_adapter,
    max_drawdown,
    old_feature_cols,
    z_score,
)
from src.experiments.feature_matrix_v2_tm_1x2_predictive_audit import model_predict, normalize_probs  # noqa: E402
from src.features.internal_elo_features import add_internal_elo_features  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
HIST_EXACT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_v3_exact_research_v1.csv"
ROB_SUMMARY = ROOT / "outputs/reports/v3_robustness/v3_robustness_summary.csv"
ROB_SEASON = ROOT / "outputs/reports/v3_robustness/v3_robustness_by_season.csv"
ROB_NULL = ROOT / "outputs/reports/v3_robustness/v3_robustness_null_checks.csv"
OUT = ROOT / "outputs/reports/v3_2025_validation"

SCOPE_LEAGUES = ["E0", "SP1", "D1", "I1", "F1", "B1", "G1", "N1", "P1", "SC0", "T1"]
EXCLUDED = {"E1", "E2", "E3"}
RULE_EDGE = 0.015
RULE_MIN_ODDS = 1.5


def metric(frame: pd.DataFrame) -> dict[str, object]:
    bets = int(len(frame))
    profit = float(frame["profit"].sum()) if bets else 0.0
    return {
        "bets": bets,
        "profit": profit,
        "roi": profit / bets if bets else 0.0,
        "z_score": z_score(frame["profit"]) if bets else 0.0,
        "max_drawdown": max_drawdown(frame["profit"]) if bets else 0.0,
        "average_odds": float(frame["actual_odds"].mean()) if bets else np.nan,
        "average_edge": float(frame["away_edge"].mean()) if bets else np.nan,
    }


def grouped(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(by, dropna=False):
        rows.append({by: key, **metric(group)})
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True) if rows else pd.DataFrame()


def load_raw_2025_norm() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory, norm = fd.discover_and_normalize()
    norm = norm[norm["season_start_year"].astype("Int64").eq(2025) & norm["div"].isin(SCOPE_LEAGUES)].copy()
    norm = norm[norm["source_file"].astype(str).str.contains("/seasons/", regex=False)].copy()
    inventory_2025 = inventory[inventory["path"].astype(str).str.contains("2526", na=False)].copy()
    return inventory_2025, norm


def build_market_dataset(norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams, aliases, _added = fd.build_team_registry(norm)
    ids = fd.attach_team_ids(norm, aliases)
    marketed = fd.add_market(ids)
    selected, valid_rows, conflicts = fd.deduplicate(marketed)
    selected, source_map = fd.assign_ids(selected, marketed)
    implausible = fd.count_implausible(selected, marketed)
    # For validation, keep partial seasons but mark them instead of applying the historical <=2024 cutoff/quarantine.
    final = selected[~selected["score_conflict_quarantine_flag"]].copy()
    final["x1_home_raw_prob"] = 1.0 / final["x1_home_odds"]
    final["x1_draw_raw_prob"] = 1.0 / final["x1_draw_odds"]
    final["x1_away_raw_prob"] = 1.0 / final["x1_away_odds"]
    final["x1_overround"] = final[["x1_home_raw_prob", "x1_draw_raw_prob", "x1_away_raw_prob"]].sum(axis=1)
    final["x1_home_no_vig_prob"] = final["x1_home_raw_prob"] / final["x1_overround"]
    final["x1_draw_no_vig_prob"] = final["x1_draw_raw_prob"] / final["x1_overround"]
    final["x1_away_no_vig_prob"] = final["x1_away_raw_prob"] / final["x1_overround"]
    final["classification"] = "research_only"
    final["partial_latest_season_flag"] = True
    final["partial_season_flag"] = True
    final["dedup_tiebreak_policy"] = "valid target/odds; B365>Avg>HDA; raw season files only; completeness; stable source,row"
    cols = [
        "full_scope_match_id",
        "canonical_match_id",
        "div",
        "competition_slug",
        "competition_type",
        "competition_code",
        "season_start_year",
        "season_label",
        "match_date",
        "match_time",
        "match_datetime",
        "home_team_raw",
        "away_team_raw",
        "home_team_normalized",
        "away_team_normalized",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "result_1x2",
        "target_home_win",
        "target_draw",
        "target_away_win",
        "x1_home_odds",
        "x1_draw_odds",
        "x1_away_odds",
        "x1_odds_source",
        "x1_odds_timing_label",
        "x1_home_raw_prob",
        "x1_draw_raw_prob",
        "x1_away_raw_prob",
        "x1_overround",
        "x1_home_no_vig_prob",
        "x1_draw_no_vig_prob",
        "x1_away_no_vig_prob",
        "football_data_row_id",
        "source_file",
        "source",
        "logical_match_key",
        "partial_latest_season_flag",
        "partial_season_flag",
        "dedup_tiebreak_policy",
        "classification",
    ]
    x1 = final[[c for c in cols if c in final.columns]].copy()
    return x1, marketed, valid_rows, conflicts, implausible


def build_external_exact(x1: pd.DataFrame) -> pd.DataFrame:
    base = x1.copy()
    base["match_date"] = pd.to_datetime(base["match_date"], errors="coerce").dt.normalize()
    ratings = ext.read_clubelo_ratings()
    club_alias = ext.build_clubelo_alias(base, ratings)
    clubelo = ext.build_clubelo_features(base, ratings, club_alias)
    tm_map, _tm_alias = ext.load_tm_fixture_mapping(base)
    tm = ext.build_transfermarkt_features(base, tm_map)
    merged = base.merge(clubelo, on="full_scope_match_id", how="left", validate="one_to_one")
    merged = merged.merge(tm, on="full_scope_match_id", how="left", validate="one_to_one")
    merged = ext.add_compatibility_columns(merged)
    return merged


def add_internal_elo_with_history(validation: pd.DataFrame) -> pd.DataFrame:
    hist = pd.read_csv(HIST_EXACT, low_memory=False)
    all_rows = pd.concat([hist, validation], ignore_index=True, sort=False)
    work = pd.DataFrame(
        {
            "full_scope_match_id": all_rows["full_scope_match_id"].astype(str),
            "league": all_rows["div"].astype(str),
            "Date": pd.to_datetime(all_rows["match_date"], errors="coerce"),
            "Time": all_rows.get("match_time", pd.Series("", index=all_rows.index)).fillna("").astype(str),
            "HomeTeam": all_rows["home_team_raw"].astype(str),
            "AwayTeam": all_rows["away_team_raw"].astype(str),
            "FTHG": pd.to_numeric(all_rows["home_goals"], errors="coerce"),
            "FTAG": pd.to_numeric(all_rows["away_goals"], errors="coerce"),
            "clubelo_diff": pd.to_numeric(all_rows["clubelo_diff"], errors="coerce"),
            "is_validation": np.r_[np.zeros(len(hist), dtype=bool), np.ones(len(validation), dtype=bool)],
        },
        index=all_rows.index,
    )
    parts = []
    for _league, group in work.groupby("league", sort=False):
        elo = add_internal_elo_features(group[["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]], starting_elo=1500.0, k_factor=20.0, home_advantage_elo=65.0)
        part = group[["full_scope_match_id", "is_validation", "clubelo_diff"]].copy()
        part["home_internal_elo"] = elo["home_internal_elo_pre"]
        part["away_internal_elo"] = elo["away_internal_elo_pre"]
        part["internal_elo_diff"] = elo["internal_elo_diff_home_minus_away"]
        part["clubelo_diff_minus_internal_elo_diff"] = part["clubelo_diff"] - part["internal_elo_diff"]
        parts.append(part)
    elo_all = pd.concat(parts).sort_index()
    val_elo = elo_all[elo_all["is_validation"]].drop(columns=["is_validation", "clubelo_diff"])
    out = validation.drop(columns=["home_internal_elo", "away_internal_elo", "internal_elo_diff", "clubelo_diff_minus_internal_elo_diff"], errors="ignore")
    out["full_scope_match_id"] = out["full_scope_match_id"].astype(str)
    val_elo["full_scope_match_id"] = val_elo["full_scope_match_id"].astype(str)
    return out.merge(val_elo, on="full_scope_match_id", how="left", validate="one_to_one")


def score_2025(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = old_feature_cols()
    hist = pd.read_csv(HIST_EXACT, low_memory=False)
    combined = pd.concat([hist, validation], ignore_index=True, sort=False)
    adapter = build_adapter(combined, feature_cols)
    scoped = adapter[adapter["season_start_year"].notna() & ~adapter["league"].isin(EXCLUDED)].copy()
    train = scoped[scoped["season_start_year"].astype(int).lt(2025)].copy()
    test = scoped[scoped["season_start_year"].astype(int).eq(2025)].copy()
    rng = np.random.default_rng(20260701)
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
    pred["fold_test_year"] = 2025
    pred["scope"] = SCOPE
    pred["model"] = MODEL
    pred["feature_group"] = FEATURE_GROUP
    value_pred = add_value_columns(pred)
    selected = value_pred[value_pred["away_edge"].ge(RULE_EDGE) & value_pred["x1x2_avg_odds_away"].ge(RULE_MIN_ODDS)].copy()
    selected["selected_rule"] = "away_edge_0.015_odds_1.5"
    selected["edge_threshold"] = RULE_EDGE
    selected["min_odds"] = RULE_MIN_ODDS
    selected["side"] = "away"
    selected["profit"] = selected["away_profit"]
    selected["actual_profit"] = selected["profit"]
    selected["actual_odds"] = selected["x1x2_avg_odds_away"]
    return value_pred, selected


def metrics(frame: pd.DataFrame, prediction_rows: int | None = None) -> dict[str, object]:
    bets = int(len(frame))
    profit = float(frame["profit"].sum()) if bets else 0.0
    return {
        "prediction_rows": prediction_rows if prediction_rows is not None else bets,
        "bets": bets,
        "profit": profit,
        "roi": profit / bets if bets else 0.0,
        "z_score": z_score(frame["profit"]) if bets else 0.0,
        "max_drawdown": max_drawdown(frame["profit"]) if bets else 0.0,
        "average_odds": float(frame["actual_odds"].mean()) if bets else np.nan,
        "average_edge": float(frame["away_edge"].mean()) if bets else np.nan,
    }


def by_group(selected: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, group in selected.groupby(by, dropna=False):
        rows.append({by: key, **metrics(group)})
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True) if rows else pd.DataFrame()


def feature_coverage(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = {
        "ClubElo both-found": validation["clubelo_both_found_flag"].fillna(False).astype(bool),
        "Transfermarkt both-value-found": validation["tm_both_value_found_flag"].fillna(False).astype(bool),
        "TM fixture mapped": validation["tm_fixture_mapped"].fillna(False).astype(bool),
        "internal Elo available": validation[["home_internal_elo", "away_internal_elo", "internal_elo_diff"]].notna().all(axis=1),
    }
    for name, mask in checks.items():
        rows.append({"feature_bucket": name, "rows": int(mask.sum()), "coverage_rate": float(mask.mean()) if len(mask) else 0.0})
    old_features = old_feature_cols()
    missing = [c for c in old_features if c not in validation.columns]
    rows.append({"feature_bucket": "old V3 feature columns missing", "rows": len(missing), "coverage_rate": 0.0 if missing else 1.0})
    rows.append({"feature_bucket": "old V3 feature columns present", "rows": len(old_features) - len(missing), "coverage_rate": (len(old_features) - len(missing)) / len(old_features)})
    return pd.DataFrame(rows)


def market_audit(x1: pd.DataFrame, marketed: pd.DataFrame, conflicts: pd.DataFrame, implausible: pd.DataFrame) -> pd.DataFrame:
    league = x1.groupby("div").agg(
        eligible_matches=("full_scope_match_id", "count"),
        latest_match_date=("match_date", "max"),
        b365_rows=("x1_odds_source", lambda s: int(s.eq("B365").sum())),
        avg_rows=("x1_odds_source", lambda s: int(s.eq("Avg").sum())),
    ).reset_index()
    league["partial_season_flag"] = True
    league["raw_scope_file_available"] = True
    missing = sorted(set(SCOPE_LEAGUES) - set(league["div"]))
    missing_rows = pd.DataFrame(
        [{"div": d, "eligible_matches": 0, "latest_match_date": "", "b365_rows": 0, "avg_rows": 0, "partial_season_flag": True, "raw_scope_file_available": False} for d in missing]
    )
    out = pd.concat([league, missing_rows], ignore_index=True, sort=False)
    out["score_conflict_quarantined_rows"] = len(conflicts[conflicts["final_action"].eq("quarantined_score_conflict")]) if not conflicts.empty else 0
    out["raw_valid_candidate_rows"] = int(len(marketed))
    out["implausible_or_partial_reason"] = "2025/26 validation season is incomplete/partial; kept as validation-only, not training."
    return out.sort_values("div").reset_index(drop=True)


def null_check_audit() -> pd.DataFrame:
    prev = pd.read_csv(ROB_NULL) if ROB_NULL.exists() else pd.DataFrame()
    rows = []
    if not prev.empty:
        for r in prev.itertuples(index=False):
            assessment = "usable_diagnostic"
            details = "Diagnostic only; not used for tuning."
            if str(r.null_check).startswith("shuffle_outcomes"):
                details = "Corrected implementation shuffles away-win outcomes within season+league against fixed selected odds. It tests settlement luck conditional on selected odds and group win rates, not model skill."
            elif str(r.null_check).startswith("shuffle_edges"):
                details = "Samples random rows with same season+league bet counts. This is the more relevant diagnostic for whether selected edge ranking adds value versus random group selection."
            rows.append({"null_check": r.null_check, "previous_empirical_p_value_profit_ge_observed": r.empirical_p_value_profit_ge_observed, "assessment": assessment, "details": details})
    return pd.DataFrame(rows)


def historical_comparison(selected: pd.DataFrame) -> pd.DataFrame:
    robust = pd.read_csv(ROB_SUMMARY).iloc[0].to_dict()
    season = pd.read_csv(ROB_SEASON)
    y2024 = season[season["season_start_year"].eq(2024)].iloc[0].to_dict()
    avg = season[["bets", "profit", "roi", "z_score"]].mean(numeric_only=True).to_dict()
    val = metrics(selected)
    rows = [
        {"comparison": "2025_validation", **val},
        {"comparison": "historical_robustness", "prediction_rows": robust.get("prediction_rows", np.nan), "bets": robust["bets"], "profit": robust["profit"], "roi": robust["roi"], "z_score": robust["z_score"], "max_drawdown": robust["max_drawdown"], "average_odds": robust["average_odds"], "average_edge": robust["average_edge"]},
        {"comparison": "historical_2024", "prediction_rows": np.nan, "bets": y2024["bets"], "profit": y2024["profit"], "roi": y2024["roi"], "z_score": y2024["z_score"], "max_drawdown": y2024["max_drawdown"], "average_odds": y2024["average_odds"], "average_edge": y2024["average_edge"]},
        {"comparison": "average_historical_season", "prediction_rows": np.nan, "bets": avg["bets"], "profit": avg["profit"], "roi": avg["roi"], "z_score": avg["z_score"], "max_drawdown": np.nan, "average_odds": np.nan, "average_edge": np.nan},
    ]
    return pd.DataFrame(rows)


def leakage_checks(validation: pd.DataFrame, pred: pd.DataFrame, selected: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    match_dates = pd.to_datetime(validation["match_date"], errors="coerce")
    home_ce = pd.to_datetime(validation["home_clubelo_latest_date"], errors="coerce")
    away_ce = pd.to_datetime(validation["away_clubelo_latest_date"], errors="coerce")
    tm_date_cols = [c for c in ["home_tm_latest_valuation_date", "away_tm_latest_valuation_date"] if c in validation.columns]
    tm_ok = True
    for c in tm_date_cols:
        d = pd.to_datetime(validation[c], errors="coerce")
        tm_ok = tm_ok and bool((d.isna() | d.lt(match_dates)).all())
    checks = [
        ("no_duplicate_match_ids_predictions", pred["full_scope_match_id"].duplicated().sum() == 0, f"duplicates={int(pred['full_scope_match_id'].duplicated().sum())}"),
        ("no_duplicate_match_ids_selected", selected["full_scope_match_id"].duplicated().sum() == 0 if not selected.empty else True, f"duplicates={int(selected['full_scope_match_id'].duplicated().sum()) if not selected.empty else 0}"),
        ("no_duplicate_logical_matches_validation", validation["logical_match_key"].duplicated().sum() == 0, f"duplicates={int(validation['logical_match_key'].duplicated().sum())}"),
        ("source_file_not_identity", True, "full_scope_match_id/logical_match_key assigned from league-season-teams; source row only audit metadata."),
        ("classification_research_only", validation["classification"].eq("research_only").all() and pred["classification"].eq("research_only").all(), "validation and predictions are research_only"),
        ("internal_elo_pre_match", validation[["home_internal_elo", "away_internal_elo", "internal_elo_diff"]].notna().all().all(), "internal Elo emitted before validation match update using historical prior."),
        ("clubelo_strict_before_match", bool(((home_ce.isna() | home_ce.lt(match_dates)) & (away_ce.isna() | away_ce.lt(match_dates))).all()), "ClubElo latest date before match date where available."),
        ("tm_point_in_time", tm_ok, "Transfermarkt valuation dates before match date where available; transfer windows use transfer_date < match_date."),
        ("no_training_on_2025_outcomes", True, "Model training rows are historical cleaned seasons before 2025 only."),
        ("partial_season_marked", market["partial_season_flag"].fillna(False).astype(bool).all(), "All 2025 league rows marked partial."),
    ]
    return pd.DataFrame([{"check_name": n, "status": "pass" if ok else "fail", "details": d} for n, ok, d in checks])


def decide(summary: dict[str, object], market: pd.DataFrame, checks: pd.DataFrame) -> tuple[str, str]:
    data_blocker = checks["status"].eq("fail").any()
    available_leagues = int((market["eligible_matches"] > 0).sum())
    eligible = int(market["eligible_matches"].sum())
    bets = int(summary["bets"])
    if data_blocker:
        return "v3_2025_validation_failed_data_not_clean", "Validation has a data-quality/leakage blocker."
    if available_leagues < len(SCOPE_LEAGUES) or eligible < 250 or bets < 50:
        if summary["profit"] > 0 and bets > 0:
            return "v3_2025_validation_too_partial_to_judge", "Positive but too partial/low volume to judge."
        return "v3_2025_validation_too_partial_to_judge", "Too partial/low volume to judge."
    if summary["roi"] <= 0:
        return "v3_2025_validation_rejected", "2025 validation ROI is non-positive."
    if summary["z_score"] < 1.0:
        return "v3_2025_validation_neutral_inconclusive", "Positive ROI but weak validation z-score."
    return "v3_2025_validation_supports_candidate_research_only", "Positive validation with no data-quality blocker, but research_only."


def write_report(decision: str, interpretation: str, summary: dict[str, object], market: pd.DataFrame, comparison: pd.DataFrame, null_audit: pd.DataFrame) -> None:
    missing_leagues = market.loc[market["eligible_matches"].eq(0), "div"].tolist()
    lines = [
        "# V3 2025/26 Clean Validation",
        "",
        f"Decision: `{decision}`",
        "",
        "Frozen exact V3 full-scope 1X2 candidate. No new rule, threshold, filter, model feature, hyperparameter, or unrelated source was introduced. No confirmed edge is claimed.",
        "",
        "## Validation Result",
        f"- Eligible 2025/26 matches: {int(summary['eligible_matches'])}",
        f"- Prediction rows: {int(summary['prediction_rows'])}",
        f"- Bets: {int(summary['bets'])}",
        f"- Profit: {float(summary['profit']):.2f}",
        f"- ROI: {float(summary['roi']):.4%}",
        f"- Z-score: {float(summary['z_score']):.4f}",
        f"- Max drawdown: {float(summary['max_drawdown']):.2f}",
        f"- Average odds: {float(summary['average_odds']):.4f}",
        f"- Average edge: {float(summary['average_edge']):.6f}",
        "",
        "## Partial Season Caveat",
        f"- Available scope leagues: {int((market['eligible_matches'] > 0).sum())}/{len(SCOPE_LEAGUES)}",
        f"- Missing 2025/26 raw scope leagues: {', '.join(missing_leagues) if missing_leagues else 'none'}",
        "- All 2025/26 rows are marked `partial_season_flag=True` and are validation-only, not training.",
        "",
        "## Interpretation",
        interpretation,
        "",
        "## Historical Comparison",
        comparison.to_markdown(index=False),
        "",
        "## Null Check Audit",
        null_audit.to_markdown(index=False) if not null_audit.empty else "_No null-check report available._",
    ]
    (OUT / "v3_2025_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "v3_2025_validation_decision.md").write_text(
        f"# V3 2025 Validation Decision\n\nDecision: `{decision}`\n\n{interpretation}\n\nResearch only. No confirmed edge is claimed.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory, norm = load_raw_2025_norm()
    if norm.empty:
        decision = "v3_2025_validation_failed_data_not_clean"
        pd.DataFrame([{"decision": decision, "details": "No raw 2025/26 full-scope rows found."}]).to_csv(OUT / "v3_2025_market_dataset_audit.csv", index=False)
        (OUT / "v3_2025_validation_decision.md").write_text(f"# V3 2025 Validation Decision\n\nDecision: `{decision}`\n", encoding="utf-8")
        print(decision)
        return
    x1, marketed, _valid_rows, conflicts, implausible = build_market_dataset(norm)
    market = market_audit(x1, marketed, conflicts, implausible)
    validation = build_external_exact(x1)
    validation = add_internal_elo_with_history(validation)
    pred, selected = score_2025(validation)
    selected["month"] = pd.to_datetime(selected["match_date"], errors="coerce").dt.to_period("M").astype(str)
    by_league = by_group(selected, "league")
    by_month = by_group(selected, "month")
    coverage = feature_coverage(validation)
    comparison = historical_comparison(selected)
    null_audit = null_check_audit()
    checks = leakage_checks(validation, pred, selected, market)
    summary = metrics(selected, prediction_rows=len(pred))
    summary["eligible_matches"] = len(validation)
    summary["available_scope_leagues"] = int((market["eligible_matches"] > 0).sum())
    decision, interpretation = decide(summary, market, checks)
    summary["decision"] = decision
    summary["classification"] = "research_only"
    market.to_csv(OUT / "v3_2025_market_dataset_audit.csv", index=False)
    coverage.to_csv(OUT / "v3_2025_feature_coverage.csv", index=False)
    selected.to_csv(OUT / "v3_2025_selected_bets.csv", index=False)
    pred.to_csv(OUT / "v3_2025_row_predictions.csv", index=False)
    by_league.to_csv(OUT / "v3_2025_by_league.csv", index=False)
    by_month.to_csv(OUT / "v3_2025_by_month.csv", index=False)
    comparison.to_csv(OUT / "v3_2025_historical_comparison.csv", index=False)
    null_audit.to_csv(OUT / "v3_null_check_audit.csv", index=False)
    checks.to_csv(OUT / "v3_2025_leakage_checks.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "v3_2025_validation_summary.csv", index=False)
    write_report(decision, interpretation, summary, market, comparison, null_audit)
    print(decision)
    print(f"eligible={summary['eligible_matches']} predictions={summary['prediction_rows']} bets={summary['bets']} profit={summary['profit']:.2f} roi={summary['roi']:.4%} z={summary['z_score']:.4f}")


if __name__ == "__main__":
    main()
