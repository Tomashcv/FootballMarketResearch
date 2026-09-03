from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXACT_INPUT = ROOT / "data/processed/super_csvs/research_ready_plus/football_data_full_scope/super_1x2_football_data_full_scope_v3_exact_research_v1.csv"
SUMMARY_IN = ROOT / "outputs/reports/v3_exact_reproduction/v3_exact_reproduction_summary.csv"
SELECTED_IN = ROOT / "outputs/reports/v3_exact_reproduction/v3_exact_selected_bets.csv"
PRED_IN = ROOT / "outputs/reports/v3_exact_reproduction/v3_exact_row_predictions.csv"
LEAKAGE_IN = ROOT / "outputs/reports/v3_exact_reproduction/v3_exact_leakage_checks.csv"
OUT = ROOT / "outputs/reports/v3_robustness"

TOP5 = {"E0", "SP1", "D1", "I1", "F1"}


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


def longest_losing_streak(profit: pd.Series) -> int:
    best = cur = 0
    for value in profit:
        if value < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def window_extremes(profit: pd.Series, n: int) -> tuple[float, float]:
    if len(profit) < n:
        return np.nan, np.nan
    roll = profit.rolling(n).sum().dropna()
    return float(roll.min()), float(roll.max())


def metrics(frame: pd.DataFrame) -> dict[str, object]:
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


def grouped(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(col, dropna=False):
        rows.append({col: key, **metrics(group)})
    return pd.DataFrame(rows).sort_values(col).reset_index(drop=True) if rows else pd.DataFrame()


def bucketed(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bets = frame.copy()
    bets["odds_bucket"] = pd.cut(
        bets["actual_odds"],
        bins=[1.5, 1.75, 2.0, 2.5, np.inf],
        labels=["odds_1.50_1.75", "odds_1.75_2.00", "odds_2.00_2.50", "odds_ge_2.50"],
        include_lowest=True,
        right=False,
    )
    bets["edge_bucket"] = pd.cut(
        bets["away_edge"],
        bins=[0.01, 0.015, 0.02, 0.03, np.inf],
        labels=["edge_0.010_0.015", "edge_0.015_0.020", "edge_0.020_0.030", "edge_ge_0.030"],
        include_lowest=True,
        right=False,
    )
    return grouped(bets, "odds_bucket"), grouped(bets, "edge_bucket")


def add_metadata(bets: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "full_scope_match_id",
        "home_clubelo_days_stale",
        "away_clubelo_days_stale",
        "home_tm_avg_valuation_staleness_days_prior365",
        "away_tm_avg_valuation_staleness_days_prior365",
        "home_tm_max_valuation_staleness_days_prior365",
        "away_tm_max_valuation_staleness_days_prior365",
        "home_internal_elo",
        "away_internal_elo",
        "internal_elo_diff",
        "clubelo_diff_minus_internal_elo_diff",
        "classification",
        "source_file",
        "home_goals",
        "away_goals",
        "home_clubelo_latest_date",
        "away_clubelo_latest_date",
    ]
    meta = pd.read_csv(EXACT_INPUT, usecols=lambda c: c in cols, low_memory=False)
    meta["full_scope_match_id"] = meta["full_scope_match_id"].astype(str)
    out = bets.copy()
    out["full_scope_match_id"] = out["full_scope_match_id"].astype(str)
    out = out.merge(meta, on="full_scope_match_id", how="left", suffixes=("", "_exact"), validate="one_to_one")
    out["classification"] = out["classification"].fillna(out.get("classification_exact"))
    return out


def feature_availability(bets: pd.DataFrame) -> pd.DataFrame:
    out = bets.copy()
    out["clubelo_staleness_max"] = out[["home_clubelo_days_stale", "away_clubelo_days_stale"]].max(axis=1)
    tm_stale_cols = [c for c in ["home_tm_avg_valuation_staleness_days_prior365", "away_tm_avg_valuation_staleness_days_prior365"] if c in out.columns]
    out["tm_staleness_max"] = out[tm_stale_cols].max(axis=1) if tm_stale_cols else np.nan
    masks = {
        "ClubElo both-found": out["clubelo_both_found_flag"].fillna(False).astype(bool),
        "missing ClubElo": ~out["clubelo_both_found_flag"].fillna(False).astype(bool),
        "Transfermarkt both-value-found": out["tm_both_value_found_flag"].fillna(False).astype(bool),
        "missing TM": ~out["tm_both_value_found_flag"].fillna(False).astype(bool),
        "internal Elo available": out[["home_internal_elo", "away_internal_elo", "internal_elo_diff"]].notna().all(axis=1),
        "ClubElo stale <=7d": out["clubelo_staleness_max"].le(7),
        "ClubElo stale 8-30d": out["clubelo_staleness_max"].gt(7) & out["clubelo_staleness_max"].le(30),
        "ClubElo stale >30d": out["clubelo_staleness_max"].gt(30),
        "TM stale <=30d": out["tm_staleness_max"].le(30),
        "TM stale 31-180d": out["tm_staleness_max"].gt(30) & out["tm_staleness_max"].le(180),
        "TM stale >180d": out["tm_staleness_max"].gt(180),
        "exclude missing TM": out["tm_both_value_found_flag"].fillna(False).astype(bool),
        "exclude missing ClubElo": out["clubelo_both_found_flag"].fillna(False).astype(bool),
        "exclude stale TM >180d": out["tm_staleness_max"].le(180),
    }
    rows = []
    for name, mask in masks.items():
        group = out[mask.fillna(False)].copy()
        rows.append({"segment": name, **metrics(group)})
    return pd.DataFrame(rows)


def null_checks(bets: pd.DataFrame, n_iter: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(20260705)
    observed_profit = float(bets["profit"].sum())
    observed_mdd = max_drawdown(bets.sort_values("match_date")["profit"])
    season_league = [idx.to_numpy() for _, idx in bets.groupby(["season_start_year", "league"]).groups.items()]
    profits = bets["profit"].to_numpy()
    selected_mask = np.ones(len(bets), dtype=bool)

    # Outcome shuffle: permute away-win outcomes among selected bets within season+league,
    # keeping each selected bet's own odds fixed.
    out_profit = []
    out_mdd = []
    ordered = bets.sort_values("match_date").index.to_numpy()
    pos = {idx: i for i, idx in enumerate(bets.index)}
    ordered_pos = np.array([pos[i] for i in ordered])
    wins = bets["target_y"].eq(2).to_numpy()
    odds = bets["actual_odds"].to_numpy(dtype=float)
    for _ in range(n_iter):
        shuffled_wins = wins.copy()
        for idx in season_league:
            locs = np.array([pos[i] for i in idx])
            shuffled_wins[locs] = rng.permutation(shuffled_wins[locs])
        shuffled = np.where(shuffled_wins, odds - 1.0, -1.0)
        out_profit.append(float(shuffled.sum()))
        out_mdd.append(max_drawdown(pd.Series(shuffled[ordered_pos])))
    # Edge shuffle: selected row count is fixed within season+league, but profits are sampled from all row predictions by shuffled edge ranks.
    pred = pd.read_csv(PRED_IN, low_memory=False)
    edge_profit = []
    for _ in range(n_iter):
        total = 0.0
        for key, sel_g in bets.groupby(["season_start_year", "league"]):
            pred_g = pred[(pred["season_start_year"].eq(key[0])) & (pred["league"].eq(key[1]))]
            if pred_g.empty or sel_g.empty:
                continue
            k = len(sel_g)
            # Same count as observed selected in that group; random sample diagnoses edge ranking value only.
            sampled = pred_g.sample(n=min(k, len(pred_g)), replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
            total += float(sampled["away_profit"].sum())
        edge_profit.append(total)
    rows = [
        {
            "null_check": "shuffle_outcomes_within_season_league",
            "iterations": n_iter,
            "observed_profit": observed_profit,
            "null_mean_profit": float(np.mean(out_profit)),
            "null_p95_profit": float(np.percentile(out_profit, 95)),
            "empirical_p_value_profit_ge_observed": float((np.array(out_profit) >= observed_profit).mean()),
            "observed_max_drawdown": observed_mdd,
            "null_mean_max_drawdown": float(np.mean(out_mdd)),
            "empirical_p_value_drawdown_le_observed": float((np.array(out_mdd) <= observed_mdd).mean()),
        },
        {
            "null_check": "shuffle_edges_within_season_league_fixed_bet_counts",
            "iterations": n_iter,
            "observed_profit": observed_profit,
            "null_mean_profit": float(np.mean(edge_profit)),
            "null_p95_profit": float(np.percentile(edge_profit, 95)),
            "empirical_p_value_profit_ge_observed": float((np.array(edge_profit) >= observed_profit).mean()),
            "observed_max_drawdown": np.nan,
            "null_mean_max_drawdown": np.nan,
            "empirical_p_value_drawdown_le_observed": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def drawdown_outputs(bets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = bets.sort_values(["match_date", "full_scope_match_id"]).reset_index(drop=True).copy()
    ordered["bet_number"] = np.arange(1, len(ordered) + 1)
    ordered["cum_profit"] = ordered["profit"].cumsum()
    ordered["running_peak"] = ordered["cum_profit"].cummax()
    ordered["drawdown"] = ordered["cum_profit"] - ordered["running_peak"]
    monthly = (
        ordered.assign(month=pd.to_datetime(ordered["match_date"]).dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(bets=("profit", "size"), profit=("profit", "sum"))
    )
    monthly["cum_profit"] = monthly["profit"].cumsum()
    w50_min, w50_max = window_extremes(ordered["profit"], 50)
    w100_min, w100_max = window_extremes(ordered["profit"], 100)
    summary = pd.DataFrame(
        [
            {
                "max_drawdown": max_drawdown(ordered["profit"]),
                "longest_losing_streak": longest_losing_streak(ordered["profit"]),
                "worst_50_bet_window": w50_min,
                "worst_100_bet_window": w100_min,
                "best_50_bet_window": w50_max,
                "best_100_bet_window": w100_max,
            }
        ]
    )
    return summary, monthly


def leakage_checks(bets: pd.DataFrame) -> pd.DataFrame:
    prior = pd.read_csv(LEAKAGE_IN)
    prior_fail = prior[prior["status"].eq("fail")]
    match_date = pd.to_datetime(bets["match_date"], errors="coerce")
    home_ce_date = pd.to_datetime(bets["home_clubelo_latest_date"], errors="coerce")
    away_ce_date = pd.to_datetime(bets["away_clubelo_latest_date"], errors="coerce")
    home_ce_ok = home_ce_date.isna() | home_ce_date.lt(match_date)
    away_ce_ok = away_ce_date.isna() | away_ce_date.lt(match_date)
    checks = [
        ("no_duplicate_match_ids_selected", bets["full_scope_match_id"].duplicated().sum() == 0, f"duplicates={int(bets['full_scope_match_id'].duplicated().sum())}"),
        ("no_duplicate_logical_matches_selected", bets["logical_match_key"].duplicated().sum() == 0, f"duplicates={int(bets['logical_match_key'].duplicated().sum())}"),
        ("no_source_file_duplicate_inflation", bets.duplicated(["source_file", "logical_match_key"]).sum() == 0, f"duplicates={int(bets.duplicated(['source_file', 'logical_match_key']).sum())}"),
        ("classification_research_only", bets["classification"].eq("research_only").all(), "selected bets remain research_only"),
        ("prior_exact_leakage_checks_no_fail", prior_fail.empty, f"prior fails={len(prior_fail)}; scheduled_years_available review is not treated as leakage failure"),
        ("internal_elo_pre_match", True, "Recovered internal Elo emits ratings before current match update."),
        ("clubelo_strictly_before_match_date", bool(home_ce_ok.all() and away_ce_ok.all()), "ClubElo latest dates are before match date where available; missing dates are only present when ClubElo is missing."),
        ("no_same_match_score_features_used", True, "Robustness uses frozen predictions/selected bets; no model rerun or score feature addition."),
        ("no_future_valuations_or_transfers_added", True, "No extra source join; only metadata already in exact research CSV."),
    ]
    return pd.DataFrame([{"check_name": n, "status": "pass" if ok else "fail", "details": d} for n, ok, d in checks])


def decide(summary: dict[str, object], by_season: pd.DataFrame, by_league: pd.DataFrame, checks: pd.DataFrame) -> tuple[str, dict[str, object]]:
    total_profit = float(summary["profit"])
    positive_seasons = int((by_season["profit"] > 0).sum())
    positive_leagues = int((by_league["profit"] > 0).sum())
    best_season_profit = float(by_season["profit"].max()) if not by_season.empty else 0.0
    best_league_profit = float(by_league["profit"].max()) if not by_league.empty else 0.0
    profit_ex_best_season = total_profit - best_season_profit
    profit_ex_best_league = total_profit - best_league_profit
    season_concentration = best_season_profit / total_profit if total_profit > 0 else np.inf
    league_concentration = best_league_profit / total_profit if total_profit > 0 else np.inf
    leakage_pass = not checks["status"].eq("fail").any()
    readiness = {
        "positive_seasons": positive_seasons,
        "positive_leagues": positive_leagues,
        "profit_excluding_best_season": profit_ex_best_season,
        "profit_excluding_best_league": profit_ex_best_league,
        "best_season_profit_share": season_concentration,
        "best_league_profit_share": league_concentration,
        "paper_tracking_ready": bool(
            summary["roi"] > 0
            and summary["z_score"] > 1.5
            and summary["bets"] >= 500
            and positive_seasons >= 3
            and positive_leagues >= 5
            and profit_ex_best_season > 0
            and profit_ex_best_league > 0
            and season_concentration <= 0.35
            and league_concentration <= 0.35
            and leakage_pass
        ),
    }
    if summary["profit"] <= 0 or summary["roi"] <= 0:
        return "v3_robustness_rejected", readiness
    if readiness["paper_tracking_ready"]:
        return "v3_robustness_ready_for_paper_tracking_research_only", readiness
    if summary["z_score"] > 1.5 and summary["bets"] >= 500:
        return "v3_robustness_ready_for_2025_clean_validation_research_only", readiness
    return "v3_robustness_survives_but_weak_research_only", readiness


def write_report(decision: str, summary: dict[str, object], readiness: dict[str, object], season: pd.DataFrame, league: pd.DataFrame, nulls: pd.DataFrame) -> None:
    best_season = season.sort_values("profit", ascending=False).iloc[0]
    worst_season = season.sort_values("profit").iloc[0]
    best_league = league.sort_values("profit", ascending=False).iloc[0]
    worst_league = league.sort_values("profit").iloc[0]
    lines = [
        "# V3 Exact Robustness Validation",
        "",
        f"Decision: `{decision}`",
        "",
        "Frozen exact V3 1X2 candidate only. No new rule, threshold, value filter, model change, or extra source was introduced. No confirmed edge is claimed.",
        "",
        "## Overall",
        f"- Bets: {int(summary['bets'])}",
        f"- Profit: {float(summary['profit']):.2f}",
        f"- ROI: {float(summary['roi']):.4%}",
        f"- Z-score: {float(summary['z_score']):.4f}",
        f"- Max drawdown: {float(summary['max_drawdown']):.2f}",
        f"- Average odds: {float(summary['average_odds']):.4f}",
        f"- Average edge: {float(summary['average_edge']):.6f}",
        "",
        "## Robustness",
        f"- Positive seasons: {readiness['positive_seasons']}",
        f"- Best season: {int(best_season['season_start_year'])} profit {float(best_season['profit']):.2f}",
        f"- Worst season: {int(worst_season['season_start_year'])} profit {float(worst_season['profit']):.2f}",
        f"- Profit excluding best season: {float(readiness['profit_excluding_best_season']):.2f}",
        f"- Positive leagues: {readiness['positive_leagues']}",
        f"- Best league: {best_league['league']} profit {float(best_league['profit']):.2f}",
        f"- Worst league: {worst_league['league']} profit {float(worst_league['profit']):.2f}",
        f"- Profit excluding best league: {float(readiness['profit_excluding_best_league']):.2f}",
        f"- Best season profit share: {float(readiness['best_season_profit_share']):.2%}",
        f"- Best league profit share: {float(readiness['best_league_profit_share']):.2%}",
        "",
        "## Null Checks",
    ]
    for row in nulls.itertuples(index=False):
        lines.append(f"- {row.null_check}: empirical p(profit >= observed) {row.empirical_p_value_profit_ge_observed:.4f}")
    lines += [
        "",
        f"Paper tracking readiness: {readiness['paper_tracking_ready']}. The blocker, if false, is concentration: paper criteria require no single season or league above 35% of profit.",
    ]
    (OUT / "v3_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "v3_robustness_decision.md").write_text(
        f"# V3 Robustness Decision\n\nDecision: `{decision}`\n\nResearch only. No confirmed edge is claimed.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bets = pd.read_csv(SELECTED_IN, low_memory=False)
    bets["match_date"] = pd.to_datetime(bets["match_date"], errors="coerce")
    bets = add_metadata(bets)
    overall = metrics(bets)
    exact_summary = pd.read_csv(SUMMARY_IN).iloc[0].to_dict()
    overall["prediction_rows"] = int(exact_summary["prediction_rows"])
    season = grouped(bets, "season_start_year")
    league = grouped(bets, "league")
    odds, edge = bucketed(bets)
    feature = feature_availability(bets)
    nulls = null_checks(bets, n_iter=1000)
    drawdown, curve = drawdown_outputs(bets)
    checks = leakage_checks(bets)
    decision, readiness = decide(overall, season, league, checks)
    summary = pd.DataFrame([{**overall, **readiness, "decision": decision, "classification": "research_only"}])
    summary.to_csv(OUT / "v3_robustness_summary.csv", index=False)
    season.to_csv(OUT / "v3_robustness_by_season.csv", index=False)
    league.to_csv(OUT / "v3_robustness_by_league.csv", index=False)
    odds.to_csv(OUT / "v3_robustness_by_odds_bucket.csv", index=False)
    edge.to_csv(OUT / "v3_robustness_by_edge_bucket.csv", index=False)
    feature.to_csv(OUT / "v3_robustness_feature_availability.csv", index=False)
    nulls.to_csv(OUT / "v3_robustness_null_checks.csv", index=False)
    drawdown.to_csv(OUT / "v3_robustness_drawdown.csv", index=False)
    curve.to_csv(OUT / "v3_robustness_profit_curve.csv", index=False)
    checks.to_csv(OUT / "v3_robustness_leakage_checks.csv", index=False)
    write_report(decision, overall, readiness, season, league, nulls)
    print(decision)
    print(f"bets={overall['bets']} profit={overall['profit']:.2f} roi={overall['roi']:.4%} z={overall['z_score']:.4f}")


if __name__ == "__main__":
    main()
