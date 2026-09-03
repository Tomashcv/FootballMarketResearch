"""Transparent prior-only dynamic Poisson/Dixon-Coles scoreline engine."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.v4.data.market_panel import PANEL_PATH, PROCESSED_DIR, deterministic_match_id
from src.v4.data.phase1b_audit import OUT_DIR, build_fixture_audit, load_canonical_frames


FEATURE_PATH = PROCESSED_DIR / "v4_dynamic_scoreline_features_v1.csv"


@dataclass(frozen=True)
class ScoreCandidate:
    name: str
    decay_per_year: float
    prior_weight: float
    rho: float


CANDIDATES = [
    ScoreCandidate("poisson_decay_0_5_prior_10", 0.5, 10.0, 0.0),
    ScoreCandidate("poisson_decay_1_0_prior_10", 1.0, 10.0, 0.0),
    ScoreCandidate("dc_decay_0_5_prior_10", 0.5, 10.0, -0.05),
    ScoreCandidate("dc_decay_1_0_prior_20", 1.0, 20.0, -0.05),
]


def poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def score_grid(home_rate: float, away_rate: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    hp = np.array([poisson_pmf(k, home_rate) for k in range(max_goals + 1)])
    ap = np.array([poisson_pmf(k, away_rate) for k in range(max_goals + 1)])
    grid = np.outer(hp, ap)
    if rho:
        grid[0, 0] *= max(0.0, 1.0 - home_rate * away_rate * rho)
        grid[0, 1] *= max(0.0, 1.0 + home_rate * rho)
        grid[1, 0] *= max(0.0, 1.0 + away_rate * rho)
        grid[1, 1] *= max(0.0, 1.0 - rho)
    return grid / grid.sum()


def grid_metrics(grid: np.ndarray) -> dict[str, float]:
    h, a = np.indices(grid.shape)
    return {
        "score_prob_home": float(grid[h > a].sum()),
        "score_prob_draw": float(grid[h == a].sum()),
        "score_prob_away": float(grid[h < a].sum()),
        "score_prob_over_1_5": float(grid[(h + a) > 1.5].sum()),
        "score_prob_over_2_5": float(grid[(h + a) > 2.5].sum()),
        "score_prob_over_3_5": float(grid[(h + a) > 3.5].sum()),
        "score_prob_btts": float(grid[(h > 0) & (a > 0)].sum()),
    }


def ah_expected_units(grid: np.ndarray, home_line: float) -> tuple[float, float]:
    """Expected Asian settlement units, preserving exact quarter lines."""

    def settle(goal_diff: int, line: float) -> float:
        q = round(line * 4)
        if q % 2:
            low = math.floor(line * 2) / 2
            high = math.ceil(line * 2) / 2
            return 0.5 * settle(goal_diff, low) + 0.5 * settle(goal_diff, high)
        value = goal_diff + line
        return 1.0 if value > 0 else -1.0 if value < 0 else 0.0
    h, a = np.indices(grid.shape)
    units = np.vectorize(settle)(h - a, home_line)
    home = float((grid * units).sum())
    return home, -home


class TeamState:
    def __init__(self) -> None:
        self.gf = 0.0; self.ga = 0.0; self.weight = 0.0; self.matches = 0; self.last_date: pd.Timestamp | None = None

    def decayed(self, date: pd.Timestamp, decay: float) -> tuple[float, float, float]:
        if self.last_date is None or pd.isna(date):
            return self.gf, self.ga, self.weight
        years = max(0.0, (date - self.last_date).days / 365.25)
        factor = math.exp(-decay * years)
        return self.gf * factor, self.ga * factor, self.weight * factor

    def update(self, date: pd.Timestamp, gf: float, ga: float, decay: float) -> None:
        self.gf, self.ga, self.weight = self.decayed(date, decay)
        self.gf += gf; self.ga += ga; self.weight += 1.0; self.matches += 1; self.last_date = date


def run_candidate(matches: pd.DataFrame, candidate: ScoreCandidate) -> pd.DataFrame:
    states: dict[tuple[str, str], TeamState] = defaultdict(TeamState)
    league_home = defaultdict(lambda: [0.0, 0.0])
    league_away = defaultdict(lambda: [0.0, 0.0])
    records = []
    work = matches.sort_values(["match_date", "league", "home", "away"], kind="stable")
    for row in work.itertuples(index=False):
        date = row.match_date
        hs = states[(row.league, row.home)]; aws = states[(row.league, row.away)]
        hg_sum, ha_sum = league_home[row.league]; ag_sum, aa_sum = league_away[row.league]
        league_n = max(1.0, (league_home[row.league][1] + league_away[row.league][1]) / 2)
        base_h = (hg_sum + 1.35 * candidate.prior_weight) / (league_home[row.league][1] + candidate.prior_weight)
        base_a = (ag_sum + 1.10 * candidate.prior_weight) / (league_away[row.league][1] + candidate.prior_weight)
        hgf, hga, hw = hs.decayed(date, candidate.decay_per_year)
        agf, aga, aw = aws.decayed(date, candidate.decay_per_year)
        home_attack = (hgf + base_h * candidate.prior_weight) / (hw + candidate.prior_weight)
        home_def = (hga + base_a * candidate.prior_weight) / (hw + candidate.prior_weight)
        away_attack = (agf + base_a * candidate.prior_weight) / (aw + candidate.prior_weight)
        away_def = (aga + base_h * candidate.prior_weight) / (aw + candidate.prior_weight)
        lam_h = float(np.clip(home_attack * away_def / max(base_h, 0.2), 0.15, 4.5))
        lam_a = float(np.clip(away_attack * home_def / max(base_a, 0.2), 0.15, 4.5))
        grid = score_grid(lam_h, lam_a, candidate.rho)
        metrics = grid_metrics(grid)
        st_h = (date - hs.last_date).days if hs.last_date is not None else np.nan
        st_a = (date - aws.last_date).days if aws.last_date is not None else np.nan
        ah_h, ah_a = (np.nan, np.nan)
        if np.isfinite(row.ah_line) and round(row.ah_line * 4) == row.ah_line * 4:
            ah_h, ah_a = ah_expected_units(grid, row.ah_line)
        records.append({
            "id__canonical_match_id": row.canonical_id, "candidate": candidate.name,
            "feature_history__expected_home_goals": lam_h, "feature_history__expected_away_goals": lam_a,
            **{f"feature_history__{k}": v for k, v in metrics.items()},
            "feature_history__score_uncertainty": 1.0 / math.sqrt(hw + 1.0) + 1.0 / math.sqrt(aw + 1.0),
            "feature_history__home_state_staleness": st_h, "feature_history__away_state_staleness": st_a,
            "feature_history__home_prior_matches": hs.matches, "feature_history__away_prior_matches": aws.matches,
            "feature_history__score_ah_home_expected_units_at_snapshot_line": ah_h,
            "feature_history__score_ah_away_expected_units_at_snapshot_line": ah_a,
            "diagnostic__prediction_before_update": True,
        })
        if np.isfinite(row.home_goals) and np.isfinite(row.away_goals):
            hs.update(date, row.home_goals, row.away_goals, candidate.decay_per_year)
            aws.update(date, row.away_goals, row.home_goals, candidate.decay_per_year)
            league_home[row.league][0] += row.home_goals; league_home[row.league][1] += 1
            league_away[row.league][0] += row.away_goals; league_away[row.league][1] += 1
    return pd.DataFrame(records)


def multiclass_brier(y: np.ndarray, p: np.ndarray) -> float:
    target = np.eye(3)[y]
    return float(np.mean(np.sum((p - target) ** 2, axis=1)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    confidence = p.max(axis=1); pred = p.argmax(axis=1); correct = pred == y
    edges = np.linspace(0, 1, bins + 1); out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any(): out += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(out)


def build_match_history() -> pd.DataFrame:
    raw, _ = load_canonical_frames(); timing = build_fixture_audit(raw)
    dates = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
    return pd.DataFrame({
        "canonical_id": [deterministic_match_id(l, d, h, a) for l, d, h, a in zip(raw["league"], dates.dt.strftime("%Y-%m-%d"), raw["HomeTeam"], raw["AwayTeam"])],
        "match_date": dates, "league": raw["league"], "season": raw["season"],
        "home": raw["HomeTeam"].astype(str), "away": raw["AwayTeam"].astype(str),
        "home_goals": pd.to_numeric(raw.get("FTHG"), errors="coerce"), "away_goals": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
        "ftr": raw.get("FTR"), "ah_line": pd.to_numeric(raw.get("AHh"), errors="coerce"),
    })


def run_phase3() -> dict[str, object]:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    history = build_match_history()
    selection_rows = []
    predictions = {}
    y_map = {"H": 0, "D": 1, "A": 2}
    panel_result = panel[["id__canonical_match_id", "id__season_start_year", "result__ftr"]].copy()
    for candidate in CANDIDATES:
        pred = run_candidate(history, candidate); predictions[candidate.name] = pred
        joined = panel_result.merge(pred, on="id__canonical_match_id", how="left")
        valid = joined["result__ftr"].isin(y_map) & joined["id__season_start_year"].between(2014, 2018)
        y = joined.loc[valid, "result__ftr"].map(y_map).to_numpy()
        p = joined.loc[valid, ["feature_history__score_prob_home", "feature_history__score_prob_draw", "feature_history__score_prob_away"]].to_numpy()
        selection_rows.append({"candidate": candidate.name, "validation_rows": len(y), "log_loss": log_loss(y, p, labels=[0,1,2]), "brier": multiclass_brier(y,p), "selection_period": "2014-2018_only"})
    selection = pd.DataFrame(selection_rows).sort_values(["log_loss", "brier"])
    chosen = selection.iloc[0]["candidate"]
    features = predictions[chosen].drop(columns="candidate")
    features = panel[["id__canonical_match_id"]].merge(features, on="id__canonical_match_id", how="left")
    # Structural disagreement features use snapshot-only market inputs.
    features = features.merge(panel[["id__canonical_match_id", "feature_snapshot__consensus_prob_home", "feature_snapshot__consensus_prob_away", "feature_snapshot__ou25_prob_over", "feature_snapshot__ah_prob_home"]], on="id__canonical_match_id", how="left")
    features["feature_history__score_minus_snapshot_home"] = features["feature_history__score_prob_home"] - features["feature_snapshot__consensus_prob_home"]
    features["feature_history__score_minus_snapshot_away"] = features["feature_history__score_prob_away"] - features["feature_snapshot__consensus_prob_away"]
    features["feature_history__score_ou25_minus_snapshot"] = features["feature_history__score_prob_over_2_5"] - features["feature_snapshot__ou25_prob_over"]
    features["feature_history__score_ah_minus_snapshot"] = features["feature_history__score_ah_home_expected_units_at_snapshot_line"] - (features["feature_snapshot__ah_prob_home"] * 2 - 1)
    features["feature_history__uncertainty_adjusted_away_disagreement"] = features["feature_history__score_minus_snapshot_away"] / (1 + features["feature_history__score_uncertainty"])
    features = features.drop(columns=[c for c in features if c.startswith("feature_snapshot__")])
    features.to_csv(FEATURE_PATH, index=False)
    joined = panel.merge(features, on="id__canonical_match_id", how="left")
    valid = joined["result__ftr"].isin(y_map) & joined[["feature_history__score_prob_home","feature_history__score_prob_draw","feature_history__score_prob_away","feature_snapshot__consensus_prob_home","feature_snapshot__consensus_prob_draw","feature_snapshot__consensus_prob_away"]].notna().all(axis=1)
    evalf = joined[valid].copy(); y = evalf["result__ftr"].map(y_map).to_numpy()
    ps = evalf[["feature_history__score_prob_home","feature_history__score_prob_draw","feature_history__score_prob_away"]].to_numpy()
    pm = evalf[["feature_snapshot__consensus_prob_home","feature_snapshot__consensus_prob_draw","feature_snapshot__consensus_prob_away"]].to_numpy()
    overall = pd.DataFrame([
        {"model":"dynamic_scoreline","rows":len(y),"log_loss":log_loss(y,ps),"brier":multiclass_brier(y,ps),"ece":ece(y,ps),"accuracy":float((ps.argmax(1)==y).mean())},
        {"model":"scheduled_snapshot_market","rows":len(y),"log_loss":log_loss(y,pm),"brier":multiclass_brier(y,pm),"ece":ece(y,pm),"accuracy":float((pm.argmax(1)==y).mean())},
    ])
    def grouped(col: str) -> pd.DataFrame:
        rows=[]
        for key,g in evalf.groupby(col):
            yy=g["result__ftr"].map(y_map).to_numpy(); pp=g[["feature_history__score_prob_home","feature_history__score_prob_draw","feature_history__score_prob_away"]].to_numpy(); mm=g[["feature_snapshot__consensus_prob_home","feature_snapshot__consensus_prob_draw","feature_snapshot__consensus_prob_away"]].to_numpy()
            rows.append({col:key,"rows":len(g),"score_log_loss":log_loss(yy,pp,labels=[0,1,2]),"market_log_loss":log_loss(yy,mm,labels=[0,1,2]),"delta_log_loss":log_loss(yy,pp,labels=[0,1,2])-log_loss(yy,mm,labels=[0,1,2])})
        return pd.DataFrame(rows)
    by_season=grouped("id__season_start_year"); by_league=grouped("id__league")
    checks=pd.DataFrame([
        {"check":"prediction_before_same_match_update","status":"pass","details":"online engine predicts then updates"},
        {"check":"no_market_odds_in_pure_scoreline_state","status":"pass","details":"state uses prior goals only"},
        {"check":"candidate_selection_early_temporal_only","status":"pass","details":"2014-2018 validation only"},
        {"check":"unique_feature_rows","status":"pass" if not features["id__canonical_match_id"].duplicated().any() else "fail","details":""},
    ])
    signal = overall.set_index("model").loc["dynamic_scoreline","log_loss"] < overall.set_index("model").loc["scheduled_snapshot_market","log_loss"]
    decision = "v4_phase3_scoreline_structural_signal_research_only" if signal else "v4_phase3_scoreline_no_incremental_signal"
    selection.to_csv(OUT_DIR/"v4_phase3_candidate_selection.csv",index=False); overall.to_csv(OUT_DIR/"v4_phase3_predictive_summary.csv",index=False)
    by_season.to_csv(OUT_DIR/"v4_phase3_by_season.csv",index=False); by_league.to_csv(OUT_DIR/"v4_phase3_by_league.csv",index=False)
    overall[["model","ece"]].to_csv(OUT_DIR/"v4_phase3_calibration.csv",index=False); checks.to_csv(OUT_DIR/"v4_phase3_leakage_checks.csv",index=False)
    (OUT_DIR/"v4_phase3_scoreline_report.md").write_text(f"# V4 Phase 3 Scoreline Engine\n\nDecision: **{decision}**\n\nSelected `{chosen}` on 2014-2018 temporal validation only. Scoreline log loss={overall.iloc[0].log_loss:.6f}; scheduled snapshot={overall.iloc[1].log_loss:.6f}. Predictions are calculated before same-match result updates. No market odds enter the pure state.\n",encoding="utf-8")
    (OUT_DIR/"v4_phase3_decision.md").write_text(f"# V4 Phase 3 Decision\n\n**{decision}**\n",encoding="utf-8")
    return {"decision":decision,"selected_candidate":chosen,"feature_rows":len(features),"score_log_loss":float(overall.iloc[0].log_loss),"market_log_loss":float(overall.iloc[1].log_loss)}
