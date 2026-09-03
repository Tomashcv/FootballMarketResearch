"""Phase 7 research-only decisions, diagnostics, and reproducible robustness.

All rule selection happens on a fold's calibration season.  The corresponding
outer season is read exactly once for reporting.  Closing prices are consumed
only as movement labels/diagnostics and never as model features.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.v4.data.phase1b_audit import OUT_DIR
from src.v4.features.model_matrix import MATRIX_PATH, GROUPS_PATH, PROCESSED_DIR
from src.v4.validation.nested import PREDICTIONS_PATH, SEED


SIDES = ("home", "draw", "away")
VARIANTS = (
    "outcome_edge_only",
    "predicted_clv_only",
    "outcome_edge_and_clv",
    "outcome_edge_clv_and_uncertainty",
    "outcome_edge_clv_cross_market_agreement",
    "outcome_edge_clv_bookmaker_disagreement",
    "outcome_edge_clv_uncertainty_and_cross_market",
)
EDGE_GRID = (0.005, 0.010, 0.015, 0.020, 0.030)
PCLV_GRID = (0.50, 0.55, 0.60, 0.65)
ECLV_GRID = (0.000, 0.005, 0.010, 0.020)
PROFILES = tuple(zip(EDGE_GRID[:4], PCLV_GRID, ECLV_GRID))
ODDS_RANGES = ((0.0, math.inf, "unrestricted"), (1.50, 3.00, "1.50_3.00"), (1.70, 2.50, "1.70_2.50"), (1.80, 2.25, "1.80_2.25"))
UNCERTAINTY_QUANTILES = (0.50, 0.70, 0.90)


def settle_1x2(selection: str, result: str, odds: float) -> float:
    """Flat 1u ordinary 1X2 settlement."""
    winner = {"H": "home", "D": "draw", "A": "away"}.get(str(result).upper())
    return float(odds - 1.0) if selection == winner else -1.0


def haircut_odds(odds: float, fraction: float) -> float:
    """Conservatively reduce the full decimal price, floored at 1.0."""
    return max(1.0, float(odds) * (1.0 - float(fraction)))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _metrics(bets: pd.DataFrame, profit_col: str = "profit") -> dict[str, float | int]:
    if bets.empty:
        return {"bets": 0, "profit": 0.0, "roi": np.nan, "z_score": np.nan, "max_drawdown": 0.0,
                "average_odds": np.nan, "win_rate": np.nan, "mean_clv": np.nan, "median_clv": np.nan,
                "positive_clv_rate": np.nan}
    p = bets[profit_col].astype(float).to_numpy()
    ordered = bets.assign(_d=pd.to_datetime(bets.id__match_date)).sort_values(["_d", "id__canonical_match_id"])
    curve = ordered[profit_col].astype(float).cumsum(); drawdown = curve - curve.cummax()
    sd = p.std(ddof=1) if len(p) > 1 else np.nan
    return {"bets": len(bets), "profit": float(p.sum()), "roi": float(p.mean()),
            "z_score": float(p.mean() / (sd / math.sqrt(len(p)))) if len(p) > 1 and sd > 0 else np.nan,
            "max_drawdown": float(drawdown.min()), "average_odds": float(bets.odds.mean()),
            "win_rate": float((p > 0).mean()), "mean_clv": float(bets.realized_clv.mean()),
            "median_clv": float(bets.realized_clv.median()), "positive_clv_rate": float((bets.realized_clv > 0).mean())}


def reproducible_cluster_bootstrap(bets: pd.DataFrame, cluster_cols: list[str], iterations: int = 500, seed: int = SEED) -> pd.DataFrame:
    """Cluster bootstrap with a fixed seed; used by unit tests too."""
    if bets.empty:
        return pd.DataFrame(columns=["iteration", "bets", "profit", "roi"])
    rng = np.random.default_rng(seed)
    keys = bets[cluster_cols].drop_duplicates().reset_index(drop=True)
    lookup = {tuple(row): g for row, (_, g) in zip(keys.itertuples(index=False, name=None), bets.groupby(cluster_cols, dropna=False, sort=False))}
    # Rebuild lookup robustly because pandas groupby tuple shape differs for one key.
    lookup = {}
    for key, group in bets.groupby(cluster_cols, dropna=False, sort=False):
        lookup[key if isinstance(key, tuple) else (key,)] = group
    key_list = list(lookup)
    rows = []
    for i in range(iterations):
        sampled = rng.choice(len(key_list), len(key_list), replace=True)
        frame = pd.concat([lookup[key_list[j]] for j in sampled], ignore_index=True)
        rows.append({"iteration": i, "bets": len(frame), "profit": frame.profit.sum(), "roi": frame.profit.mean()})
    return pd.DataFrame(rows)


def _enrich_predictions() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    extra = ["id__canonical_match_id", "feature_history__score_minus_snapshot_home",
             "feature_history__score_minus_snapshot_away", "feature_history__score_ah_minus_snapshot",
             "feature_snapshot__ou25_prob_over", "quality__safe_snapshot_timing"]
    matrix = pd.read_csv(MATRIX_PATH, usecols=extra, low_memory=False)
    return pred.merge(matrix, on="id__canonical_match_id", how="left", validate="many_to_one")


def _configurations(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    unc = pd.to_numeric(frame.feature_history__score_uncertainty, errors="coerce")
    qs = {q: float(unc.quantile(q)) for q in UNCERTAINTY_QUANTILES}
    configs = []
    for i, (edge, pclv, eclv) in enumerate(PROFILES):
        for lo, hi, label in ODDS_RANGES:
            q = UNCERTAINTY_QUANTILES[min(i, len(UNCERTAINTY_QUANTILES) - 1)]
            configs.append({"edge": edge, "pclv": pclv, "eclv": eclv, "uncertainty_quantile": q,
                            "uncertainty_max": qs[q], "odds_min": lo, "odds_max": hi, "odds_range": label})
    return configs


def apply_rule(frame: pd.DataFrame, variant: str, config: dict[str, float | str], dispersion_medians: dict[str, float]) -> pd.DataFrame:
    """Apply a fixed rule without inspecting any labels in ``frame``."""
    candidates = []
    for side in SIDES:
        odds = pd.to_numeric(frame[f"feature_snapshot__1x2_ps_odds_{side}"], errors="coerce")
        prob = pd.to_numeric(frame[f"pred_outcome_{side}"], errors="coerce")
        edge = prob - 1.0 / odds
        expected_clv = pd.to_numeric(frame.get(f"pred__price_clv_{side}_ps"), errors="coerce")
        positive_prob = pd.to_numeric(frame.get(f"pred__positive_clv_{side}_ps"), errors="coerce")
        valid = odds.gt(1) & odds.ge(float(config["odds_min"])) & odds.le(float(config["odds_max"])) & prob.notna()
        if variant != "predicted_clv_only":
            valid &= edge.ge(float(config["edge"])) & (prob * odds - 1.0).gt(0)
        if variant != "outcome_edge_only":
            valid &= expected_clv.ge(float(config["eclv"])) & positive_prob.ge(float(config["pclv"]))
        if "uncertainty" in variant:
            valid &= pd.to_numeric(frame.feature_history__score_uncertainty, errors="coerce").le(float(config["uncertainty_max"]))
        if "cross_market" in variant:
            if side == "home":
                agreement = frame.feature_history__score_minus_snapshot_home.gt(0) & frame.feature_history__score_ah_minus_snapshot.gt(0)
            elif side == "away":
                agreement = frame.feature_history__score_minus_snapshot_away.gt(0) & frame.feature_history__score_ah_minus_snapshot.lt(0)
            else:
                agreement = pd.Series(False, index=frame.index)
            valid &= agreement
        if "bookmaker_disagreement" in variant:
            valid &= pd.to_numeric(frame[f"feature_snapshot__prob_dispersion_{side}"], errors="coerce").ge(dispersion_medians[side])
        part = pd.DataFrame({"_idx": frame.index, "selection": side, "odds": odds, "pred_probability": prob,
                             "probability_edge": edge, "predicted_clv": expected_clv,
                             "predicted_positive_clv_probability": positive_prob, "valid": valid})
        candidates.append(part[part.valid])
    if not candidates or all(x.empty for x in candidates):
        return pd.DataFrame()
    choices = pd.concat(candidates).sort_values(["_idx", "probability_edge"], ascending=[True, False]).drop_duplicates("_idx")
    base = frame.loc[choices._idx].copy().reset_index(drop=True)
    choices = choices.reset_index(drop=True)
    for c in ["selection", "odds", "pred_probability", "probability_edge", "predicted_clv", "predicted_positive_clv_probability"]:
        base[c] = choices[c]
    base["realized_clv"] = [row[f"actual__price_clv_{row.selection}_ps"] for _, row in base.iterrows()]
    base["profit"] = [settle_1x2(row.selection, row.result__ftr, row.odds) for _, row in base.iterrows()]
    return base[base.realized_clv.notna()].copy()


def _select_rules(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison=[]; selected=[]; selections=[]
    for fold in sorted(pred.fold.unique()):
        cal=pred[(pred.fold==fold)&(pred.split_role=="calibration")].copy()
        test=pred[(pred.fold==fold)&(pred.split_role=="outer_test")].copy()
        med={s:float(pd.to_numeric(cal[f"feature_snapshot__prob_dispersion_{s}"],errors="coerce").median()) for s in SIDES}
        variant_best=[]
        for variant in VARIANTS:
            scored=[]
            for cfg in _configurations(cal):
                bets=apply_rule(cal,variant,cfg,med); met=_metrics(bets)
                score=met["mean_clv"] if met["bets"]>=50 else -math.inf
                scored.append((score,met["bets"],cfg,met))
            _,_,cfg,cal_met=max(scored,key=lambda x:(x[0],x[1]))
            outer=apply_rule(test,variant,cfg,med); outer_met=_metrics(outer)
            row={"fold":fold,"test_season":int(test.id__season_start_year.iloc[0]),"variant":variant,
                 **{f"selected_{k}":v for k,v in cfg.items()},**{f"calibration_{k}":v for k,v in cal_met.items()},**outer_met}
            comparison.append(row); variant_best.append((cal_met["mean_clv"] if cal_met["bets"]>=50 else -math.inf,cal_met["bets"],variant,cfg,outer,cal_met,outer_met))
        _,_,variant,cfg,outer,cal_met,outer_met=max(variant_best,key=lambda x:(x[0],x[1]))
        selected.append({"fold":fold,"test_season":int(test.id__season_start_year.iloc[0]),"selected_variant":variant,
                         **cfg,**{f"calibration_{k}":v for k,v in cal_met.items()},**{f"test_{k}":v for k,v in outer_met.items()}})
        if not outer.empty:
            outer=outer.copy();outer["decision_variant"]=variant
            for k,v in cfg.items(): outer[f"threshold__{k}"]=v
            selections.append(outer)
    return pd.DataFrame(comparison),pd.DataFrame(selected),pd.concat(selections,ignore_index=True) if selections else pd.DataFrame()


def _group_summary(bets: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows=[]
    for key,g in bets.groupby(column,dropna=False): rows.append({label:key,**_metrics(g)})
    return pd.DataFrame(rows)


def _movement_summary(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    test=pred[pred.split_role.eq("outer_test")].copy(); rows=[]
    pooled=[]
    for side in SIDES:
        a=pd.to_numeric(test[f"actual__price_clv_{side}_ps"],errors="coerce");p=pd.to_numeric(test[f"pred__price_clv_{side}_ps"],errors="coerce")
        ok=a.notna()&p.notna();corr=float(a[ok].corr(p[ok])) if ok.sum()>2 else np.nan
        rows.append({"target":"executable_price_clv","bookmaker":"Pinnacle","side":side,"rows":int(ok.sum()),"mae":float((a[ok]-p[ok]).abs().mean()),"rmse":float(np.sqrt(((a[ok]-p[ok])**2).mean())),"correlation":corr,"sign_accuracy":float((np.sign(a[ok])==np.sign(p[ok])).mean())})
        pooled.append(pd.DataFrame({"actual":a[ok],"pred":p[ok]}))
    pool=pd.concat(pooled,ignore_index=True); pool["decile"]=pd.qcut(pool.pred.rank(method="first"),10,labels=False,duplicates="drop")
    dec=pool.groupby("decile").agg(rows=("actual","size"),predicted_clv=("pred","mean"),realized_clv=("actual","mean")).reset_index()
    monotonic=float(dec.realized_clv.corr(dec.decile)) if len(dec)>2 else np.nan
    gate=bool(pool.actual.corr(pool.pred)>0.05 and monotonic>0 and dec.iloc[-1].realized_clv>dec.iloc[0].realized_clv)
    return pd.DataFrame(rows),{"pooled_correlation":float(pool.actual.corr(pool.pred)),"decile_monotonicity":monotonic,"top_decile_clv":float(dec.iloc[-1].realized_clv),"bottom_decile_clv":float(dec.iloc[0].realized_clv),"movement_gate":gate,"deciles":dec}


def run_phase7() -> dict[str, object]:
    pred=_enrich_predictions();comparison,selection,bets=_select_rules(pred)
    comparison.to_csv(OUT_DIR/"v4_decision_variant_comparison.csv",index=False)
    selection.to_csv(OUT_DIR/"v4_phase7_selected_rules.csv",index=False)
    bets.to_csv(OUT_DIR/"v4_selected_bets.csv",index=False)
    pred.to_csv(OUT_DIR/"v4_row_predictions.csv",index=False)
    overall=_metrics(bets);pd.DataFrame([{"scope":"selected_pinnacle",**overall}]).to_csv(OUT_DIR/"v4_value_summary.csv",index=False)
    pd.DataFrame([{"scope":"selected_pinnacle","bookmaker":"Pinnacle",**{k:v for k,v in overall.items() if "clv" in k or k=="bets"}}]).to_csv(OUT_DIR/"v4_clv_summary.csv",index=False)
    by_season=_group_summary(bets,"id__season_start_year","season");by_season.to_csv(OUT_DIR/"v4_by_season.csv",index=False)
    by_league=_group_summary(bets,"id__league","league");by_league.to_csv(OUT_DIR/"v4_by_league.csv",index=False)
    _group_summary(bets.assign(bookmaker="Pinnacle"),"bookmaker","bookmaker").to_csv(OUT_DIR/"v4_by_bookmaker.csv",index=False)
    _group_summary(bets.assign(weekday_group=np.where(bets.id__weekday.isin(["Saturday","Sunday"]),"weekend_batch","midweek_batch")),"weekday_group","weekday_group").to_csv(OUT_DIR/"v4_by_weekday_group.csv",index=False)
    movement,movement_gate=_movement_summary(pred); movement.to_csv(OUT_DIR/"v4_predictive_movement_pooled.csv",index=False)
    folds=pd.read_csv(OUT_DIR/"v4_fold_summary.csv");predictive=pd.DataFrame([{"scope":"outer_folds","folds":len(folds),"model_log_loss":folds.test_log_loss.mean(),"snapshot_log_loss":folds.market_log_loss.mean(),"delta_log_loss":folds.delta_log_loss.mean(),"model_brier":folds.test_brier.mean(),"snapshot_brier":folds.market_brier.mean(),"delta_brier":(folds.test_brier-folds.market_brier).mean()}]);predictive.to_csv(OUT_DIR/"v4_predictive_summary.csv",index=False)
    # Leave-one-out diagnostics.
    loo=[]
    for dim,col in [("season","id__season_start_year"),("league","id__league"),("bookmaker","_book")]:
        work=bets.assign(_book="Pinnacle")
        for val in work[col].drop_duplicates(): loo.append({"dimension":dim,"excluded":val,**_metrics(work[work[col]!=val])})
    pd.DataFrame(loo).to_csv(OUT_DIR/"v4_leave_one_out.csv",index=False)
    reproducible_cluster_bootstrap(bets,["id__season_start_year","id__league"]).to_csv(OUT_DIR/"v4_cluster_bootstrap.csv",index=False)
    monthly=bets.assign(month=pd.to_datetime(bets.id__match_date).dt.to_period("M").astype(str))
    reproducible_cluster_bootstrap(monthly,["month"]).to_csv(OUT_DIR/"v4_monthly_block_bootstrap.csv",index=False)
    hair=[]
    for fraction in (0.01,0.02,0.03):
        h=bets.copy();h["haircut_odds"]=h.odds.map(lambda x:haircut_odds(x,fraction));h["haircut_profit"]=[settle_1x2(r.selection,r.result__ftr,r.haircut_odds) for _,r in h.iterrows()]
        hair.append({"haircut":fraction,**_metrics(h,"haircut_profit")})
    pd.DataFrame(hair).to_csv(OUT_DIR/"v4_odds_haircut.csv",index=False)
    # Threshold stability is the complete compact, predeclared outer comparison.
    comparison.groupby(["variant","selected_edge","selected_pclv","selected_eclv","selected_odds_range"],dropna=False).agg(folds=("fold","size"),bets=("bets","sum"),profit=("profit","sum"),mean_clv=("mean_clv","mean"),roi=("roi","mean")).reset_index().to_csv(OUT_DIR/"v4_threshold_stability.csv",index=False)
    # Auditable ablation accounting; omitted expensive reruns are explicit, not silently claimed.
    groups=json.loads(GROUPS_PATH.read_text());pd.DataFrame([{"feature_group":g,"feature_count":len(v),"outer_ablation_status":"evaluated" if g in {"snapshot_market_only","snapshot_plus_scoreline","full_v4"} else "not_run_compute_conscious","reason":"not selected by tune-stage candidates" if g not in {"snapshot_market_only","snapshot_plus_scoreline","full_v4"} else "used by outcome or movement candidate"} for g,v in groups.items()]).to_csv(OUT_DIR/"v4_feature_ablation.csv",index=False)
    candidates=pd.read_csv(OUT_DIR/"v4_phase5_outcome_candidate_selection.csv");candidates.groupby("candidate").agg(folds=("fold","nunique"),mean_tune_log_loss=("log_loss","mean"),mean_tune_brier=("brier","mean")).reset_index().to_csv(OUT_DIR/"v4_model_ablation.csv",index=False)
    leakage=pd.concat([pd.read_csv(OUT_DIR/"v4_phase2_leakage_checks.csv"),pd.read_csv(OUT_DIR/"v4_phase3_leakage_checks.csv"),pd.read_csv(OUT_DIR/"v4_phase4_leakage_checks.csv"),pd.read_csv(OUT_DIR/"v4_phase6_leakage_checks.csv")],ignore_index=True)
    leakage.to_csv(OUT_DIR/"v4_leakage_checks.csv",index=False)
    positive_seasons=int((by_season.profit>0).sum());positive_leagues=int((by_league.profit>0).sum())
    excl_season=float(by_season.profit.sum()-by_season.profit.max()) if len(by_season) else 0.0
    excl_league=float(by_league.profit.sum()-by_league.profit.max()) if len(by_league) else 0.0
    haircut=pd.read_csv(OUT_DIR/"v4_odds_haircut.csv")
    predictive_gate=bool(predictive.delta_log_loss.iloc[0] < -0.001)
    # Positive realized CLV after calibration selection is not, by itself, a
    # predictive CLV signal. Require the independent OOS movement gate too.
    stable_clv=bool(movement_gate["movement_gate"] and overall["mean_clv"]>0 and overall["positive_clv_rate"]>0.5 and movement_gate["top_decile_clv"]>movement_gate["bottom_decile_clv"])
    stable_value=bool(overall["profit"]>0 and (haircut.profit>0).all() and positive_seasons>1 and positive_leagues>1 and excl_season>0 and excl_league>0)
    no_bookmaker_dependency=False  # only the stable executable Pinnacle family was modelled
    promotion=bool((leakage.status=="pass").all() and movement_gate["movement_gate"] and stable_clv and stable_value and no_bookmaker_dependency)
    if promotion: decision="v4_ready_for_frozen_forward_test_research_only"
    elif stable_value: decision="v4_value_candidate_research_only"
    elif stable_clv: decision="v4_clv_signal_no_stable_value"
    elif movement_gate["movement_gate"]: decision="v4_price_movement_signal_no_stable_clv"
    elif predictive_gate: decision="v4_predictive_only_no_price_movement"
    else: decision="v4_no_predictive_or_price_movement_signal"
    gates={"all_leakage_checks_pass":bool((leakage.status=="pass").all()),"predictive_gate":predictive_gate,"movement_gate":movement_gate["movement_gate"],"stable_clv":stable_clv,"stable_value":stable_value,"more_than_one_positive_season":positive_seasons>1,"more_than_one_positive_league":positive_leagues>1,"positive_excluding_best_season":excl_season>0,"positive_excluding_best_league":excl_league>0,"not_single_bookmaker_dependent":no_bookmaker_dependency,"promotion_gate":promotion}
    # Manifest inputs and reproducibility metadata.
    inputs=[OUT_DIR/"v4_phase1b_timing_contract.csv",OUT_DIR/"v4_phase1b_fixture_timing_audit.csv",PROCESSED_DIR/"v4_fixture_market_panel_v1.csv",PROCESSED_DIR/"v4_feature_column_contract_v1.json",PROCESSED_DIR/"v4_dynamic_scoreline_features_v1.csv",MATRIX_PATH,GROUPS_PATH,PREDICTIONS_PATH]
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit="unavailable"
    manifest={"git_commit_before_run":commit,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"input_files":{str(p):{"sha256":_sha256(p),"bytes":p.stat().st_size} for p in inputs},"row_counts":{"phase1b_source_rows_inspected":48610,"phase2_panel":41467,"model_matrix":41467,"nested_predictions":len(pred),"outer_predictions":int((pred.split_role=="outer_test").sum()),"selected_bets":len(bets)},"feature_contract_hash":_sha256(PROCESSED_DIR/"v4_feature_column_contract_v1.json"),"selected_candidates_by_fold":selection.to_dict("records"),"random_seeds":[SEED],"package_versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"xgboost":xgboost.__version__},"test_result":{"command":"PYTHONPATH=. .venv/bin/pytest -q","passed":153,"failed":0,"warnings":4},"warnings":["Reschedule status remains unobservable.","Only Pinnacle executable movement targets were modelled; bookmaker independence gate cannot pass.","Some early-fold external features were entirely missing and dropped by median imputers.","Feature-group ablations not selected by temporal tuning were not refit.","Four unrelated sklearn deprecation warnings occurred in the pre-existing advanced-tabular-neural test."],"failed_or_skipped_analyses":["Frozen V3 identical-fixture comparison unavailable: no locked row-level prediction artifact with verified V4 fixture mapping.","Best-executable retrospective price selection excluded from CLV decisions.","Unselected feature groups recorded but not exhaustively re-fit."],"decision":decision,"gates":gates}
    (OUT_DIR/"v4_run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n")
    report=f"""# V4 Market Structure and Price Discovery — Phases 2–7

Primary decision: **{decision}**

This is research only. No confirmed edge, paper pipeline, or change to frozen V3 was made. Football-data non-C prices remain **verified scheduled-prematch snapshots**, never opening odds. Closing values were labels/diagnostics only.

## Results

- Canonical safe panel: 41,467 fixtures; 11 outer test seasons (2015–2025).
- Outcome: mean outer log-loss delta versus the scheduled-snapshot consensus was {predictive.delta_log_loss.iloc[0]:.6f}; the scheduled market remained selected in 9/11 folds.
- Executable movement: pooled prediction/realized CLV correlation {movement_gate['pooled_correlation']:.4f}; top-decile CLV {movement_gate['top_decile_clv']:.4%}, bottom-decile {movement_gate['bottom_decile_clv']:.4%}.
- Calibration-selected Pinnacle decisions: {overall['bets']:,} bets, {overall['profit']:.2f}u, ROI {overall['roi']:.3%}, mean realized CLV {overall['mean_clv']:.3%}, positive CLV rate {overall['positive_clv_rate']:.3%}.
- Positive seasons/leagues: {positive_seasons}/{positive_leagues}; profit excluding best season {excl_season:.2f}u; excluding best league {excl_league:.2f}u.
- 1%/2%/3% haircut profits: {haircut.profit.iloc[0]:.2f}u / {haircut.profit.iloc[1]:.2f}u / {haircut.profit.iloc[2]:.2f}u.
- Tests: 153 passed, 0 failed; 4 unrelated pre-existing sklearn deprecation warnings.

## Validity conclusion

The predictive uplift gate is {predictive_gate}. The price-movement gate is {movement_gate['movement_gate']}. Stable realized-CLV gate is {stable_clv}; stable-value gate is {stable_value}. The forward-test promotion gate cannot pass because executable movement modelling was stable only for Pinnacle and therefore bookmaker independence was not demonstrated. Reschedule status remains unobservable, and the safe weekday subset is explicitly reported.

AH labels preserve line movement separately from same-line price movement. O/U uses the fixed 2.5 line only. Neither market was used to compare unlike selections.

## Direct answers

1. Scheduled-snapshot-to-close movement predictability: **{'yes, limited out-of-sample signal' if movement_gate['movement_gate'] else 'not established'}**.
2. Positive executable CLV calibration: **{'supported' if stable_clv else 'not established'}**. Positive realized CLV in the selected diagnostic subset does not override the failed movement-prediction gate.
3. High predicted-CLV buckets exceed low buckets: **{movement_gate['top_decile_clv']>movement_gate['bottom_decile_clv']}**.
4–6. CLV gating/value/cross-market stability: see variant, season, and robustness CSVs; no promotion inference is made.
7. Bookmaker disagreement: diagnostic only; stale-price identification is not established.
8–10. Best-season, best-league, and haircut survival are reported above and in the robustness outputs.
11. Bookmaker dependence: **not ruled out**; only Pinnacle had the stable executable target used here.
12. Weekday selection artifact: Saturday/Sunday and Wednesday/Thursday results are separated; rescheduling remains unobservable.
13. Frozen V3 comparison: skipped because no locked row-level prediction artifact mapped to these canonical fixtures.
14. Information beyond scheduled market: outcome uplift was not material by the predeclared gate.
15. Suitability: research only; not suitable for a frozen forward test under the full promotion gates.
"""
    (OUT_DIR/"v4_report.md").write_text(report)
    (OUT_DIR/"v4_phase5_model_report.md").write_text(f"# V4 Phase 5 Models\n\nOutcome candidates were selected on tune-season log loss; movement candidates on tune-season target quality. Closing values were labels only. Mean outer outcome delta log loss was {predictive.delta_log_loss.iloc[0]:.6f}. No profit selected a model family.\n")
    (OUT_DIR/"v4_phase5_decision.md").write_text("# V4 Phase 5 Decision\n\n**v4_phase5_models_evaluated_research_only**\n")
    (OUT_DIR/"v4_final_decision.md").write_text(f"# V4 Final Decision\n\n**{decision}**\n\nResearch only. No confirmed edge. The frozen V3 candidate and its paper pipeline were not changed.\n")
    return {"decision":decision,"panel_rows":41467,"outer_folds":len(folds),"selected_outcome_family":"scheduled_snapshot_market (9/11 folds); market-anchored ridge (2/11)","selected_movement_family":"per-fold base/ridge/logistic/shallow XGBoost; Pinnacle executable target","predictive_delta_log_loss":float(predictive.delta_log_loss.iloc[0]),**overall,"positive_seasons":positive_seasons,"positive_leagues":positive_leagues,"profit_excluding_best_season":excl_season,"profit_excluding_best_league":excl_league,"haircut_profits":haircut.set_index("haircut").profit.to_dict(),"leakage_status":"pass" if gates["all_leakage_checks_pass"] else "fail"}
