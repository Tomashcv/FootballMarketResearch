"""Conservative nested-temporal price-movement research for V5 BASIC LTP."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss, log_loss, mean_absolute_error, mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .core import SIDES
from .pipeline import REPORTS, SEED, staleness_bucket, write_csv, write_md

DECISIONS = {"t24h": "t5m", "t6h": "t5m", "t1h": "t5m", "t15m": "last_preplay"}


def build_research_rows(cutoffs: pd.DataFrame) -> pd.DataFrame:
    """Create features only from states at or earlier than each decision time."""
    ordered_labels = ["t72h","t48h","t24h","t12h","t6h","t3h","t1h","t30m","t15m","t5m","t1m","last_preplay"]
    rank = {x:i for i,x in enumerate(ordered_labels)}
    rows = []
    for market_id, market in cutoffs.groupby("market_id", sort=True):
        states = {r.cutoff: r for r in market.itertuples()}
        for decision, future in DECISIONS.items():
            if decision not in states or future not in states:
                continue
            current, target = states[decision], states[future]
            previous = [states[x] for x in ordered_labels if x in states and rank[x] < rank[decision]]
            for side in SIDES:
                current_p = float(getattr(current, f"{side}_ltp_probability_proxy"))
                future_p = float(getattr(target, f"{side}_ltp_probability_proxy"))
                current_ltp = float(getattr(current, f"{side}_ltp"))
                future_ltp = float(getattr(target, f"{side}_ltp"))
                p_path = [float(getattr(x, f"{side}_ltp_probability_proxy")) for x in previous] + [current_p]
                ltp_path = [float(getattr(x, f"{side}_ltp")) for x in previous] + [current_ltp]
                prev = previous[-1] if previous else None
                prev_p = float(getattr(prev, f"{side}_ltp_probability_proxy")) if prev else current_p
                prev_ltp = float(getattr(prev, f"{side}_ltp")) if prev else current_ltp
                start = pd.Timestamp(current.market_start_utc)
                cutoff_at = pd.Timestamp(current.cutoff_utc)
                shift = future_p - current_p
                rows.append({
                    "market_id":market_id, "canonical_fixture_id":current.canonical_fixture_id,
                    "season":int(current.season), "market_start_utc":current.market_start_utc,
                    "decision_horizon":decision, "target_horizon":future, "side":side,
                    "current_probability_proxy":current_p, "current_ltp":current_ltp,
                    "previous_probability_proxy":prev_p, "probability_change":current_p-prev_p,
                    "log_price_change":np.log(current_ltp)-np.log(prev_ltp),
                    "path_volatility":float(np.std(np.diff(p_path))) if len(p_path)>1 else 0.0,
                    "maximum_movement":float(max(p_path)-min(p_path)),
                    "distance_from_path_high":float(max(p_path)-current_p),
                    "distance_from_path_low":float(current_p-min(p_path)),
                    "number_previous_valid":len(previous),
                    "runner_staleness_seconds":float(getattr(current,f"{side}_staleness_seconds")),
                    "max_runner_staleness_seconds":max(float(getattr(current,f"{s}_staleness_seconds")) for s in SIDES),
                    "proxy_overround":float(current.proxy_overround),
                    "market_entropy":float(-sum(float(getattr(current,f"{s}_ltp_probability_proxy"))*np.log(float(getattr(current,f"{s}_ltp_probability_proxy"))) for s in SIDES)),
                    "time_to_start_hours":float((start-cutoff_at).total_seconds()/3600),
                    "day_of_week":start.dayofweek, "month":start.month,
                    "future_probability_proxy":future_p, "probability_shift":shift,
                    "positive_movement":int(shift>0),
                    "ltp_change":future_ltp-current_ltp, "ltp_direction_down":int(future_ltp<current_ltp),
                })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["staleness_bucket"] = frame.runner_staleness_seconds.map(staleness_bucket)
        frame = frame.sort_values(["market_start_utc","market_id","decision_horizon","side"], kind="stable").reset_index(drop=True)
    return frame


FEATURES = [
    "current_probability_proxy","current_ltp","previous_probability_proxy",
    "probability_change","log_price_change","path_volatility","maximum_movement",
    "distance_from_path_high","distance_from_path_low","number_previous_valid",
    "runner_staleness_seconds","max_runner_staleness_seconds","proxy_overround",
    "market_entropy","time_to_start_hours","day_of_week","month",
]


def regression_metrics(y, pred) -> dict:
    return {
        "mae":mean_absolute_error(y,pred), "rmse":mean_squared_error(y,pred)**0.5,
        "correlation":np.corrcoef(y,pred)[0,1] if len(y)>2 and np.std(pred)>0 and np.std(y)>0 else np.nan,
        "directional_accuracy":np.mean((pred>0)==(np.asarray(y)>0)),
    }


def _ece(y, p, bins=10) -> float:
    table = pd.DataFrame({"y":np.asarray(y),"p":np.asarray(p)})
    table["bin"] = pd.cut(table.p, np.linspace(0,1,bins+1), include_lowest=True)
    return float(sum(len(g)/len(table)*abs(g.y.mean()-g.p.mean()) for _,g in table.groupby("bin",observed=True)))


def run_research(cutoffs: pd.DataFrame, mapped: pd.DataFrame) -> str:
    data = build_research_rows(cutoffs)
    if data.empty or data.season.nunique() < 4:
        return _empty_outputs("insufficient complete temporal seasons")
    predictions, folds = [], []
    seasons = sorted(data.season.unique())
    # Expanding leave-one-season-out temporal folds: each eligible season is
    # untouched and only earlier seasons may contribute to its model.
    test_seasons = seasons[3:]
    for horizon in DECISIONS:
        subset_h = data[data.decision_horizon.eq(horizon)]
        for test_season in test_seasons:
            train = subset_h[subset_h.season < test_season].copy()
            test = subset_h[subset_h.season.eq(test_season)].copy()
            if len(train)<100 or len(test)<20: continue
            # Seven-day date purge before the test boundary.
            boundary = pd.to_datetime(test.market_start_utc,utc=True,format="mixed").min()
            train = train[pd.to_datetime(train.market_start_utc,utc=True,format="mixed") < boundary-pd.Timedelta(days=7)]
            yte = test.probability_shift
            # Nested chronology: historical fit -> tuning season -> calibration
            # season -> untouched outer test. Whole seasons provide additional purge.
            prior_seasons=sorted(train.season.unique())
            if len(prior_seasons)<3: continue
            calibration_season=prior_seasons[-1]; tuning_season=prior_seasons[-2]
            history=train[train.season<tuning_season]; tuning=train[train.season.eq(tuning_season)]
            calibration=train[train.season.eq(calibration_season)]
            if len(history)<100 or len(tuning)<20 or len(calibration)<20: continue
            best_alpha, best_mae = 1.0, np.inf
            for alpha in (0.1,1.0,10.0):
                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(history[FEATURES].fillna(0),history.probability_shift)
                score=mean_absolute_error(tuning.probability_shift,model.predict(tuning[FEATURES].fillna(0)))
                if score<best_mae: best_alpha,best_mae=alpha,score
            xgb=XGBRegressor(n_estimators=100,max_depth=2,learning_rate=.03,subsample=.8,colsample_bytree=.8,
                             objective="reg:squarederror",random_state=SEED,n_jobs=1)
            xgb.fit(history[FEATURES].fillna(0),history.probability_shift)
            xgb_mae=mean_absolute_error(tuning.probability_shift,xgb.predict(tuning[FEATURES].fillna(0)))
            selected_model="xgboost" if xgb_mae<best_mae else "ridge"
            fit_data=pd.concat([history,tuning],ignore_index=True)
            if selected_model=="xgboost":
                reg=XGBRegressor(n_estimators=100,max_depth=2,learning_rate=.03,subsample=.8,colsample_bytree=.8,
                                 objective="reg:squarederror",random_state=SEED,n_jobs=1).fit(fit_data[FEATURES].fillna(0),fit_data.probability_shift)
            else:
                reg=make_pipeline(StandardScaler(),Ridge(alpha=best_alpha)).fit(fit_data[FEATURES].fillna(0),fit_data.probability_shift)
            # Affine calibration learned only on the dedicated calibration season.
            cal_raw=reg.predict(calibration[FEATURES].fillna(0))
            slope,intercept=np.polyfit(cal_raw,calibration.probability_shift,1) if np.std(cal_raw)>0 else (0.0,calibration.probability_shift.mean())
            pred=slope*reg.predict(test[FEATURES].fillna(0))+intercept
            ytr=fit_data.probability_shift
            baseline_zero=np.zeros(len(test)); baseline_mean=np.full(len(test),ytr.mean())
            momentum=test.probability_change.to_numpy(); mean_reversion=-momentum
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=SEED)).fit(fit_data[FEATURES].fillna(0),(ytr>0).astype(int))
            cal_prob=np.clip(clf.predict_proba(calibration[FEATURES].fillna(0))[:,1],1e-6,1-1e-6)
            calibrator=LogisticRegression(C=10,max_iter=1000,random_state=SEED).fit(np.log(cal_prob/(1-cal_prob)).reshape(-1,1),(calibration.probability_shift>0).astype(int))
            raw_prob=np.clip(clf.predict_proba(test[FEATURES].fillna(0))[:,1],1e-6,1-1e-6)
            prob=calibrator.predict_proba(np.log(raw_prob/(1-raw_prob)).reshape(-1,1))[:,1]
            for i, (_, row) in enumerate(test.iterrows()):
                rec=row.to_dict(); rec.update(predicted_shift=pred[i],predicted_positive_probability=prob[i],
                                               baseline_no_movement=0.0,baseline_historical_mean=ytr.mean(),
                                               baseline_momentum=momentum[i],baseline_mean_reversion=mean_reversion[i],
                                               selected_model=selected_model,ridge_alpha=best_alpha,outer_test_season=test_season,
                                               tuning_season=tuning_season,calibration_season=calibration_season)
                predictions.append(rec)
            metrics=regression_metrics(yte,pred)
            folds.append({"decision_horizon":horizon,"test_season":test_season,"train_rows":len(train),"test_rows":len(test),
                          "tuning_season":tuning_season,"calibration_season":calibration_season,
                          "selected_model":selected_model,"ridge_alpha":best_alpha,"ridge_tuning_mae":best_mae,"xgboost_tuning_mae":xgb_mae,**metrics,
                          "baseline_no_movement_mae":mean_absolute_error(yte,baseline_zero),
                          "baseline_historical_mean_mae":mean_absolute_error(yte,baseline_mean),
                          "baseline_momentum_mae":mean_absolute_error(yte,momentum),
                          "baseline_mean_reversion_mae":mean_absolute_error(yte,mean_reversion),
                          "log_loss":log_loss((yte>0).astype(int),prob,labels=[0,1]),
                          "brier":brier_score_loss((yte>0).astype(int),prob),"ece":_ece((yte>0).astype(int),prob),
                          "roc_auc":roc_auc_score((yte>0).astype(int),prob) if yte.gt(0).nunique()>1 else np.nan})
    pred = pd.DataFrame(predictions); fold = pd.DataFrame(folds)
    if pred.empty: return _empty_outputs("no eligible outer folds")
    pred.to_parquet(REPORTS/"v5_row_predictions.parquet",index=False)
    write_csv(REPORTS/"v5_fold_summary.csv",fold)
    summary=[]
    for model_col, label in [("predicted_shift","selected_model"),("baseline_no_movement","no_movement"),("baseline_historical_mean","historical_mean"),("baseline_momentum","momentum"),("baseline_mean_reversion","mean_reversion")]:
        summary.append({"model":label,**regression_metrics(pred.probability_shift,pred[model_col]),"rows":len(pred)})
    robustness_slices = {
        "selected_model_no_stale_over_120m": pred[pred.runner_staleness_seconds<=7200],
        "selected_model_minimum_activity_2": pred[pred.number_previous_valid>=2],
        "selected_model_pre_covid": pred[pred.season<2020],
        "selected_model_post_covid": pred[pred.season>=2020],
    }
    for label, sample in robustness_slices.items():
        if len(sample): summary.append({"model":label,**regression_metrics(sample.probability_shift,sample.predicted_shift),"rows":len(sample)})
    write_csv(REPORTS/"v5_predictive_summary.csv",summary)
    _group_metrics(pred,"season",REPORTS/"v5_by_season.csv")
    _group_metrics(pred,"decision_horizon",REPORTS/"v5_by_horizon.csv")
    _group_metrics(pred,"side",REPORTS/"v5_by_side.csv")
    _group_metrics(pred,"staleness_bucket",REPORTS/"v5_by_staleness.csv")
    deciles=[]
    for keys,g in pred.groupby(["decision_horizon","side"],sort=True):
        if len(g)<20: continue
        g=g.copy();g["prediction_decile"]=pd.qcut(g.predicted_shift,10,labels=False,duplicates="drop")+1
        for dec,d in g.groupby("prediction_decile"):
            deciles.append({"decision_horizon":keys[0],"side":keys[1],"prediction_decile":dec,"rows":len(d),"mean_target_movement":d.probability_shift.mean(),"mean_prediction":d.predicted_shift.mean()})
    write_csv(REPORTS/"v5_decile_analysis.csv",deciles)
    # Deterministic monthly block bootstrap of model-minus-zero-baseline MAE.
    rng=np.random.default_rng(SEED); pred["month_block"]=pd.to_datetime(pred.market_start_utc,utc=True,format="mixed").dt.to_period("M").astype(str)
    blocks=list(pred.month_block.unique()); boots=[]
    for b in range(500):
        chosen=rng.choice(blocks,len(blocks),replace=True); sample=pd.concat([pred[pred.month_block.eq(x)] for x in chosen],ignore_index=True)
        boots.append({"replicate":b,"model_minus_zero_mae":mean_absolute_error(sample.probability_shift,sample.predicted_shift)-mean_absolute_error(sample.probability_shift,np.zeros(len(sample)))})
    write_csv(REPORTS/"v5_bootstrap.csv",boots)
    best=fold.sort_values("mae").iloc[0]
    actual_test_seasons=sorted(int(x) for x in fold.test_season.unique())
    model_mae=float(pd.DataFrame(summary).query("model=='selected_model'").mae.iloc[0]); zero_mae=float(pd.DataFrame(summary).query("model=='no_movement'").mae.iloc[0])
    improvement=(zero_mae-model_mae)/zero_mae if zero_mae else 0
    stable=float(np.mean(pd.DataFrame(boots).model_minus_zero_mae<0))
    decision="v5_betfair_weak_price_movement_signal_research_only" if improvement>0 and stable>=0.8 else "v5_betfair_no_price_movement_signal"
    if improvement>=0.05 and stable>=0.95:
        decision="v5_betfair_price_movement_candidate_research_only"
    write_md(REPORTS/"v5_report.md","V5 Betfair BASIC price-path research",[
        f"Final research decision: `{decision}`.",
        f"Strict outer test seasons: {', '.join(map(str,actual_test_seasons))}. Each uses historical fit, tuning season, calibration season, then untouched test. Best observed fold: {best.decision_horizon}, {best.selected_model}.",
        f"Aggregate selected-model MAE {model_mae:.6f} versus no-movement MAE {zero_mae:.6f}; bootstrap probability of lower MAE {stable:.3f}.",
        "A pre-COVID outer-test comparison was not estimable under the minimum-history and nested tuning/calibration rules: early mapped seasons (2015–2016) had only 24 and 9 approved fixtures. This robustness analysis is explicitly skipped rather than weakened.",
        "Targets and features are normalized inverse-LTP probability proxies and last-traded-price movements. They are not executable CLV, available back/lay quotes, verified liquidity, betting profit, or a strategy.",
        "No confirmed edge is claimed. Betfair ADVANCED data validation is required before any strategy work.",
    ])
    return decision


def _group_metrics(frame, column, path):
    rows=[]
    for key,g in frame.groupby(column,dropna=False,sort=True): rows.append({column:key,"rows":len(g),**regression_metrics(g.probability_shift,g.predicted_shift)})
    write_csv(path,rows)


def _empty_outputs(reason):
    empty={"v5_predictive_summary.csv":["model","mae"],"v5_fold_summary.csv":["decision_horizon","test_season"],"v5_by_season.csv":["season","rows"],"v5_by_horizon.csv":["decision_horizon","rows"],"v5_by_side.csv":["side","rows"],"v5_by_staleness.csv":["staleness_bucket","rows"],"v5_decile_analysis.csv":["prediction_decile","mean_target_movement"],"v5_bootstrap.csv":["replicate","model_minus_zero_mae"]}
    for name,cols in empty.items(): write_csv(REPORTS/name,[],cols)
    pd.DataFrame().to_parquet(REPORTS/"v5_row_predictions.parquet",index=False)
    write_md(REPORTS/"v5_report.md","V5 Betfair BASIC price-path research",[f"Model training skipped: {reason}.","No confirmed edge is claimed."])
    return "v5_betfair_no_price_movement_signal"
