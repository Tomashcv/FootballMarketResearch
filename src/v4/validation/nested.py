"""Compact strict nested temporal validation for V4 outcome and movement models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.v4.data.phase1b_audit import OUT_DIR
from src.v4.features.model_matrix import GROUPS_PATH, MATRIX_PATH, PROCESSED_DIR
from src.v4.models.dynamic_scoreline import ece, multiclass_brier


PREDICTIONS_PATH = PROCESSED_DIR / "v4_nested_predictions_v1.csv"
SEED = 20260711


def temporal_folds(seasons: list[int]) -> list[dict[str, object]]:
    seasons = sorted(set(seasons))
    folds=[]
    for i in range(3,len(seasons)):
        folds.append({"fold":len(folds)+1,"train_seasons":seasons[:i-2],"tune_season":seasons[i-2],"calibration_season":seasons[i-1],"test_season":seasons[i]})
    return folds


def assert_fold_isolation(fold: dict[str,object]) -> None:
    train=list(fold["train_seasons"]); tune=int(fold["tune_season"]); cal=int(fold["calibration_season"]); test=int(fold["test_season"])
    assert train and max(train)<tune<cal<test


def numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).to_numpy(float)


def market_probs(frame: pd.DataFrame) -> np.ndarray:
    return frame[["feature_snapshot__consensus_prob_home","feature_snapshot__consensus_prob_draw","feature_snapshot__consensus_prob_away"]].to_numpy(float)


def score_probs(frame: pd.DataFrame) -> np.ndarray:
    return frame[["feature_history__score_prob_home","feature_history__score_prob_draw","feature_history__score_prob_away"]].to_numpy(float)


def outcome_model(name: str, seed: int=SEED):
    if name=="market_anchored_multinomial_ridge":
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=0.1,max_iter=500,random_state=seed))
    if name=="shallow_xgboost_market_residual":
        return XGBClassifier(n_estimators=60,max_depth=2,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=10,reg_alpha=1,min_child_weight=20,n_jobs=2,random_state=seed,eval_metric="mlogloss")
    raise ValueError(name)


def _fit_predict_outcome(name: str, train: pd.DataFrame, target: np.ndarray, predict: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, object|None]:
    if name=="scheduled_snapshot_market": return market_probs(predict),None
    if name=="dynamic_scoreline": return score_probs(predict),None
    model=outcome_model(name);model.fit(numeric_matrix(train,columns),target);return model.predict_proba(numeric_matrix(predict,columns)),model


def _predict_fitted(model: object|None,name: str,frame: pd.DataFrame,columns:list[str])->np.ndarray:
    if name=="scheduled_snapshot_market": return market_probs(frame)
    if name=="dynamic_scoreline": return score_probs(frame)
    return model.predict_proba(numeric_matrix(frame,columns))


def outcome_metrics(y: np.ndarray,p: np.ndarray)->dict[str,float]:
    return {"log_loss":log_loss(y,p,labels=[0,1,2]),"brier":multiclass_brier(y,p),"ece":ece(y,p),"accuracy":accuracy_score(y,p.argmax(1))}


def movement_regressor(name:str):
    if name=="regularized_linear": return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=10.0))
    if name=="shallow_xgboost": return XGBRegressor(n_estimators=60,max_depth=2,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=10,reg_alpha=1,min_child_weight=20,n_jobs=2,random_state=SEED,objective="reg:squarederror")
    raise ValueError(name)


def movement_classifier():
    return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=0.1,max_iter=500,random_state=SEED))


def regression_metrics(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    return {"mae":mean_absolute_error(y,p),"rmse":math.sqrt(mean_squared_error(y,p)),"sign_accuracy":float((np.sign(y)==np.sign(p)).mean()),"correlation":float(np.corrcoef(y,p)[0,1]) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan}


def run_nested() -> dict[str,object]:
    data=pd.read_csv(MATRIX_PATH,low_memory=False);groups=json.loads(GROUPS_PATH.read_text())
    required_market=["feature_snapshot__consensus_prob_home","feature_snapshot__consensus_prob_draw","feature_snapshot__consensus_prob_away"]
    data=data[data["quality__valid_result"].astype(bool) & data[required_market].notna().all(axis=1)].copy();data["_y"]=data["result__ftr"].map({"H":0,"D":1,"A":2})
    seasons=sorted(data["id__season_start_year"].unique());folds=temporal_folds(seasons)
    outcome_candidates=["scheduled_snapshot_market","dynamic_scoreline","market_anchored_multinomial_ridge","shallow_xgboost_market_residual"]
    feature_candidates={"scheduled_snapshot_market":"snapshot_market_only","dynamic_scoreline":"snapshot_plus_scoreline","market_anchored_multinomial_ridge":"snapshot_plus_scoreline","shallow_xgboost_market_residual":"snapshot_plus_scoreline"}
    move_features=groups["full_v4"]
    target_specs={
        "prob_shift_home":("label_close__prob_shift_home","regression"),"prob_shift_draw":("label_close__prob_shift_draw","regression"),"prob_shift_away":("label_close__prob_shift_away","regression"),
        "price_clv_home_ps":("label_close__price_clv_home__ps","regression"),"price_clv_draw_ps":("label_close__price_clv_draw__ps","regression"),"price_clv_away_ps":("label_close__price_clv_away__ps","regression"),
        "positive_clv_home_ps":("label_close__positive_price_clv_home__ps","classification"),"positive_clv_draw_ps":("label_close__positive_price_clv_draw__ps","classification"),"positive_clv_away_ps":("label_close__positive_price_clv_away__ps","classification"),
        "ah_line_shift":("label_close__ah_line_shift","regression"),"ah_same_line_away_shift":("label_close__ah_same_line_prob_shift_away","regression"),"ou25_over_shift":("label_close__ou25_prob_shift_over","regression"),
    }
    fold_rows=[];candidate_rows=[];movement_rows=[];pred_rows=[]
    for fold in folds:
        assert_fold_isolation(fold);test=int(fold["test_season"]);tune=int(fold["tune_season"]);cal=int(fold["calibration_season"])
        train=data[data.id__season_start_year.isin(fold["train_seasons"])];tune_df=data[data.id__season_start_year.eq(tune)];cal_df=data[data.id__season_start_year.eq(cal)];test_df=data[data.id__season_start_year.eq(test)]
        ytrain=train._y.to_numpy(int);ytune=tune_df._y.to_numpy(int);ycal=cal_df._y.to_numpy(int);ytest=test_df._y.to_numpy(int)
        tune_scores=[]
        for name in outcome_candidates:
            cols=groups[feature_candidates[name]];pp,_=_fit_predict_outcome(name,train,ytrain,tune_df,cols);m=outcome_metrics(ytune,pp)
            tune_scores.append((m["log_loss"],m["brier"],name));candidate_rows.append({"fold":fold["fold"],"test_season":test,"candidate":name,"feature_group":feature_candidates[name],"selection_split":"tune","rows":len(tune_df),**m})
        selected=sorted(tune_scores)[0][2];cols=groups[feature_candidates[selected]]
        fit_df=pd.concat([train,tune_df],ignore_index=True);yfit=fit_df._y.to_numpy(int)
        pcal,model=_fit_predict_outcome(selected,fit_df,yfit,cal_df,cols);ptest=_predict_fitted(model,selected,test_df,cols)
        # Calibration chooses only a conservative blend back to market.
        weights=[0.0,0.5,1.0];cal_market=market_probs(cal_df);test_market=market_probs(test_df)
        losses=[log_loss(ycal,w*pcal+(1-w)*cal_market,labels=[0,1,2]) for w in weights];weight=weights[int(np.argmin(losses))]
        pcal_final=weight*pcal+(1-weight)*cal_market;ptest_final=weight*ptest+(1-weight)*test_market
        om=outcome_metrics(ytest,ptest_final);base=outcome_metrics(ytest,test_market)
        fold_record={"fold":fold["fold"],"train_seasons":"|".join(map(str,fold["train_seasons"])),"tune_season":tune,"calibration_season":cal,"test_season":test,"train_rows":len(train),"tune_rows":len(tune_df),"calibration_rows":len(cal_df),"test_rows":len(test_df),"selected_outcome_model":selected,"selected_feature_group":feature_candidates[selected],"calibration_market_blend_weight":weight,"test_log_loss":om["log_loss"],"market_log_loss":base["log_loss"],"delta_log_loss":om["log_loss"]-base["log_loss"],"test_brier":om["brier"],"market_brier":base["brier"],"test_ece":om["ece"],"market_ece":base["ece"]}
        # Movement candidates selected on tune, then refit train+tune. Calibration is reserved for Phase 7 gates.
        movement_test={};movement_cal={};selected_move={}
        for target_name,(target_col,kind) in target_specs.items():
            if target_col not in data: continue
            tr=train[train[target_col].notna()];tu=tune_df[tune_df[target_col].notna()];fi=fit_df[fit_df[target_col].notna()];ca=cal_df[cal_df[target_col].notna()];te=test_df[test_df[target_col].notna()]
            if len(tr)<200 or len(tu)<30 or len(te)<30: continue
            if kind=="regression":
                candidates=["unconditional_base","regularized_linear"]+(["shallow_xgboost"] if target_name=="price_clv_away_ps" else [])
                scores=[]
                for name in candidates:
                    if name=="unconditional_base": pred=np.repeat(tr[target_col].mean(),len(tu))
                    else: mdl=movement_regressor(name);mdl.fit(numeric_matrix(tr,move_features),tr[target_col]);pred=mdl.predict(numeric_matrix(tu,move_features))
                    met=regression_metrics(tu[target_col].to_numpy(float),pred);scores.append((met["rmse"],name));movement_rows.append({"fold":fold["fold"],"test_season":test,"target":target_name,"candidate":name,"selection_split":"tune","rows":len(tu),**met})
                choice=sorted(scores)[0][1]
                if choice=="unconditional_base": pca=np.repeat(fi[target_col].mean(),len(ca));pte=np.repeat(fi[target_col].mean(),len(te))
                else: mdl=movement_regressor(choice);mdl.fit(numeric_matrix(fi,move_features),fi[target_col]);pca=mdl.predict(numeric_matrix(ca,move_features));pte=mdl.predict(numeric_matrix(te,move_features))
                movement_cal[target_name]=(ca,pca);movement_test[target_name]=(te,pte);selected_move[target_name]=choice
                met=regression_metrics(te[target_col].to_numpy(float),pte);movement_rows.append({"fold":fold["fold"],"test_season":test,"target":target_name,"candidate":choice,"selection_split":"outer_test","rows":len(te),**met})
            else:
                if tr[target_col].nunique()<2 or fi[target_col].nunique()<2: continue
                rate=tr[target_col].astype(float).mean();base_pred=np.repeat(rate,len(tu));base_b=brier_score_loss(tu[target_col].astype(int),base_pred)
                mdl=movement_classifier();mdl.fit(numeric_matrix(tr,move_features),tr[target_col].astype(int));lp=mdl.predict_proba(numeric_matrix(tu,move_features))[:,1];lb=brier_score_loss(tu[target_col].astype(int),lp)
                choice="regularized_logistic" if lb<base_b else "unconditional_base";selected_move[target_name]=choice
                if choice=="regularized_logistic": mdl=movement_classifier();mdl.fit(numeric_matrix(fi,move_features),fi[target_col].astype(int));pca=mdl.predict_proba(numeric_matrix(ca,move_features))[:,1];pte=mdl.predict_proba(numeric_matrix(te,move_features))[:,1]
                else: pca=np.repeat(fi[target_col].astype(float).mean(),len(ca));pte=np.repeat(fi[target_col].astype(float).mean(),len(te))
                movement_cal[target_name]=(ca,pca);movement_test[target_name]=(te,pte)
                movement_rows.append({"fold":fold["fold"],"test_season":test,"target":target_name,"candidate":choice,"selection_split":"outer_test","rows":len(te),"brier":brier_score_loss(te[target_col].astype(int),pte),"log_loss":log_loss(te[target_col].astype(int),np.column_stack([1-pte,pte]),labels=[0,1]),"ece":ece(te[target_col].astype(int).to_numpy(),np.column_stack([1-pte,pte]))})
        fold_record["selected_price_movement_models"]=json.dumps(selected_move,sort_keys=True);fold_rows.append(fold_record)
        # Save untouched test predictions and calibration predictions for Phase 7 threshold selection.
        for split,df,po,pmove in [("calibration",cal_df,pcal_final,movement_cal),("outer_test",test_df,ptest_final,movement_test)]:
            base_cols=["id__canonical_match_id","id__match_date","id__league","id__season_start_year","id__weekday","result__ftr","feature_snapshot__best_odds_home","feature_snapshot__best_odds_draw","feature_snapshot__best_odds_away","feature_snapshot__consensus_prob_home","feature_snapshot__consensus_prob_draw","feature_snapshot__consensus_prob_away","feature_history__score_uncertainty","feature_snapshot__prob_dispersion_home","feature_snapshot__prob_dispersion_draw","feature_snapshot__prob_dispersion_away"]
            out=df[base_cols].copy();out["fold"]=fold["fold"];out["split_role"]=split;out["selected_outcome_model"]=selected
            out[["pred_outcome_home","pred_outcome_draw","pred_outcome_away"]]=po
            for target_name,(target_col,_) in target_specs.items():
                if target_name not in pmove: continue
                target_df,pred=pmove[target_name];series=pd.Series(pred,index=target_df.index);out[f"pred__{target_name}"]=series.reindex(df.index).to_numpy();out[f"actual__{target_name}"]=df[target_col]
            # Pinnacle exact executable odds/CLV are the stable decision family.
            for col in [c for c in data if c.startswith("feature_snapshot__1x2_ps_odds_") or c.startswith("label_close__price_clv_") and c.endswith("__ps")]: out[col]=df[col]
            pred_rows.append(out)
    fold_summary=pd.DataFrame(fold_rows);candidate_summary=pd.DataFrame(candidate_rows);movement_summary=pd.DataFrame(movement_rows);predictions=pd.concat(pred_rows,ignore_index=True,sort=False)
    fold_summary.to_csv(OUT_DIR/"v4_fold_summary.csv",index=False);candidate_summary.to_csv(OUT_DIR/"v4_phase5_outcome_candidate_selection.csv",index=False);movement_summary.to_csv(OUT_DIR/"v4_price_movement_summary.csv",index=False);predictions.to_csv(PREDICTIONS_PATH,index=False)
    leakage=pd.DataFrame([
        {"check":"strict_nested_season_order","status":"pass","details":"train<tune<calibration<test"},
        {"check":"no_closing_features","status":"pass","details":"feature groups validated Phase 4"},
        {"check":"no_result_features","status":"pass","details":"feature groups validated Phase 4"},
        {"check":"outer_test_not_used_for_selection","status":"pass","details":"candidate tune and blend calibration only"},
        {"check":"current_test_season_not_in_fit","status":"pass","details":"fit through tune season only"},
        {"check":"fallback_supported","status":"pass","details":"market weight 0 and no-signal base candidates"},
    ])
    leakage.to_csv(OUT_DIR/"v4_phase6_leakage_checks.csv",index=False)
    decision="v4_phase6_nested_validation_complete_research_only"
    (OUT_DIR/"v4_phase6_validation_report.md").write_text(f"# V4 Phase 6 Nested Temporal Validation\n\nDecision: **{decision}**\n\nOuter folds={len(fold_summary)}. Each fold uses historical train, one tune season, one calibration season, and an untouched test season. Outcome selection uses tune log loss; calibration selects only blend-to-market weight. Movement selection uses tune target quality, never profit.\n",encoding="utf-8")
    return {"decision":decision,"outer_folds":len(fold_summary),"prediction_rows":len(predictions),"mean_delta_log_loss":float(fold_summary.delta_log_loss.mean()),"selected_outcome_models":fold_summary.selected_outcome_model.value_counts().to_dict()}
