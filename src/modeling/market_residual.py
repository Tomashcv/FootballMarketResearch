from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.probability import market_probs, normalize_probs, probs_to_logits

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    feature_group: str
    params: dict[str, Any]
    recency_half_life_years: float | None = None
    league_balance_strength: float = 0.0


@dataclass
class FitMetadata:
    candidate_name: str
    family: str
    feature_count: int
    train_rows: int
    validation_rows: int
    best_rounds: int | None
    used_native_missing: bool
    recency_half_life_years: float | None
    league_balance_strength: float


class FittedCandidate:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:  # pragma: no cover - protocol method
        raise NotImplementedError


class RawMarketCandidate(FittedCandidate):
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return market_probs(frame)


class XGBResidualCandidate(FittedCandidate):
    def __init__(
        self,
        booster,
        feature_cols: list[str],
        imputer: SimpleImputer | None,
        best_rounds: int,
    ) -> None:
        self.booster = booster
        self.feature_cols = feature_cols
        self.imputer = imputer
        self.best_rounds = int(best_rounds)

    def _matrix(self, frame: pd.DataFrame):
        values = frame[self.feature_cols].to_numpy(dtype=np.float32)
        if self.imputer is not None:
            values = self.imputer.transform(values)
        base_margin = probs_to_logits(market_probs(frame)).reshape(-1)
        return xgb.DMatrix(values, base_margin=base_margin, missing=np.nan)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._matrix(frame)
        try:
            pred = self.booster.predict(matrix, iteration_range=(0, self.best_rounds))
        except TypeError:  # xgboost < 1.6 compatibility
            pred = self.booster.predict(matrix, ntree_limit=self.best_rounds)
        return normalize_probs(pred)


class LogisticMarketCandidate(FittedCandidate):
    def __init__(self, pipeline: Pipeline, feature_cols: list[str]) -> None:
        self.pipeline = pipeline
        self.feature_cols = feature_cols

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame[self.feature_cols].to_numpy(dtype=float)
        logits = probs_to_logits(market_probs(frame))
        return np.column_stack([numeric, logits])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return normalize_probs(self.pipeline.predict_proba(self._matrix(frame)))


def available_feature_columns(train: pd.DataFrame, requested: list[str]) -> list[str]:
    result: list[str] = []
    for col in requested:
        if col not in train.columns:
            continue
        numeric = pd.to_numeric(train[col], errors="coerce")
        if numeric.notna().any() and numeric.nunique(dropna=True) > 1:
            result.append(col)
    return result


def build_training_weights(
    train: pd.DataFrame,
    recency_half_life_years: float | None,
    league_balance_strength: float,
) -> np.ndarray:
    weights = np.ones(len(train), dtype=float)
    if recency_half_life_years is not None and recency_half_life_years > 0:
        years = pd.to_numeric(train["season_start_year"], errors="coerce").to_numpy(dtype=float)
        max_year = float(np.nanmax(years))
        ages = np.maximum(0.0, max_year - years)
        weights *= np.power(0.5, ages / float(recency_half_life_years))
    if league_balance_strength > 0 and "league" in train.columns:
        league_values = train["league"].astype(str)
        counts = league_values.value_counts()
        league_counts = league_values.map(counts).to_numpy(dtype=float, copy=True)
        inv_sqrt = 1.0 / np.sqrt(league_counts)
        inv_sqrt /= float(np.mean(inv_sqrt))
        strength = float(np.clip(league_balance_strength, 0.0, 1.0))
        weights *= (1.0 - strength) + strength * inv_sqrt
    weights /= float(np.mean(weights))
    return weights


def _xgb_params(spec: CandidateSpec) -> tuple[dict[str, Any], int, int | None, bool]:
    p = dict(spec.params)
    rounds = int(p.pop("num_boost_round", 100))
    early = p.pop("early_stopping_rounds", 20)
    early = int(early) if early is not None else None
    native_missing = bool(p.pop("native_missing", True))
    defaults = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": 17,
        "nthread": 2,
    }
    defaults.update(p)
    return defaults, rounds, early, native_missing


def fit_candidate(
    spec: CandidateSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    requested_features: list[str],
    fixed_rounds: int | None = None,
) -> tuple[FittedCandidate, np.ndarray | None, FitMetadata]:
    if spec.family == "raw_market":
        model = RawMarketCandidate()
        val_prob = model.predict_proba(validation) if validation is not None and not validation.empty else None
        return model, val_prob, FitMetadata(spec.name, spec.family, 0, len(train), len(validation) if validation is not None else 0, None, True, spec.recency_half_life_years, spec.league_balance_strength)

    feature_cols = available_feature_columns(train, requested_features)
    if not feature_cols:
        model = RawMarketCandidate()
        val_prob = model.predict_proba(validation) if validation is not None and not validation.empty else None
        return model, val_prob, FitMetadata(spec.name, "raw_market_fallback", 0, len(train), len(validation) if validation is not None else 0, None, True, spec.recency_half_life_years, spec.league_balance_strength)

    y_train = train["target_y"].to_numpy(dtype=int)
    weights = build_training_weights(train, spec.recency_half_life_years, spec.league_balance_strength)

    if spec.family == "xgb_market_residual":
        if xgb is None:
            raise RuntimeError("xgboost is required for xgb_market_residual")
        params, configured_rounds, early_stopping, native_missing = _xgb_params(spec)
        rounds = int(fixed_rounds or configured_rounds)
        train_values = train[feature_cols].to_numpy(dtype=np.float32)
        val_values = validation[feature_cols].to_numpy(dtype=np.float32) if validation is not None and not validation.empty else None
        imputer: SimpleImputer | None = None
        if not native_missing:
            imputer = SimpleImputer(strategy="median")
            train_values = imputer.fit_transform(train_values)
            if val_values is not None:
                val_values = imputer.transform(val_values)
        dtrain = xgb.DMatrix(
            train_values,
            label=y_train,
            weight=weights,
            base_margin=probs_to_logits(market_probs(train)).reshape(-1),
            missing=np.nan,
        )
        evals = []
        dval = None
        if validation is not None and not validation.empty:
            dval = xgb.DMatrix(
                val_values,
                label=validation["target_y"].to_numpy(dtype=int),
                base_margin=probs_to_logits(market_probs(validation)).reshape(-1),
                missing=np.nan,
            )
            evals = [(dval, "validation")]
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=evals,
            early_stopping_rounds=early_stopping if evals and fixed_rounds is None else None,
            verbose_eval=False,
        )
        best_rounds = rounds
        if fixed_rounds is None and getattr(booster, "best_iteration", None) is not None:
            best_rounds = int(booster.best_iteration) + 1
        model = XGBResidualCandidate(booster, feature_cols, imputer, best_rounds)
        val_prob = model.predict_proba(validation) if validation is not None and not validation.empty else None
        metadata = FitMetadata(
            candidate_name=spec.name,
            family=spec.family,
            feature_count=len(feature_cols),
            train_rows=len(train),
            validation_rows=len(validation) if validation is not None else 0,
            best_rounds=best_rounds,
            used_native_missing=native_missing,
            recency_half_life_years=spec.recency_half_life_years,
            league_balance_strength=spec.league_balance_strength,
        )
        return model, val_prob, metadata

    if spec.family == "logistic_market_plus_features":
        c = float(spec.params.get("C", 0.05))
        max_iter = int(spec.params.get("max_iter", 1500))
        numeric = train[feature_cols].to_numpy(dtype=float)
        logits = probs_to_logits(market_probs(train))
        matrix = np.column_stack([numeric, logits])
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        l1_ratio=0.0,
                        solver="lbfgs",
                        max_iter=max_iter,
                        random_state=17,
                    ),
                ),
            ]
        )
        pipeline.fit(matrix, y_train, model__sample_weight=weights)
        model = LogisticMarketCandidate(pipeline, feature_cols)
        val_prob = model.predict_proba(validation) if validation is not None and not validation.empty else None
        return model, val_prob, FitMetadata(spec.name, spec.family, len(feature_cols), len(train), len(validation) if validation is not None else 0, None, False, spec.recency_half_life_years, spec.league_balance_strength)

    raise ValueError(f"Unknown candidate family: {spec.family}")
