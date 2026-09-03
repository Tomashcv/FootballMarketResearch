from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import warnings

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

from src.common.metrics import expected_calibration_error
from src.common.paths import get_league_matches_path
from src.experiments import i1_away_ah_contextual_memory_review as contextual_review
from src.features.contextual_features import assert_no_closing_columns
from src.features.contextual_features import build_contextual_features
from src.features.travel_features import build_travel_features
from src.features.weather_features import add_weather_features
from src.features.weather_features import add_weather_shock_features
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import asian_profit
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


LEAGUE = "E0"
REPORT_PATH = Path("outputs/reports/e0_away_ah_advanced_tabular_neural_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_advanced_tabular_neural_summary.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/advanced_tabular_neural_review")
REPORT_TITLE = "E0 Away AH Advanced Tabular Neural Review"
REPORT_CONTEXT = (
    "This run compares the existing E0 rule candidates with logistic regression, available XGBoost, "
    "a small NumPy dropout MLP fallback, and a true small PyTorch FT-Transformer-style tabular model when Torch is installed."
)

MIN_SEASON_START = 2020
MAX_SEASON_START = 2024
MIN_VALIDATION_BETS = 12
SEEDS = [11, 23, 37, 41, 53]
SCORE_QUANTILES = [0.50, 0.60, 0.70, 0.80]

NUMERIC_FEATURE_COLUMNS = [
    "ah_line",
    "home_ah_odds",
    "away_ah_odds",
    "home_market_probability",
    "away_market_probability",
    "overround",
    "avg_1x2_AvgH_no_vig_probability",
    "avg_1x2_AvgD_no_vig_probability",
    "avg_1x2_AvgA_no_vig_probability",
    "travel_distance_km",
    "away_rest_days",
    "home_rest_days",
    "rest_days_diff",
    "away_matches_last_7d",
    "away_matches_last_14d",
    "matches_last_14d_diff",
    "min_team_season_matches_before",
    "season_match_count_diff",
    "weather_temperature_c",
    "weather_precipitation_mm",
    "weather_wind_speed_kph",
    "away_temperature_shock_c",
    "away_precipitation_shock_mm",
    "away_wind_speed_shock_kph",
    "home_internal_elo_pre",
    "away_internal_elo_pre",
    "internal_elo_diff_home_minus_away",
    "internal_elo_home_win_prob",
    "market_home_prob_minus_internal_elo_prob",
    "market_away_prob_minus_internal_elo_prob",
    contextual_review.MEMORY_COLUMN,
]

CATEGORICAL_FEATURE_COLUMNS = ["HomeTeam", "AwayTeam"]
TARGET_COLUMN = "away_ah_cover"


@dataclass(frozen=True)
class TemporalSplit:
    test_year: int
    train_years: tuple[int, ...]
    validation_year: int


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_temporal_splits(seasons: list[int], min_train_seasons: int = 1) -> list[TemporalSplit]:
    ordered = sorted(int(season) for season in seasons)
    splits = []
    for index in range(min_train_seasons + 1, len(ordered)):
        test_year = ordered[index]
        validation_year = ordered[index - 1]
        train_years = tuple(ordered[: index - 1])
        splits.append(TemporalSplit(test_year=test_year, train_years=train_years, validation_year=validation_year))
    return splits


def load_weather_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    weather = pd.read_csv("data/external/weather/historical_match_weather.csv")
    normals = pd.read_csv("data/external/weather/monthly_climate_normals.csv")
    return weather, normals


def prepare_e0_data() -> pd.DataFrame:
    matches = pd.read_csv(get_league_matches_path(LEAGUE), low_memory=False)
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce").dt.normalize()
    matches["season_start_year"] = pd.to_numeric(matches["season_start_year"], errors="coerce")
    matches = matches[
        (matches["season_start_year"] >= MIN_SEASON_START) & (matches["season_start_year"] <= MAX_SEASON_START)
    ].copy()

    dataframe = build_contextual_features(matches)
    coordinates = pd.read_csv("data/external/stadiums/stadiums_with_gps_coordinates.csv")
    overrides = pd.read_csv("data/manual/team_stadium_overrides.csv")
    dataframe = build_travel_features(dataframe, coordinates, overrides)
    weather, normals = load_weather_tables()
    dataframe = add_weather_features(dataframe, weather)
    dataframe = add_weather_shock_features(dataframe, normals)

    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "season_end_year", "AHh", "AvgAHH", "AvgAHA"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dataframe["ah_line"] = pd.to_numeric(dataframe["AHh"], errors="coerce")
    dataframe["home_ah_odds"] = pd.to_numeric(dataframe["AvgAHH"], errors="coerce")
    dataframe["away_ah_odds"] = pd.to_numeric(dataframe["AvgAHA"], errors="coerce")
    dataframe["season_end_year"] = pd.to_numeric(dataframe["season_end_year"], errors="coerce")
    dataframe = dataframe.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "season_end_year",
            "ah_line",
            "home_ah_odds",
            "away_ah_odds",
        ]
    ).copy()
    dataframe = dataframe[(dataframe["home_ah_odds"] > 1.0) & (dataframe["away_ah_odds"] > 1.0)].copy()
    dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)
    dataframe["away_margin"] = dataframe["FTAG"].astype(float) - dataframe["FTHG"].astype(float)
    dataframe["away_handicap"] = -dataframe["ah_line"]
    dataframe["profit"] = dataframe.apply(
        lambda row: asian_profit(row["away_margin"], row["away_handicap"], row["away_ah_odds"]),
        axis=1,
    )
    dataframe[TARGET_COLUMN] = (dataframe["profit"] > 0.0).astype(int)

    dataframe["home_raw_implied"] = 1.0 / dataframe["home_ah_odds"]
    dataframe["away_raw_implied"] = 1.0 / dataframe["away_ah_odds"]
    dataframe["overround"] = dataframe["home_raw_implied"] + dataframe["away_raw_implied"]
    dataframe["home_market_probability"] = dataframe["home_raw_implied"] / dataframe["overround"]
    dataframe["away_market_probability"] = dataframe["away_raw_implied"] / dataframe["overround"]

    close_home = 1.0 / pd.to_numeric(dataframe.get("AvgCAHH"), errors="coerce")
    close_away = 1.0 / pd.to_numeric(dataframe.get("AvgCAHA"), errors="coerce")
    dataframe["clv_probability_pp"] = ((close_away / (close_home + close_away)) - dataframe["away_market_probability"]) * 100.0
    dataframe["line_move_to_away"] = dataframe["ah_line"] - pd.to_numeric(dataframe.get("AHCh"), errors="coerce")

    dataframe = contextual_review.add_knn_profit_memory(dataframe)
    return dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def available_feature_columns(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in NUMERIC_FEATURE_COLUMNS if column in dataframe.columns]
    categorical = [column for column in CATEGORICAL_FEATURE_COLUMNS if column in dataframe.columns]
    assert_no_closing_columns(numeric + categorical)
    return numeric, categorical


def build_preprocessor(dataframe: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric, categorical = available_feature_columns(dataframe)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric, categorical


def fit_preprocessor(train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    preprocessor, numeric, categorical = build_preprocessor(train)
    frame = train[numeric + categorical].copy()
    preprocessor.fit(frame)
    return preprocessor, numeric, categorical


def transform(preprocessor: ColumnTransformer, dataframe: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    return preprocessor.transform(dataframe[numeric + categorical].copy())


def safe_probability_fit_predict(model, train_x, train_y, val_x, test_x) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(train_y)) < 2:
        base = float(np.mean(train_y)) if len(train_y) else 0.5
        return np.full(len(val_x), base), np.full(len(test_x), base)
    try:
        model.fit(train_x, train_y)
    except ValueError as error:
        if "least populated classes" not in str(error):
            raise
        base = float(np.mean(train_y)) if len(train_y) else 0.5
        return np.full(len(val_x), base), np.full(len(test_x), base)
    return model.predict_proba(val_x)[:, 1], model.predict_proba(test_x)[:, 1]


def safe_probability_fit_predict_with_validation(model, train_x, train_y, val_x, val_y, test_x) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(train_y)) < 2:
        base = float(np.mean(train_y)) if len(train_y) else 0.5
        return np.full(len(val_x), base), np.full(len(test_x), base)
    model.fit(train_x, train_y, validation_x=val_x, validation_y=val_y)
    return model.predict_proba(val_x)[:, 1], model.predict_proba(test_x)[:, 1]


def logistic_model(seed: int):
    return LogisticRegression(max_iter=1000, C=0.5, penalty="l2", solver="lbfgs", random_state=seed)


def xgboost_model(seed: int):
    if XGBClassifier is None:
        return None
    return XGBClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_lambda=4.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


class NumpyDropoutMLPClassifier:
    def __init__(
        self,
        seed: int,
        hidden_sizes: tuple[int, int] = (32, 16),
        dropout: float = 0.15,
        weight_decay: float = 0.001,
        learning_rate: float = 0.003,
        max_epochs: int = 120,
        patience: int = 12,
        batch_size: int = 64,
    ):
        self.seed = int(seed)
        self.hidden_sizes = hidden_sizes
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.best_epoch_ = 0

    def _initialize(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        layer_sizes = [n_features, *self.hidden_sizes, 1]
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = math.sqrt(2.0 / max(1, fan_in))
            self.weights_.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))

    def _forward(self, x: np.ndarray, training: bool, rng: np.random.Generator | None = None):
        activations = [x]
        pre_activations = []
        dropout_masks = []
        current = x
        for layer_index, (weights, bias) in enumerate(zip(self.weights_, self.biases_)):
            z = current @ weights + bias
            pre_activations.append(z)
            is_output = layer_index == len(self.weights_) - 1
            if is_output:
                current = self._sigmoid(z)
                dropout_masks.append(None)
            else:
                current = np.maximum(z, 0.0)
                if training and self.dropout > 0.0 and rng is not None:
                    mask = (rng.random(current.shape) >= self.dropout).astype(float) / (1.0 - self.dropout)
                    current = current * mask
                    dropout_masks.append(mask)
                else:
                    dropout_masks.append(None)
            activations.append(current)
        return activations, pre_activations, dropout_masks

    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probabilities = np.clip(self.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)
        return float(log_loss(y, probabilities, labels=[0, 1]))

    def fit(self, x: np.ndarray, y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        validation_y = np.asarray(validation_y, dtype=int)
        self._initialize(x.shape[1])
        rng = np.random.default_rng(self.seed)
        best_loss = math.inf
        best_weights = [weights.copy() for weights in self.weights_]
        best_biases = [bias.copy() for bias in self.biases_]
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                activations, pre_activations, dropout_masks = self._forward(batch_x, training=True, rng=rng)
                delta = (activations[-1] - batch_y) / max(1, len(batch_x))
                grad_weights = []
                grad_biases = []
                for layer_index in reversed(range(len(self.weights_))):
                    grad_w = activations[layer_index].T @ delta + self.weight_decay * self.weights_[layer_index]
                    grad_b = delta.sum(axis=0)
                    grad_weights.insert(0, grad_w)
                    grad_biases.insert(0, grad_b)
                    if layer_index > 0:
                        delta = delta @ self.weights_[layer_index].T
                        mask = dropout_masks[layer_index - 1]
                        if mask is not None:
                            delta = delta * mask
                        delta = delta * (pre_activations[layer_index - 1] > 0.0)
                for layer_index in range(len(self.weights_)):
                    self.weights_[layer_index] -= self.learning_rate * grad_weights[layer_index]
                    self.biases_[layer_index] -= self.learning_rate * grad_biases[layer_index]

            validation_loss = self._loss(validation_x, validation_y)
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_weights = [weights.copy() for weights in self.weights_]
                best_biases = [bias.copy() for bias in self.biases_]
                self.best_epoch_ = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        self.weights_ = best_weights
        self.biases_ = best_biases
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        probabilities = self._forward(x, training=False)[0][-1].reshape(-1)
        return np.column_stack([1.0 - probabilities, probabilities])


def neural_model(seed: int):
    return NumpyDropoutMLPClassifier(seed=seed)


if nn is not None:

    class FTTransformerNetwork(nn.Module):
        def __init__(self, n_features: int, d_token: int = 16, n_heads: int = 2, dropout: float = 0.15):
            super().__init__()
            self.feature_weight = nn.Parameter(torch.empty(n_features, d_token))
            self.feature_bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=32,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Dropout(dropout),
                nn.Linear(d_token, 1),
            )
            nn.init.normal_(self.feature_weight, mean=0.0, std=0.02)

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)


class TorchFTTransformerClassifier:
    def __init__(
        self,
        seed: int,
        d_token: int = 16,
        dropout: float = 0.15,
        weight_decay: float = 0.01,
        learning_rate: float = 0.001,
        max_epochs: int = 80,
        patience: int = 10,
        batch_size: int = 64,
    ):
        if torch is None or nn is None:
            raise RuntimeError("Torch is not available.")
        self.seed = int(seed)
        self.d_token = int(d_token)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.model_: FTTransformerNetwork | None = None
        self.best_epoch_ = 0
        self.device_ = torch.device("cpu")

    def fit(self, x: np.ndarray, y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray):
        set_random_seeds(self.seed)
        train_x = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=self.device_)
        train_y = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32, device=self.device_)
        val_x = torch.tensor(np.asarray(validation_x, dtype=np.float32), dtype=torch.float32, device=self.device_)
        val_y = torch.tensor(np.asarray(validation_y, dtype=np.float32), dtype=torch.float32, device=self.device_)

        self.model_ = FTTransformerNetwork(train_x.shape[1], d_token=self.d_token, dropout=self.dropout).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        best_loss = math.inf
        best_state = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
        stale_epochs = 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        for epoch in range(self.max_epochs):
            self.model_.train()
            order = torch.randperm(train_x.shape[0], generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                logits = self.model_(train_x[batch_indices])
                loss = loss_fn(logits, train_y[batch_indices])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=2.0)
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(val_x), val_y).detach().cpu().item())
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
                self.best_epoch_ = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        self.model_.eval()
        features = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=self.device_)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model_(features)).detach().cpu().numpy()
        return np.column_stack([1.0 - probabilities, probabilities])


def torch_transformer_model(seed: int):
    return TorchFTTransformerClassifier(seed=seed)


def candidate_score(dataframe: pd.DataFrame, probabilities: np.ndarray, target_style: str) -> pd.Series:
    probability = pd.Series(probabilities, index=dataframe.index)
    if target_style == "market_residual":
        return probability - pd.to_numeric(dataframe["away_market_probability"], errors="coerce")
    return probability


def candidate_thresholds(scores: pd.Series) -> list[float]:
    clean = pd.to_numeric(scores, errors="coerce").dropna()
    if len(clean) == 0:
        return []
    return sorted(set(float(clean.quantile(q)) for q in SCORE_QUANTILES))


def select_validation_candidate(validation: pd.DataFrame, probabilities: np.ndarray, target_style: str) -> dict | None:
    scores = candidate_score(validation, probabilities, target_style)
    candidates = []
    for ah_threshold in THRESHOLDS:
        for score_threshold in candidate_thresholds(scores):
            selected = validation[(pd.to_numeric(validation["ah_line"], errors="coerce") <= ah_threshold) & (scores >= score_threshold)].copy()
            if len(selected) < MIN_VALIDATION_BETS:
                continue
            summary = summarize(selected)
            if summary["profit"] <= 0.0 or summary["roi"] <= 0.0:
                continue
            candidates.append(
                {
                    "selected_threshold": ah_threshold,
                    "selected_score_threshold": score_threshold,
                    "validation_bets": summary["bets"],
                    "validation_profit": summary["profit"],
                    "validation_roi": summary["roi"],
                    "validation_z_score": summary["z_score"],
                    "validation_max_drawdown": summary["max_drawdown"],
                }
            )
    if not candidates:
        return None
    return (
        pd.DataFrame(candidates)
        .sort_values(["validation_z_score", "validation_roi", "validation_bets"], ascending=[False, False, False])
        .iloc[0]
        .to_dict()
    )


def probability_metrics(dataframe: pd.DataFrame, probabilities: np.ndarray, model_name: str, test_year: int) -> dict:
    y = dataframe[TARGET_COLUMN].astype(int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    market = np.clip(pd.to_numeric(dataframe["away_market_probability"], errors="coerce").fillna(0.5).to_numpy(), 1e-6, 1.0 - 1e-6)
    return {
        "model": model_name,
        "test_year": int(test_year),
        "rows": int(len(dataframe)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
        "market_log_loss": float(log_loss(y, market, labels=[0, 1])),
        "market_brier": float(brier_score_loss(y, market)),
        "market_ece": float(expected_calibration_error(y, market)),
    }


def clv_summary(dataframe: pd.DataFrame) -> dict:
    if len(dataframe) == 0:
        return {"avg_clv_pp": pd.NA, "clv_positive_rate": pd.NA}
    clv = pd.to_numeric(dataframe["clv_probability_pp"], errors="coerce")
    return {
        "avg_clv_pp": float(clv.mean()) if clv.notna().any() else pd.NA,
        "clv_positive_rate": float((clv > 0.0).mean()) if clv.notna().any() else pd.NA,
    }


def concentration(dataframe: pd.DataFrame) -> dict:
    if len(dataframe) == 0:
        return {"top3_home_bet_share": pd.NA, "top3_away_bet_share": pd.NA, "home_hhi_bets": pd.NA, "away_hhi_bets": pd.NA}

    def top3(column: str) -> float:
        return float(dataframe[column].value_counts(normalize=True).head(3).sum())

    def hhi(column: str) -> float:
        shares = dataframe[column].value_counts(normalize=True)
        return float((shares * shares).sum())

    return {
        "top3_home_bet_share": top3("HomeTeam"),
        "top3_away_bet_share": top3("AwayTeam"),
        "home_hhi_bets": hhi("HomeTeam"),
        "away_hhi_bets": hhi("AwayTeam"),
    }


def overall_row(strategy: str, dataframe: pd.DataFrame, model_family: str, target_style: str) -> dict:
    summary = summarize(dataframe)
    row = {
        "strategy": strategy,
        "model_family": model_family,
        "target_style": target_style,
        "bets": summary["bets"],
        "profit": summary["profit"],
        "roi": summary["roi"],
        "z_score": summary["z_score"],
        "max_drawdown": summary["max_drawdown"],
    }
    row.update(clv_summary(dataframe))
    row.update(concentration(dataframe))
    return row


def run_model_nested(dataframe: pd.DataFrame, model_name: str, model_family: str, target_style: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_random_seeds(seed)
    splits = make_temporal_splits(sorted(dataframe["season_end_year"].unique()))
    by_year_rows = []
    bet_frames = []
    metric_rows = []

    for split in splits:
        train = dataframe[dataframe["season_end_year"].isin(split.train_years)].copy()
        validation = dataframe[dataframe["season_end_year"] == split.validation_year].copy()
        test = dataframe[dataframe["season_end_year"] == split.test_year].copy()
        if len(train) == 0 or len(validation) == 0 or len(test) == 0:
            continue

        preprocessor, numeric, categorical = fit_preprocessor(train)
        train_x = transform(preprocessor, train, numeric, categorical)
        validation_x = transform(preprocessor, validation, numeric, categorical)
        test_x = transform(preprocessor, test, numeric, categorical)
        train_y = train[TARGET_COLUMN].astype(int).to_numpy()
        validation_y = validation[TARGET_COLUMN].astype(int).to_numpy()

        if model_family == "logistic":
            model = logistic_model(seed)
        elif model_family == "xgboost":
            model = xgboost_model(seed)
            if model is None:
                continue
        elif model_family == "neural_mlp":
            model = neural_model(seed)
        elif model_family == "torch_ft_transformer":
            if torch is None:
                continue
            model = torch_transformer_model(seed)
        else:
            raise ValueError(f"Unknown model family: {model_family}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if model_family in {"neural_mlp", "torch_ft_transformer"}:
                validation_probability, test_probability = safe_probability_fit_predict_with_validation(
                    model,
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    test_x,
                )
            else:
                validation_probability, test_probability = safe_probability_fit_predict(model, train_x, train_y, validation_x, test_x)

        metric_rows.append(probability_metrics(test, test_probability, model_name, split.test_year))
        selected = select_validation_candidate(validation, validation_probability, target_style)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": model_name,
                    "test_year": split.test_year,
                    "train_years": ";".join(str(year) for year in split.train_years),
                    "validation_year": split.validation_year,
                    "selected_threshold": pd.NA,
                    "selected_score_threshold": pd.NA,
                    "selected_filter": "no_valid_validation_candidate",
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "test_z_score": 0.0,
                    "test_max_drawdown": 0.0,
                }
            )
            continue

        test_scores = candidate_score(test, test_probability, target_style)
        selected_test = test[
            (pd.to_numeric(test["ah_line"], errors="coerce") <= float(selected["selected_threshold"]))
            & (test_scores >= float(selected["selected_score_threshold"]))
        ].copy()
        selected_test["model_probability"] = test_probability[selected_test.index.map(test.index.get_loc)]
        selected_test["model_score"] = test_scores.loc[selected_test.index].to_numpy()
        selected_test["strategy"] = model_name
        selected_test["model_family"] = model_family
        selected_test["target_style"] = target_style
        selected_test["seed"] = seed
        selected_test["nested_test_year"] = split.test_year
        selected_test["validation_year"] = split.validation_year
        selected_test["selected_threshold"] = selected["selected_threshold"]
        selected_test["selected_score_threshold"] = selected["selected_score_threshold"]
        summary = summarize(selected_test)
        by_year_rows.append(
            {
                "strategy": model_name,
                "test_year": split.test_year,
                "train_years": ";".join(str(year) for year in split.train_years),
                "validation_year": split.validation_year,
                "selected_threshold": selected["selected_threshold"],
                "selected_score_threshold": selected["selected_score_threshold"],
                "selected_filter": f"{target_style}_score>={selected['selected_score_threshold']:.6f}",
                "validation_bets": selected["validation_bets"],
                "validation_profit": selected["validation_profit"],
                "validation_roi": selected["validation_roi"],
                "validation_z_score": selected["validation_z_score"],
                "test_bets": summary["bets"],
                "test_profit": summary["profit"],
                "test_roi": summary["roi"],
                "test_z_score": summary["z_score"],
                "test_max_drawdown": summary["max_drawdown"],
            }
        )
        if len(selected_test):
            bet_frames.append(selected_test)

    return (
        pd.DataFrame(by_year_rows),
        pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def run_rule_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.read_csv("outputs/E0/asian_handicap_big_home_favorite_away/memory_odds_combo_falsification/overall.csv")
    overall = overall[overall["strategy"].isin(["away_odds_ge_1_85", "away_odds_ge_1_85_plus_memory_knn_profit"])].copy()
    overall["model_family"] = "rule_benchmark"
    overall["target_style"] = "rule"
    by_year = pd.read_csv("outputs/E0/asian_handicap_big_home_favorite_away/memory_odds_combo_review/nested_by_year.csv")
    by_year = by_year[by_year["strategy"].isin(overall["strategy"])].copy()
    return overall, by_year


def aggregate_seed_rows(overall: pd.DataFrame, model_family: str) -> pd.DataFrame:
    frame = overall[overall["model_family"].eq(model_family)].copy()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for target_style, group in frame.groupby("target_style"):
        rows.append(
            {
                "strategy": f"{model_family}_{target_style}_seed_mean",
                "model_family": f"{model_family}_seed_mean",
                "target_style": target_style,
                "bets": float(group["bets"].mean()),
                "profit": float(group["profit"].mean()),
                "roi": float(group["roi"].mean()),
                "z_score": float(group["z_score"].mean()),
                "max_drawdown": float(group["max_drawdown"].mean()),
                "avg_clv_pp": float(group["avg_clv_pp"].mean()),
                "clv_positive_rate": float(group["clv_positive_rate"].mean()),
                "top3_home_bet_share": float(group["top3_home_bet_share"].mean()),
                "top3_away_bet_share": float(group["top3_away_bet_share"].mean()),
                "home_hhi_bets": float(group["home_hhi_bets"].mean()),
                "away_hhi_bets": float(group["away_hhi_bets"].mean()),
                "seed_profit_std": float(group["profit"].std(ddof=0)),
                "seed_roi_std": float(group["roi"].std(ddof=0)),
                "seed_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def seasonal_rows(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame()
    for strategy, strategy_frame in bets.groupby("strategy"):
        for season, group in strategy_frame.groupby("season_end_year"):
            row = overall_row(strategy, group, group["model_family"].iloc[0], group["target_style"].iloc[0])
            row["season"] = int(season)
            rows.append(row)
    return pd.DataFrame(rows)


def exclude_rows(overall_bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if overall_bets.empty:
        return pd.DataFrame()
    for strategy, strategy_frame in overall_bets.groupby("strategy"):
        for season in sorted(strategy_frame["season_end_year"].unique()):
            row = overall_row(
                strategy,
                strategy_frame[strategy_frame["season_end_year"] != season],
                strategy_frame["model_family"].iloc[0],
                strategy_frame["target_style"].iloc[0],
            )
            row["excluded_season"] = int(season)
            row["exclusion_reason"] = "exclude_each_season"
            rows.append(row)
        row = overall_row(
            strategy,
            strategy_frame[strategy_frame["season_end_year"] != 2025],
            strategy_frame["model_family"].iloc[0],
            strategy_frame["target_style"].iloc[0],
        )
        row["excluded_season"] = 2025
        row["exclusion_reason"] = "exclude_2025"
        rows.append(row)
    return pd.DataFrame(rows)


def classify(overall: pd.DataFrame, probability_metrics_frame: pd.DataFrame) -> tuple[str, str]:
    model_rows = overall[~overall["model_family"].eq("rule_benchmark")].copy()
    if model_rows.empty:
        return "reject", "No advanced tabular model produced test bets."
    profitable = model_rows[(model_rows["profit"] > 0.0) & (model_rows["roi"] > 0.0)].copy()
    if profitable.empty:
        return "reject", "No advanced tabular model improved historical profit out of sample."
    best = profitable.sort_values(["z_score", "roi", "profit"], ascending=[False, False, False]).iloc[0]
    rule_combo = overall[overall["strategy"].eq("away_odds_ge_1_85_plus_memory_knn_profit")]
    rule_z = float(rule_combo["z_score"].iloc[0]) if len(rule_combo) else 2.0
    calibration_ok = False
    model_metrics = probability_metrics_frame[probability_metrics_frame["model"].eq(best["strategy"])]
    if len(model_metrics):
        calibration_ok = bool(model_metrics["brier"].mean() < model_metrics["market_brier"].mean())
    clv_ok = bool(best["avg_clv_pp"] > 0.0 and best["clv_positive_rate"] >= 0.50)
    concentration_ok = bool(best["top3_home_bet_share"] <= 0.58 and best["home_hhi_bets"] <= 0.15)
    robustness_ok = bool(best["z_score"] >= rule_z and best["bets"] >= 80)
    if clv_ok and calibration_ok and concentration_ok and robustness_ok and best["z_score"] >= 2.5:
        return "confirmed edge", "Advanced model clears strict CLV, robustness, calibration, and concentration gates."
    if clv_ok and calibration_ok and concentration_ok and best["z_score"] >= 1.5:
        return "paper challenger", f"{best['strategy']} is positive and calibrated enough to paper-track as a challenger, but not confirmed."
    return "research only", f"{best['strategy']} is historically positive, but CLV, robustness, calibration, and concentration do not all improve."


def markdown_table(dataframe: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if dataframe.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in dataframe.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                values.append("")
            elif column in {"roi", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share"}:
                values.append(f"{100.0 * float(value):.2f}%")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_outputs(
    dataframe: pd.DataFrame,
    overall: pd.DataFrame,
    by_year: pd.DataFrame,
    bets: pd.DataFrame,
    seasonal: pd.DataFrame,
    exclude: pd.DataFrame,
    probability_metrics_frame: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(SUMMARY_PATH, index=False)
    overall.to_csv(DETAIL_DIR / "overall.csv", index=False)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "nested_bets.csv", index=False)
    seasonal.to_csv(DETAIL_DIR / "seasonal.csv", index=False)
    exclude.to_csv(DETAIL_DIR / "exclude_each_season.csv", index=False)
    probability_metrics_frame.to_csv(DETAIL_DIR / "probability_metrics.csv", index=False)

    classification, rationale = classify(overall, probability_metrics_frame)
    feature_columns, categorical_columns = available_feature_columns(dataframe)
    neural_rows = overall[overall["model_family"].str.contains("neural_mlp", na=False)].copy()
    transformer_rows = overall[overall["model_family"].str.contains("torch_ft_transformer", na=False)].copy()
    torch_version = torch.__version__ if torch is not None else "not installed"

    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"Scope: controlled E0 / Premier League Away Asian Handicap big home favourite experiment. {REPORT_CONTEXT}",
        "",
        "No broad model search was run. Raw match data was not edited. External APIs were not used. Closing odds are used only after selection for CLV diagnostics.",
        "",
        f"Torch import: {torch_version}. Input rows after bet-time-safe cleanup: {len(dataframe)}. Numeric feature count: {len(feature_columns)}. Categorical columns: {', '.join(categorical_columns)}. Neural seeds: {', '.join(str(seed) for seed in SEEDS)}.",
        "",
        "## Overall Results",
        "",
        markdown_table(
            overall,
            [
                "strategy",
                "model_family",
                "target_style",
                "bets",
                "profit",
                "roi",
                "z_score",
                "max_drawdown",
                "avg_clv_pp",
                "clv_positive_rate",
                "top3_home_bet_share",
                "top3_away_bet_share",
                "home_hhi_bets",
                "away_hhi_bets",
            ],
            ["Strategy", "Family", "Target", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Top3 away", "Home HHI", "Away HHI"],
        ),
        "",
        "## Neural Seed Variance",
        "",
        markdown_table(
            neural_rows,
            ["strategy", "target_style", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate"],
            ["Strategy", "Target", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+ rate"],
        ),
        "",
        "## Torch Transformer Seed Variance",
        "",
        markdown_table(
            transformer_rows,
            ["strategy", "target_style", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate"],
            ["Strategy", "Target", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+ rate"],
        ),
        "",
        "## Probability Calibration",
        "",
        markdown_table(
            probability_metrics_frame,
            ["model", "test_year", "rows", "log_loss", "market_log_loss", "brier", "market_brier", "ece", "market_ece"],
            ["Model", "Test year", "Rows", "Log loss", "Market log loss", "Brier", "Market Brier", "ECE", "Market ECE"],
        ),
        "",
        "## Season By Season",
        "",
        markdown_table(
            seasonal,
            ["strategy", "season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"],
            ["Strategy", "Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate"],
        ),
        "",
        "## Exclude Each Season",
        "",
        markdown_table(
            exclude,
            ["strategy", "exclusion_reason", "excluded_season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp"],
            ["Strategy", "Reason", "Excluded season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp"],
        ),
        "",
        "## Nested Split Controls",
        "",
        markdown_table(
            by_year,
            ["strategy", "test_year", "train_years", "validation_year", "selected_threshold", "selected_filter", "validation_roi", "test_bets", "test_profit", "test_roi"],
            ["Strategy", "Test", "Train years", "Validation", "AH threshold", "Selected filter", "Val ROI", "Test bets", "Test profit", "Test ROI"],
        ),
        "",
        "## Methodology",
        "",
        "- Splits are nested temporal: train seasons precede the validation season, and the validation season precedes the held-out test season.",
        "- Scalers and team encoders are fit only on the train seasons for each split. Unknown future teams are handled without looking ahead.",
        "- Feature columns explicitly reject closing odds markers before preprocessing.",
        "- The binary target is away AH positive cover. The market-residual target style selects by model probability minus no-vig away AH market probability.",
        "- XGBoost is included because it is installed locally; LightGBM is not installed and was not added.",
        "- The NumPy fallback neural model is a small tabular MLP with dropout, weight decay, capped epochs, validation-season early stopping, fixed seeds, and five-seed variance reporting.",
        "- The true tabular transformer is a small PyTorch FT-Transformer-style model with one transformer block, 16-dimensional feature tokens, two attention heads, dropout, AdamW weight decay, capped epochs, validation-season early stopping, and five fixed seeds.",
        "",
        "## Final Classification",
        "",
        f"**{classification}**",
        "",
        f"Rationale: {rationale}",
        "",
        "Do not call this a confirmed edge.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dataframe = prepare_e0_data()
    rule_overall, rule_by_year = run_rule_benchmarks()

    overall_rows = []
    by_year_frames = [rule_by_year]
    bet_frames = []
    metric_frames = []

    model_specs = [
        ("logistic_binary_cover", "logistic", "binary_cover", 101),
        ("logistic_market_residual", "logistic", "market_residual", 101),
    ]
    if XGBClassifier is not None:
        model_specs += [
            ("xgboost_binary_cover", "xgboost", "binary_cover", 101),
            ("xgboost_market_residual", "xgboost", "market_residual", 101),
        ]
    for seed in SEEDS:
        model_specs += [
            (f"neural_mlp_binary_cover_seed_{seed}", "neural_mlp", "binary_cover", seed),
            (f"neural_mlp_market_residual_seed_{seed}", "neural_mlp", "market_residual", seed),
        ]
        if torch is not None:
            model_specs += [
                (f"torch_ft_transformer_binary_cover_seed_{seed}", "torch_ft_transformer", "binary_cover", seed),
                (f"torch_ft_transformer_market_residual_seed_{seed}", "torch_ft_transformer", "market_residual", seed),
            ]

    for model_name, family, target_style, seed in model_specs:
        by_year, bets, metrics = run_model_nested(dataframe, model_name, family, target_style, seed)
        by_year_frames.append(by_year)
        if len(bets):
            bet_frames.append(bets)
        if len(metrics):
            metric_frames.append(metrics)
        overall_rows.append(overall_row(model_name, bets, family, target_style))

    model_overall = pd.DataFrame(overall_rows)
    neural_seed_mean = aggregate_seed_rows(model_overall, "neural_mlp")
    transformer_seed_mean = aggregate_seed_rows(model_overall, "torch_ft_transformer")
    overall = pd.concat([rule_overall, model_overall, neural_seed_mean, transformer_seed_mean], ignore_index=True, sort=False)
    by_year = pd.concat(by_year_frames, ignore_index=True, sort=False)
    bets = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    probability_metrics_frame = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    seasonal = seasonal_rows(bets)
    exclude = exclude_rows(bets)
    write_outputs(dataframe, overall, by_year, bets, seasonal, exclude, probability_metrics_frame)
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
