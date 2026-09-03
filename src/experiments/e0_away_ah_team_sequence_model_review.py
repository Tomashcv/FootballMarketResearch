from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from src.common.metrics import expected_calibration_error
from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/e0_away_ah_team_sequence_model_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_team_sequence_model_summary.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/team_sequence_model_review")

SEEDS = [11, 23, 37]
SEQUENCE_LENGTHS = [5, 10, 20]
MODEL_TYPES = ["gru", "tcn", "sequence_transformer"]
SCORE_QUANTILES = [0.50, 0.60, 0.70, 0.80]
MIN_VALIDATION_BETS = 12
TARGET_COLUMN = advanced.TARGET_COLUMN

SEQUENCE_FEATURE_COLUMNS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "result_points",
    "is_home",
    "team_rest_days",
    "opponent_internal_elo_pre",
    "team_internal_elo_pre",
    "team_ah_line",
    "team_ah_odds",
    "opponent_ah_odds",
    "away_market_probability",
    "travel_distance_km",
    "weather_temperature_c",
    "weather_precipitation_mm",
    "weather_wind_speed_kph",
]


@dataclass(frozen=True)
class SequenceBundle:
    current_x: np.ndarray
    home_sequence: np.ndarray
    away_sequence: np.ndarray
    y: np.ndarray
    dataframe: pd.DataFrame


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _team_perspective_row(match: pd.Series, team: str) -> dict:
    is_home = str(match["HomeTeam"]) == str(team)
    goals_for = float(match["FTHG"] if is_home else match["FTAG"])
    goals_against = float(match["FTAG"] if is_home else match["FTHG"])
    result_points = 1.0 if goals_for > goals_against else 0.5 if goals_for == goals_against else 0.0
    return {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
        "result_points": result_points,
        "is_home": 1.0 if is_home else 0.0,
        "team_rest_days": float(match["home_rest_days"] if is_home else match["away_rest_days"]),
        "opponent_internal_elo_pre": float(
            match["away_internal_elo_pre"] if is_home else match["home_internal_elo_pre"]
        ),
        "team_internal_elo_pre": float(match["home_internal_elo_pre"] if is_home else match["away_internal_elo_pre"]),
        "team_ah_line": float(match["ah_line"] if is_home else -match["ah_line"]),
        "team_ah_odds": float(match["home_ah_odds"] if is_home else match["away_ah_odds"]),
        "opponent_ah_odds": float(match["away_ah_odds"] if is_home else match["home_ah_odds"]),
        "away_market_probability": float(match["away_market_probability"]),
        "travel_distance_km": 0.0 if is_home else float(match.get("travel_distance_km", 0.0)),
        "weather_temperature_c": float(match.get("weather_temperature_c", np.nan)),
        "weather_precipitation_mm": float(match.get("weather_precipitation_mm", np.nan)),
        "weather_wind_speed_kph": float(match.get("weather_wind_speed_kph", np.nan)),
    }


def build_team_histories(dataframe: pd.DataFrame) -> dict[str, list[dict]]:
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Sequence construction missing columns: {sorted(missing)}")
    ordered = dataframe.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="coerce")
    ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index()
    histories: dict[str, list[dict]] = {}
    for _, row in ordered.iterrows():
        current_date = row["Date"]
        for team in [row["HomeTeam"], row["AwayTeam"]]:
            histories.setdefault(str(team), [])
        home_record = _team_perspective_row(row, row["HomeTeam"])
        away_record = _team_perspective_row(row, row["AwayTeam"])
        home_record.update({"source_index": int(row["index"]), "source_date": current_date})
        away_record.update({"source_index": int(row["index"]), "source_date": current_date})
        histories[str(row["HomeTeam"])].append(home_record)
        histories[str(row["AwayTeam"])].append(away_record)
    return histories


def sequence_for_team(histories: dict[str, list[dict]], team: str, current_date, sequence_length: int) -> np.ndarray:
    current_date = pd.Timestamp(current_date)
    past = [row for row in histories.get(str(team), []) if pd.Timestamp(row["source_date"]) < current_date]
    selected = past[-sequence_length:]
    output = np.zeros((sequence_length, len(SEQUENCE_FEATURE_COLUMNS)), dtype=float)
    start = sequence_length - len(selected)
    for offset, row in enumerate(selected):
        output[start + offset] = [float(row.get(column, np.nan)) for column in SEQUENCE_FEATURE_COLUMNS]
    return output


def build_sequence_arrays(dataframe: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    histories = build_team_histories(dataframe)
    home_rows = []
    away_rows = []
    for _, row in dataframe.iterrows():
        home_rows.append(sequence_for_team(histories, row["HomeTeam"], row["Date"], sequence_length))
        away_rows.append(sequence_for_team(histories, row["AwayTeam"], row["Date"], sequence_length))
    return np.stack(home_rows), np.stack(away_rows)


def fit_sequence_scaler(train_home: np.ndarray, train_away: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    flat = np.concatenate([train_home.reshape(-1, train_home.shape[-1]), train_away.reshape(-1, train_away.shape[-1])])
    medians = np.nanmedian(flat, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    inds = np.where(np.isnan(flat))
    if len(inds[0]):
        flat = flat.copy()
        flat[inds] = np.take(medians, inds[1])
    scaler.fit(flat)
    return scaler


def transform_sequences(scaler: StandardScaler, array: np.ndarray) -> np.ndarray:
    shape = array.shape
    flat = array.reshape(-1, shape[-1])
    medians = np.nan_to_num(scaler.mean_, nan=0.0)
    inds = np.where(np.isnan(flat))
    if len(inds[0]):
        flat = flat.copy()
        flat[inds] = np.take(medians, inds[1])
    return scaler.transform(flat).reshape(shape).astype(np.float32)


def prepare_bundle(dataframe: pd.DataFrame, train: pd.DataFrame, subset: pd.DataFrame, preprocessor, numeric, categorical, seq_scaler, sequence_length: int) -> SequenceBundle:
    current_x = advanced.transform(preprocessor, subset, numeric, categorical).astype(np.float32)
    all_home, all_away = build_sequence_arrays(dataframe, sequence_length)
    home = transform_sequences(seq_scaler, all_home[subset.index.to_numpy()])
    away = transform_sequences(seq_scaler, all_away[subset.index.to_numpy()])
    y = subset[TARGET_COLUMN].astype(int).to_numpy()
    return SequenceBundle(current_x=current_x, home_sequence=home, away_sequence=away, y=y, dataframe=subset.copy())


class SequenceEncoder(nn.Module):
    def __init__(self, model_type: str, n_features: int, hidden_dim: int = 24, dropout: float = 0.15):
        super().__init__()
        self.model_type = model_type
        self.input_projection = nn.Linear(n_features, hidden_dim)
        if model_type == "gru":
            self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif model_type == "tcn":
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=2),
                nn.ReLU(),
            )
        elif model_type == "sequence_transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=48,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        else:
            raise ValueError(f"Unknown sequence model: {model_type}")

    def forward(self, x):
        x = self.input_projection(x)
        if self.model_type == "gru":
            _, hidden = self.encoder(x)
            return hidden[-1]
        if self.model_type == "tcn":
            encoded = self.encoder(x.transpose(1, 2))
            return encoded.mean(dim=-1)
        encoded = self.encoder(x)
        return encoded.mean(dim=1)


class TeamSequenceNetwork(nn.Module):
    def __init__(self, model_type: str, current_dim: int, sequence_features: int, hidden_dim: int = 24, dropout: float = 0.15):
        super().__init__()
        self.home_encoder = SequenceEncoder(model_type, sequence_features, hidden_dim, dropout)
        self.away_encoder = SequenceEncoder(model_type, sequence_features, hidden_dim, dropout)
        self.current_projection = nn.Sequential(nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, current_x, home_sequence, away_sequence):
        features = torch.cat(
            [self.current_projection(current_x), self.home_encoder(home_sequence), self.away_encoder(away_sequence)],
            dim=1,
        )
        return self.head(features).squeeze(-1)


class TeamSequenceClassifier:
    def __init__(self, model_type: str, seed: int, max_epochs: int = 40, patience: int = 5):
        self.model_type = model_type
        self.seed = int(seed)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.model_: TeamSequenceNetwork | None = None

    def fit(self, train: SequenceBundle, validation: SequenceBundle):
        set_random_seeds(self.seed)
        if len(np.unique(train.y)) < 2:
            self.base_rate_ = float(np.mean(train.y)) if len(train.y) else 0.5
            return self
        self.base_rate_ = None
        self.model_ = TeamSequenceNetwork(
            self.model_type,
            current_dim=train.current_x.shape[1],
            sequence_features=train.home_sequence.shape[-1],
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=0.001, weight_decay=0.01)
        loss_fn = nn.BCEWithLogitsLoss()
        train_tensors = tuple(torch.tensor(x, dtype=torch.float32) for x in [train.current_x, train.home_sequence, train.away_sequence])
        train_y = torch.tensor(train.y, dtype=torch.float32)
        val_tensors = tuple(torch.tensor(x, dtype=torch.float32) for x in [validation.current_x, validation.home_sequence, validation.away_sequence])
        val_y = torch.tensor(validation.y, dtype=torch.float32)
        best_loss = math.inf
        best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
        stale = 0
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            self.model_.train()
            order = torch.randperm(len(train_y), generator=generator)
            for start in range(0, len(order), 64):
                idx = order[start : start + 64]
                logits = self.model_(train_tensors[0][idx], train_tensors[1][idx], train_tensors[2][idx])
                loss = loss_fn(logits, train_y[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 2.0)
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(*val_tensors), val_y).item())
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, bundle: SequenceBundle) -> np.ndarray:
        if getattr(self, "base_rate_", None) is not None:
            p = np.full(len(bundle.y), self.base_rate_)
            return np.column_stack([1.0 - p, p])
        assert self.model_ is not None
        self.model_.eval()
        tensors = tuple(torch.tensor(x, dtype=torch.float32) for x in [bundle.current_x, bundle.home_sequence, bundle.away_sequence])
        with torch.no_grad():
            p = torch.sigmoid(self.model_(*tensors)).numpy()
        return np.column_stack([1.0 - p, p])


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


def select_validation_candidate(validation: pd.DataFrame, probabilities: np.ndarray, target_style: str) -> dict | None:
    scores = advanced.candidate_score(validation, probabilities, target_style)
    candidates = []
    for ah_threshold in THRESHOLDS:
        for score_threshold in advanced.candidate_thresholds(scores):
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
                }
            )
    if not candidates:
        return None
    return pd.DataFrame(candidates).sort_values(["validation_z_score", "validation_roi"], ascending=[False, False]).iloc[0].to_dict()


def run_sequence_nested(dataframe: pd.DataFrame, model_type: str, target_style: str, sequence_length: int, seed: int):
    set_random_seeds(seed)
    by_year_rows = []
    bet_frames = []
    metric_rows = []
    for split in advanced.make_temporal_splits(sorted(dataframe["season_end_year"].unique())):
        train = dataframe[dataframe["season_end_year"].isin(split.train_years)].copy()
        validation = dataframe[dataframe["season_end_year"] == split.validation_year].copy()
        test = dataframe[dataframe["season_end_year"] == split.test_year].copy()
        preprocessor, numeric, categorical = advanced.fit_preprocessor(train)
        train_home_all, train_away_all = build_sequence_arrays(dataframe, sequence_length)
        seq_scaler = fit_sequence_scaler(train_home_all[train.index.to_numpy()], train_away_all[train.index.to_numpy()])
        train_bundle = prepare_bundle(dataframe, train, train, preprocessor, numeric, categorical, seq_scaler, sequence_length)
        validation_bundle = prepare_bundle(dataframe, train, validation, preprocessor, numeric, categorical, seq_scaler, sequence_length)
        test_bundle = prepare_bundle(dataframe, train, test, preprocessor, numeric, categorical, seq_scaler, sequence_length)
        model_name = f"{model_type}_n{sequence_length}_{target_style}_seed_{seed}"
        model = TeamSequenceClassifier(model_type, seed).fit(train_bundle, validation_bundle)
        validation_probability = model.predict_proba(validation_bundle)[:, 1]
        test_probability = model.predict_proba(test_bundle)[:, 1]
        metric_rows.append(probability_metrics(test, test_probability, model_name, split.test_year))
        selected = select_validation_candidate(validation, validation_probability, target_style)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": model_name,
                    "test_year": split.test_year,
                    "train_years": ";".join(str(y) for y in split.train_years),
                    "validation_year": split.validation_year,
                    "selected_filter": "no_valid_validation_candidate",
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                }
            )
            continue
        test_scores = advanced.candidate_score(test, test_probability, target_style)
        selected_test = test[
            (pd.to_numeric(test["ah_line"], errors="coerce") <= float(selected["selected_threshold"]))
            & (test_scores >= float(selected["selected_score_threshold"]))
        ].copy()
        selected_test["strategy"] = model_name
        selected_test["model_family"] = model_type
        selected_test["target_style"] = target_style
        selected_test["sequence_length"] = sequence_length
        selected_test["seed"] = seed
        summary = summarize(selected_test)
        by_year_rows.append(
            {
                "strategy": model_name,
                "test_year": split.test_year,
                "train_years": ";".join(str(y) for y in split.train_years),
                "validation_year": split.validation_year,
                "selected_threshold": selected["selected_threshold"],
                "selected_score_threshold": selected["selected_score_threshold"],
                "selected_filter": f"{target_style}_score>={selected['selected_score_threshold']:.6f}",
                "validation_bets": selected["validation_bets"],
                "validation_roi": selected["validation_roi"],
                "test_bets": summary["bets"],
                "test_profit": summary["profit"],
                "test_roi": summary["roi"],
            }
        )
        if len(selected_test):
            bet_frames.append(selected_test)
    return (
        pd.DataFrame(by_year_rows),
        pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def aggregate_model_rows(overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_family, target_style, sequence_length), group in overall.groupby(["model_family", "target_style", "sequence_length"], dropna=False):
        rows.append(
            {
                "strategy": f"{model_family}_n{int(sequence_length)}_{target_style}_seed_mean",
                "model_family": f"{model_family}_seed_mean",
                "target_style": target_style,
                "sequence_length": sequence_length,
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


def classify(overall: pd.DataFrame, metrics: pd.DataFrame) -> tuple[str, str]:
    seq = overall[overall["model_family"].astype(str).str.contains("seed_mean", na=False)].copy()
    profitable = seq[(seq["profit"] > 0) & (seq["roi"] > 0)].copy()
    if profitable.empty:
        return "reject", "No sequence model family produced positive mean out-of-sample profit."
    best = profitable.sort_values(["z_score", "roi", "profit"], ascending=[False, False, False]).iloc[0]
    calibration = metrics[metrics["model"].str.startswith(str(best["strategy"]).replace("_seed_mean", "_seed_"), na=False)]
    calibration_ok = bool(len(calibration) and calibration["brier"].mean() < calibration["market_brier"].mean())
    clv_ok = bool(best["avg_clv_pp"] > 0 and best["clv_positive_rate"] >= 0.50)
    concentration_ok = bool(best["top3_home_bet_share"] <= 0.58 and best["home_hhi_bets"] <= 0.15)
    robustness_ok = bool(best["z_score"] >= 2.028 and best["bets"] >= 80)
    if clv_ok and calibration_ok and concentration_ok and robustness_ok and best["z_score"] >= 2.5:
        return "confirmed edge", f"{best['strategy']} clears all strict gates."
    if clv_ok and calibration_ok and concentration_ok and best["z_score"] >= 1.5:
        return "paper challenger", f"{best['strategy']} is positive and has enough diagnostics for paper-challenger status."
    return "research only", f"{best['strategy']} is historically positive, but CLV, robustness, calibration, and concentration do not all improve."


def write_outputs(dataframe, overall, by_year, bets, seasonal, exclude, metrics):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(SUMMARY_PATH, index=False)
    overall.to_csv(DETAIL_DIR / "overall.csv", index=False)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "nested_bets.csv", index=False)
    seasonal.to_csv(DETAIL_DIR / "seasonal.csv", index=False)
    exclude.to_csv(DETAIL_DIR / "exclude_each_season.csv", index=False)
    metrics.to_csv(DETAIL_DIR / "probability_metrics.csv", index=False)
    classification, rationale = classify(overall, metrics)
    baseline = overall[overall["model_family"].isin(["rule_benchmark", "logistic", "xgboost", "torch_ft_transformer_seed_mean"])].copy()
    lines = [
        "# E0 Away AH Team Sequence Model Review",
        "",
        "Scope: controlled E0 team-sequence experiment for Away AH big home favourites. Histories use only matches with `Date` strictly before the current match date.",
        "",
        "No broad model search was run. Raw match data was not edited. External APIs were not used. Closing odds are used only after selection for CLV diagnostics.",
        "",
        f"Torch version: {torch.__version__}. Sequence lengths: {SEQUENCE_LENGTHS}. Seeds: {SEEDS}. Sequence feature count: {len(SEQUENCE_FEATURE_COLUMNS)}.",
        "",
        "## Overall Results",
        "",
        advanced.markdown_table(
            overall,
            ["strategy", "model_family", "target_style", "sequence_length", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "top3_away_bet_share", "home_hhi_bets", "away_hhi_bets"],
            ["Strategy", "Family", "Target", "N", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Top3 away", "Home HHI", "Away HHI"],
        ),
        "",
        "## Baseline Comparison",
        "",
        advanced.markdown_table(
            baseline,
            ["strategy", "model_family", "target_style", "sequence_length", "bets", "profit", "roi", "z_score", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share"],
            ["Strategy", "Family", "Target", "N", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "CLV+ rate", "Top3 home"],
        ),
        "",
        "## Probability Calibration",
        "",
        advanced.markdown_table(metrics, ["model", "test_year", "rows", "log_loss", "market_log_loss", "brier", "market_brier", "ece", "market_ece"], ["Model", "Year", "Rows", "Log loss", "Market log loss", "Brier", "Market Brier", "ECE", "Market ECE"]),
        "",
        "## Season By Season",
        "",
        advanced.markdown_table(seasonal, ["strategy", "season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate"], ["Strategy", "Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate"]),
        "",
        "## Exclude Each Season",
        "",
        advanced.markdown_table(exclude, ["strategy", "exclusion_reason", "excluded_season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp"], ["Strategy", "Reason", "Excluded season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp"]),
        "",
        "## Methodology",
        "",
        "- Nested temporal validation only: train seasons before validation season, validation before held-out test season.",
        "- Current feature scalers/encoders are fit only on train seasons.",
        "- Sequence scalers are fit only on train-season sequence rows.",
        "- Team histories use only prior matches with `source_date < current_date`; the current match is not included in its own sequence.",
        "- Closing odds are absent from current and sequence feature matrices.",
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


def main():
    dataframe = advanced.prepare_e0_data()
    rule_overall, rule_by_year = advanced.run_rule_benchmarks()
    true_tabular = pd.read_csv("outputs/reports/e0_away_ah_true_tabular_transformer_summary.csv")
    baseline_rows = true_tabular[
        true_tabular["strategy"].isin(
            [
                "logistic_binary_cover",
                "logistic_market_residual",
                "xgboost_binary_cover",
                "xgboost_market_residual",
                "torch_ft_transformer_binary_cover_seed_mean",
                "torch_ft_transformer_market_residual_seed_mean",
            ]
        )
    ].copy()
    by_year_frames = [rule_by_year]
    bet_frames = []
    metric_frames = []
    overall_rows = []
    for n in SEQUENCE_LENGTHS:
        for model_type in MODEL_TYPES:
            for target_style in ["binary_cover", "market_residual"]:
                for seed in SEEDS:
                    by_year, bets, metrics = run_sequence_nested(dataframe, model_type, target_style, n, seed)
                    by_year_frames.append(by_year)
                    if len(bets):
                        bet_frames.append(bets)
                    if len(metrics):
                        metric_frames.append(metrics)
                    overall_rows.append(advanced.overall_row(f"{model_type}_n{n}_{target_style}_seed_{seed}", bets, model_type, target_style) | {"sequence_length": n})
    model_overall = pd.DataFrame(overall_rows)
    seed_means = aggregate_model_rows(model_overall)
    overall = pd.concat([rule_overall, baseline_rows, model_overall, seed_means], ignore_index=True, sort=False)
    by_year = pd.concat(by_year_frames, ignore_index=True, sort=False)
    bets = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    seasonal = advanced.seasonal_rows(bets)
    exclude = advanced.exclude_rows(bets)
    write_outputs(dataframe, overall, by_year, bets, seasonal, exclude, metrics)
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
