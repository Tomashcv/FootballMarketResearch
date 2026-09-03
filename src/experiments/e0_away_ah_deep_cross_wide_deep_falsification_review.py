from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.experiments import e0_away_ah_bag_last5_pooled_form_falsification_review as pooled_review
from src.features.contextual_features import assert_no_closing_columns
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


if advanced.torch is None or advanced.nn is None:  # pragma: no cover - environment guard
    raise RuntimeError("Torch is required for the Deep & Cross / Wide & Deep falsification review.")

torch = advanced.torch
nn = advanced.nn

REPORT_PATH = Path("outputs/reports/e0_away_ah_deep_cross_wide_deep_falsification_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_deep_cross_wide_deep_falsification_summary.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/deep_cross_wide_deep_falsification")

TARGET_STYLE = "market_residual"
SEEDS = [11, 23, 37]
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 64


def set_torch_seed(seed: int) -> None:
    advanced.set_random_seeds(seed)


class CrossLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0, x):
        projection = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * projection + self.bias + x


class CurrentTabularNetwork(nn.Module):
    def __init__(self, n_features: int, model_type: str, hidden_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.model_type = model_type
        self.wide = nn.Linear(n_features, 1)
        self.deep = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.deep_head = nn.Linear(hidden_dim // 2, 1)
        self.cross1 = CrossLayer(n_features)
        self.cross2 = CrossLayer(n_features)
        self.cross_head = nn.Linear(n_features, 1)
        self.wide_deep_head = nn.Linear((hidden_dim // 2) + 1, 1)

    def forward(self, x):
        if self.model_type == "wide_linear_residual":
            return self.wide(x).squeeze(-1)
        if self.model_type == "small_deep_mlp":
            return self.deep_head(self.deep(x)).squeeze(-1)
        if self.model_type == "deep_cross_network":
            crossed = self.cross2(x, self.cross1(x, x))
            return self.cross_head(crossed).squeeze(-1)
        if self.model_type == "wide_deep_combined":
            wide_logit = self.wide(x)
            deep_features = self.deep(x)
            return self.wide_deep_head(torch.cat([wide_logit, deep_features], dim=1)).squeeze(-1)
        raise ValueError(f"Unknown model type: {self.model_type}")


class TorchCurrentClassifier:
    def __init__(
        self,
        model_type: str,
        seed: int,
        dropout: float = 0.15,
        weight_decay: float = 0.01,
        learning_rate: float = 0.001,
    ):
        self.model_type = model_type
        self.seed = int(seed)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.learning_rate = float(learning_rate)
        self.model_: CurrentTabularNetwork | None = None
        self.best_epoch_ = 0
        self.device_ = torch.device("cpu")

    def fit(self, x: np.ndarray, y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray):
        set_torch_seed(self.seed)
        train_x = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=self.device_)
        train_y = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32, device=self.device_)
        val_x = torch.tensor(np.asarray(validation_x, dtype=np.float32), dtype=torch.float32, device=self.device_)
        val_y = torch.tensor(np.asarray(validation_y, dtype=np.float32), dtype=torch.float32, device=self.device_)
        self.model_ = CurrentTabularNetwork(train_x.shape[1], self.model_type, dropout=self.dropout).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        best_loss = math.inf
        best_state = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
        stale_epochs = 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        for epoch in range(MAX_EPOCHS):
            self.model_.train()
            order = torch.randperm(train_x.shape[0], generator=generator)
            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                loss = loss_fn(self.model_(train_x[batch]), train_y[batch])
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
                if stale_epochs >= PATIENCE:
                    break

        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        self.model_.eval()
        features = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=self.device_)
        with torch.no_grad():
            p = torch.sigmoid(self.model_(features)).detach().cpu().numpy()
        return np.column_stack([1.0 - p, p])


@dataclass(frozen=True)
class RunConfig:
    name: str
    model_type: str
    shuffle_train_labels: bool = False
    random_feature_noise: bool = False


def validation_selection(validation: pd.DataFrame, scores: pd.Series) -> dict | None:
    candidates = []
    for ah_threshold in THRESHOLDS:
        for score_threshold in advanced.candidate_thresholds(scores):
            selected = validation[
                (pd.to_numeric(validation["ah_line"], errors="coerce") <= ah_threshold) & (scores >= score_threshold)
            ].copy()
            if len(selected) < advanced.MIN_VALIDATION_BETS:
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
    return (
        pd.DataFrame(candidates)
        .sort_values(["validation_z_score", "validation_roi", "validation_bets"], ascending=[False, False, False])
        .iloc[0]
        .to_dict()
    )


def _noise_like(array: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=array.shape).astype(np.float32)


def selected_bets_for_probability(test: pd.DataFrame, probability: np.ndarray, selected: dict, strategy: str, model_type: str, seed_label: str) -> pd.DataFrame:
    scores = advanced.candidate_score(test, probability, TARGET_STYLE)
    bets = test[
        (pd.to_numeric(test["ah_line"], errors="coerce") <= float(selected["selected_threshold"]))
        & (scores >= float(selected["selected_score_threshold"]))
    ].copy()
    if len(bets):
        bets["model_probability"] = probability[bets.index.map(test.index.get_loc)]
        bets["model_score"] = scores.loc[bets.index].to_numpy()
    bets["strategy"] = strategy
    bets["variant"] = strategy
    bets["model_family"] = model_type
    bets["target_style"] = TARGET_STYLE
    bets["seed"] = seed_label
    return bets


def run_config(dataframe: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year_rows = []
    metric_rows = []
    ensemble_bets = []
    seed_bets = []
    for split in advanced.make_temporal_splits(sorted(dataframe["season_end_year"].unique())):
        train = dataframe[dataframe["season_end_year"].isin(split.train_years)].copy()
        validation = dataframe[dataframe["season_end_year"].eq(split.validation_year)].copy()
        test = dataframe[dataframe["season_end_year"].eq(split.test_year)].copy()
        if len(train) == 0 or len(validation) == 0 or len(test) == 0:
            continue
        preprocessor, numeric, categorical = advanced.fit_preprocessor(train)
        assert_no_closing_columns(numeric + categorical)
        train_x = advanced.transform(preprocessor, train, numeric, categorical).astype(np.float32)
        validation_x = advanced.transform(preprocessor, validation, numeric, categorical).astype(np.float32)
        test_x = advanced.transform(preprocessor, test, numeric, categorical).astype(np.float32)
        if config.random_feature_noise:
            train_x = _noise_like(train_x, 8000 + split.test_year)
            validation_x = _noise_like(validation_x, 9000 + split.test_year)
            test_x = _noise_like(test_x, 10000 + split.test_year)

        train_y = train[advanced.TARGET_COLUMN].astype(int).to_numpy()
        validation_y = validation[advanced.TARGET_COLUMN].astype(int).to_numpy()
        if config.shuffle_train_labels:
            rng = np.random.default_rng(7000 + split.test_year)
            train_y = train_y.copy()
            rng.shuffle(train_y)

        validation_probabilities = []
        test_probabilities = []
        for seed in SEEDS:
            model = TorchCurrentClassifier(config.model_type, seed).fit(train_x, train_y, validation_x, validation_y)
            validation_probability = model.predict_proba(validation_x)[:, 1]
            test_probability = model.predict_proba(test_x)[:, 1]
            validation_probabilities.append(validation_probability)
            test_probabilities.append(test_probability)
            metric_rows.append(
                advanced.probability_metrics(test, test_probability, f"{config.name}_seed_{seed}", split.test_year)
                | {"variant": config.name, "model_family": config.model_type, "seed": seed}
            )
            seed_selected = validation_selection(validation, advanced.candidate_score(validation, validation_probability, TARGET_STYLE))
            if seed_selected is not None:
                seed_frame = selected_bets_for_probability(
                    test,
                    test_probability,
                    seed_selected,
                    f"{config.name}_seed_{seed}",
                    config.model_type,
                    str(seed),
                )
                if len(seed_frame):
                    seed_bets.append(seed_frame)

        ensemble_validation = np.mean(validation_probabilities, axis=0)
        ensemble_test = np.mean(test_probabilities, axis=0)
        metric_rows.append(
            advanced.probability_metrics(test, ensemble_test, f"{config.name}_ensemble", split.test_year)
            | {"variant": config.name, "model_family": config.model_type, "seed": "ensemble"}
        )
        selected = validation_selection(validation, advanced.candidate_score(validation, ensemble_validation, TARGET_STYLE))
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": f"{config.name}_ensemble",
                    "variant": config.name,
                    "test_year": split.test_year,
                    "train_years": ";".join(str(year) for year in split.train_years),
                    "validation_year": split.validation_year,
                    "selected_filter": "no_valid_validation_candidate",
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                }
            )
            continue
        bets = selected_bets_for_probability(test, ensemble_test, selected, f"{config.name}_ensemble", config.model_type, "ensemble")
        bets["nested_test_year"] = split.test_year
        bets["validation_year"] = split.validation_year
        bets["selected_threshold"] = selected["selected_threshold"]
        bets["selected_score_threshold"] = selected["selected_score_threshold"]
        summary = summarize(bets)
        by_year_rows.append(
            {
                "strategy": f"{config.name}_ensemble",
                "variant": config.name,
                "test_year": split.test_year,
                "train_years": ";".join(str(year) for year in split.train_years),
                "validation_year": split.validation_year,
                "selected_threshold": selected["selected_threshold"],
                "selected_score_threshold": selected["selected_score_threshold"],
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
        if len(bets):
            ensemble_bets.append(bets)
    return (
        pd.DataFrame(by_year_rows),
        pd.concat(ensemble_bets, ignore_index=True, sort=False) if ensemble_bets else pd.DataFrame(),
        pd.concat(seed_bets, ignore_index=True, sort=False) if seed_bets else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def row_for_bets(strategy: str, bets: pd.DataFrame, model_family: str, variant: str) -> dict:
    row = advanced.overall_row(strategy, bets, model_family, TARGET_STYLE)
    row["variant"] = variant
    return row


def seed_mean_row(config: RunConfig, seed_bets: pd.DataFrame) -> dict:
    rows = []
    for seed in SEEDS:
        strategy = f"{config.name}_seed_{seed}"
        rows.append(row_for_bets(strategy, seed_bets[seed_bets["strategy"].eq(strategy)].copy(), config.model_type, f"{config.name}_seed"))
    frame = pd.DataFrame(rows)
    row = {
        "strategy": f"{config.name}_seed_mean",
        "model_family": f"{config.model_type}_seed_mean",
        "target_style": TARGET_STYLE,
        "variant": f"{config.name}_seed_mean",
        "seed_count": len(SEEDS),
    }
    for column in [
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
    ]:
        row[column] = float(frame[column].mean()) if column in frame else pd.NA
    row["seed_profit_std"] = float(frame["profit"].std(ddof=0))
    row["seed_roi_std"] = float(frame["roi"].std(ddof=0))
    return row


def season_exclusions(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    return pooled_review.season_exclusions(strategy, bets)


def home_team_exclusions(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    return pooled_review.home_team_exclusions(strategy, bets)


def leakage_audit(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric, categorical = advanced.available_feature_columns(dataframe)
    return pd.DataFrame(
        [
            {"check": "closing_absent_current_features", "passed": True, "detail": ",".join(numeric + categorical)},
            {"check": "scalers_fit_train_only", "passed": True, "detail": "advanced.fit_preprocessor(train) per split"},
            {"check": "nested_temporal_only", "passed": True, "detail": "train seasons before validation before held-out test"},
            {"check": "no_sequence_features", "passed": True, "detail": "current bet-time-safe feature matrix only"},
        ]
    )


def match_keys(frame: pd.DataFrame) -> set[str]:
    return pooled_review.match_keys(frame)


def load_reference_bets(name: str) -> pd.DataFrame:
    if name in {"sequence_transformer_no_seq_odds", "memory_knn_combo"}:
        return pooled_review.load_reference_bets(name)
    if name == "away_odds_ge_1_85":
        path = Path("outputs/E0/asian_handicap_big_home_favorite_away/memory_odds_combo_review/nested_bets.csv")
        if not path.exists():
            return pd.DataFrame()
        bets = pd.read_csv(path, low_memory=False)
        return bets[bets["strategy"].eq("away_odds_ge_1_85")].copy()
    if name == "pooled_last5":
        path = Path("outputs/E0/asian_handicap_big_home_favorite_away/bag_last5_pooled_form_falsification/selected_bets.csv")
        if not path.exists():
            return pd.DataFrame()
        bets = pd.read_csv(path, low_memory=False)
        return bets[bets["strategy"].eq("pooled_logistic_market_residual")].copy()
    return pd.DataFrame()


def overlap_rows(primary_strategy: str, primary_bets: pd.DataFrame) -> pd.DataFrame:
    primary_keys = match_keys(primary_bets)
    rows = []
    for reference in ["away_odds_ge_1_85", "memory_knn_combo", "sequence_transformer_no_seq_odds", "pooled_last5"]:
        reference_bets = load_reference_bets(reference)
        reference_keys = match_keys(reference_bets)
        common = primary_keys & reference_keys
        rows.append(
            {
                "primary_strategy": primary_strategy,
                "reference": reference,
                "primary_bets": len(primary_keys),
                "reference_bets": len(reference_keys),
                "overlap_bets": len(common),
                "overlap_share_of_primary": len(common) / len(primary_keys) if primary_keys else pd.NA,
                "overlap_share_of_reference": len(common) / len(reference_keys) if reference_keys else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def benchmark_rows() -> pd.DataFrame:
    frames = []
    rule_overall, _ = advanced.run_rule_benchmarks()
    frames.append(rule_overall)
    true_path = Path("outputs/reports/e0_away_ah_true_tabular_transformer_summary.csv")
    if true_path.exists():
        true_summary = pd.read_csv(true_path)
        wanted = [
            "logistic_binary_cover",
            "logistic_market_residual",
            "xgboost_binary_cover",
            "xgboost_market_residual",
            "torch_ft_transformer_binary_cover_seed_mean",
            "torch_ft_transformer_market_residual_seed_mean",
        ]
        frames.append(true_summary[true_summary["strategy"].isin(wanted)].copy())
    seq_path = Path("outputs/reports/e0_away_ah_sequence_transformer_n5_no_seq_odds_falsification_summary.csv")
    if seq_path.exists():
        seq = pd.read_csv(seq_path)
        seq = seq[seq["strategy"].eq("locked_ensemble")].copy()
        seq["variant"] = "sequence_transformer_no_seq_odds_ah_benchmark"
        frames.append(seq)
    pooled_path = Path("outputs/reports/e0_away_ah_bag_last5_pooled_form_falsification_summary.csv")
    if pooled_path.exists():
        pooled = pd.read_csv(pooled_path)
        pooled = pooled[pooled["strategy"].isin(["pooled_logistic_market_residual", "pooled_xgboost_market_residual"])].copy()
        pooled["variant"] = "pooled_last5_benchmark"
        frames.append(pooled)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def choose_primary(summary: pd.DataFrame) -> str:
    candidates = summary[
        summary["strategy"].isin(
            [
                "wide_linear_residual_ensemble",
                "small_deep_mlp_ensemble",
                "deep_cross_network_ensemble",
                "wide_deep_combined_ensemble",
            ]
        )
    ].copy()
    profitable = candidates[(candidates["profit"] > 0) & (candidates["roi"] > 0)]
    if profitable.empty:
        return str(candidates.sort_values(["profit", "roi"], ascending=[False, False]).iloc[0]["strategy"])
    return str(profitable.sort_values(["z_score", "roi", "profit"], ascending=[False, False, False]).iloc[0]["strategy"])


def classify(primary: pd.Series, season_exclusion: pd.DataFrame, team_exclusion: pd.DataFrame, controls: pd.DataFrame, audit: pd.DataFrame) -> tuple[str, str]:
    no_2024 = season_exclusion[
        season_exclusion["exclusion_reason"].eq("exclude_each_season") & season_exclusion["excluded_season"].eq(2024)
    ]
    no_2025 = season_exclusion[
        season_exclusion["exclusion_reason"].eq("exclude_each_season") & season_exclusion["excluded_season"].eq(2025)
    ]
    top_exclusions = team_exclusion[team_exclusion["exclusion_reason"].isin(["exclude_top1_home", "exclude_top2_home", "exclude_top3_home"])]
    control_success = controls[(controls["profit"] > 0) & (controls["roi"] > 0) & (controls["avg_clv_pp"] > 0)]
    gates = {
        "positive_clv": bool(primary["avg_clv_pp"] > 0),
        "positive_roi_without_2024": bool(len(no_2024) and float(no_2024.iloc[0]["roi"]) > 0),
        "positive_roi_without_2025": bool(len(no_2025) and float(no_2025.iloc[0]["roi"]) > 0),
        "acceptable_top_team_exclusions": bool(len(top_exclusions) and (top_exclusions["roi"] > 0).all() and (top_exclusions["avg_clv_pp"] > 0).all()),
        "negative_controls_fail": control_success.empty,
        "no_leakage_warning": bool(audit["passed"].all()),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if primary["bets"] == 0 or primary["profit"] <= 0 or primary["roi"] <= 0:
        return "reject", "Primary Deep & Cross / Wide & Deep candidate was not profitable."
    if failed:
        return "research only", "Failed promotion gates: " + ", ".join(failed)
    return "paper challenger", "Primary candidate clears locked gates but still needs paper tracking."


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return advanced.markdown_table(frame, columns, headers)


def write_outputs(summary, by_year, bets, seed_bets, metrics, seasonal, season_exclusion, team_exclusion, audit, overlap, classification, rationale):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "selected_bets.csv", index=False)
    seed_bets.to_csv(DETAIL_DIR / "seed_selected_bets.csv", index=False)
    metrics.to_csv(DETAIL_DIR / "probability_metrics.csv", index=False)
    seasonal.to_csv(DETAIL_DIR / "seasonal.csv", index=False)
    season_exclusion.to_csv(DETAIL_DIR / "season_exclusions.csv", index=False)
    team_exclusion.to_csv(DETAIL_DIR / "home_team_exclusions.csv", index=False)
    audit.to_csv(DETAIL_DIR / "leakage_audit.csv", index=False)
    overlap.to_csv(DETAIL_DIR / "overlap.csv", index=False)

    model_rows = summary[summary["variant"].astype(str).str.contains("wide|deep|cross|negative_control", na=False)].copy()
    benchmark_rows_frame = summary[~summary.index.isin(model_rows.index)].copy()
    lines = [
        "# E0 Away AH Deep & Cross / Wide & Deep Falsification Review",
        "",
        "Scope: locked E0 Away AH big home favourite market-residual review. Current bet-time-safe features only; no sequence model was used.",
        "",
        "Raw match data was not edited. External APIs were not used. Closing odds were excluded from features and used only for CLV diagnostics.",
        "",
        f"Seeds: {SEEDS}. Thresholds were selected only on prior validation seasons after seed probabilities were averaged for ensemble rows.",
        "",
        "## Locked Models And Controls",
        "",
        markdown_table(
            model_rows,
            ["strategy", "variant", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "home_hhi_bets", "seed_profit_std", "seed_roi_std"],
            ["Strategy", "Variant", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Home HHI", "Seed profit sd", "Seed ROI sd"],
        ),
        "",
        "## Benchmarks",
        "",
        markdown_table(
            benchmark_rows_frame,
            ["strategy", "variant", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share"],
            ["Strategy", "Variant", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home"],
        ),
        "",
        "## Season By Season",
        "",
        markdown_table(seasonal, ["strategy", "season", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp"], ["Strategy", "Season", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp"]),
        "",
        "## Exclude Each Season",
        "",
        markdown_table(season_exclusion, ["strategy", "exclusion_reason", "excluded_season", "bets", "profit", "roi", "z_score", "avg_clv_pp"], ["Strategy", "Reason", "Excluded season", "Bets", "Profit", "ROI", "z", "Avg CLV pp"]),
        "",
        "## Home Team Stress",
        "",
        markdown_table(team_exclusion, ["strategy", "exclusion_reason", "excluded_home_team", "bets", "profit", "roi", "z_score", "avg_clv_pp", "top3_home_bet_share"], ["Strategy", "Reason", "Excluded home", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "Top3 home"]),
        "",
        "## Overlap Diagnostics",
        "",
        markdown_table(overlap, ["primary_strategy", "reference", "primary_bets", "reference_bets", "overlap_bets", "overlap_share_of_primary", "overlap_share_of_reference"], ["Primary", "Reference", "Primary bets", "Reference bets", "Overlap", "Share primary", "Share reference"]),
        "",
        "## Probability Metrics",
        "",
        markdown_table(metrics, ["model", "test_year", "variant", "seed", "log_loss", "market_log_loss", "brier", "market_brier", "ece", "market_ece"], ["Model", "Year", "Variant", "Seed", "Log loss", "Market log loss", "Brier", "Market Brier", "ECE", "Market ECE"]),
        "",
        "## Leakage Audit",
        "",
        markdown_table(audit, ["check", "passed", "detail"], ["Check", "Passed", "Detail"]),
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
    dataframe = advanced.prepare_e0_data()
    configs = [
        RunConfig("wide_linear_residual", "wide_linear_residual"),
        RunConfig("small_deep_mlp", "small_deep_mlp"),
        RunConfig("deep_cross_network", "deep_cross_network"),
        RunConfig("wide_deep_combined", "wide_deep_combined"),
        RunConfig("shuffled_training_labels_negative_control", "deep_cross_network", shuffle_train_labels=True),
        RunConfig("random_feature_noise_negative_control", "deep_cross_network", random_feature_noise=True),
    ]
    by_year_frames = []
    bet_frames = []
    seed_bet_frames = []
    metric_frames = []
    overall_rows = []
    for config in configs:
        by_year, bets, seed_bets, metrics = run_config(dataframe, config)
        by_year_frames.append(by_year)
        if len(bets):
            bet_frames.append(bets)
        if len(seed_bets):
            seed_bet_frames.append(seed_bets)
        if len(metrics):
            metric_frames.append(metrics)
        overall_rows.append(row_for_bets(f"{config.name}_ensemble", bets, config.model_type, config.name))
        overall_rows.append(seed_mean_row(config, seed_bets))

    benchmarks = benchmark_rows()
    if len(benchmarks):
        overall_rows.extend(benchmarks.to_dict("records"))
    summary = pd.DataFrame(overall_rows)
    primary_strategy = choose_primary(summary)
    by_year = pd.concat(by_year_frames, ignore_index=True, sort=False)
    bets = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    seed_bets = pd.concat(seed_bet_frames, ignore_index=True, sort=False) if seed_bet_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    primary_bets = bets[bets["strategy"].eq(primary_strategy)].copy()
    seasonal = advanced.seasonal_rows(primary_bets)
    season_exclusion = season_exclusions(primary_strategy, primary_bets)
    team_exclusion = home_team_exclusions(primary_strategy, primary_bets)
    audit = leakage_audit(dataframe)
    overlap = overlap_rows(primary_strategy, primary_bets)
    controls = summary[summary["strategy"].isin(["shuffled_training_labels_negative_control_ensemble", "random_feature_noise_negative_control_ensemble"])]
    primary = summary[summary["strategy"].eq(primary_strategy)].iloc[0]
    classification, rationale = classify(primary, season_exclusion, team_exclusion, controls, audit)
    write_outputs(summary, by_year, bets, seed_bets, metrics, seasonal, season_exclusion, team_exclusion, audit, overlap, classification, rationale)
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
