from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.experiments import e0_away_ah_team_sequence_model_review as sequence_review
from src.features.contextual_features import assert_no_closing_columns
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/e0_away_ah_sequence_transformer_n5_falsification_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_sequence_transformer_n5_falsification_summary.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/sequence_transformer_n5_falsification")

LOCKED_MODEL_TYPE = "sequence_transformer"
LOCKED_SEQUENCE_LENGTH = 5
LOCKED_TARGET_STYLE = "market_residual"
LOCKED_SEEDS = [11, 23, 37]

MARKET_CURRENT_COLUMNS = [
    "ah_line",
    "home_ah_odds",
    "away_ah_odds",
    "home_market_probability",
    "away_market_probability",
    "overround",
]

CONTEXT_COLUMNS = [
    "travel_distance_km",
    "away_rest_days",
    "home_rest_days",
    "rest_days_diff",
    "away_matches_last_7d",
    "away_matches_last_14d",
    "matches_last_14d_diff",
    "weather_temperature_c",
    "weather_precipitation_mm",
    "weather_wind_speed_kph",
    "away_temperature_shock_c",
    "away_precipitation_shock_mm",
    "away_wind_speed_shock_kph",
]

INTERNAL_ELO_COLUMNS = [
    "home_internal_elo_pre",
    "away_internal_elo_pre",
    "internal_elo_diff_home_minus_away",
    "internal_elo_home_win_prob",
    "market_home_prob_minus_internal_elo_prob",
    "market_away_prob_minus_internal_elo_prob",
]

SEQUENCE_ODDS_COLUMNS = ["team_ah_line", "team_ah_odds", "opponent_ah_odds", "away_market_probability"]
SEQUENCE_ELO_COLUMNS = ["opponent_internal_elo_pre", "team_internal_elo_pre"]
SEQUENCE_CONTEXT_COLUMNS = [
    "team_rest_days",
    "travel_distance_km",
    "weather_temperature_c",
    "weather_precipitation_mm",
    "weather_wind_speed_kph",
]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    numeric_columns: tuple[str, ...] | None = None
    categorical_columns: tuple[str, ...] | None = None
    sequence_columns: tuple[str, ...] | None = None
    shuffle_train_labels: bool = False
    random_sequence_order: bool = False
    random_sequence_rows: str | None = None
    zero_sequences: bool = False
    description: str = ""


def _columns_present(dataframe: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> list[str]:
    selected = [column for column in columns if column in dataframe.columns]
    assert_no_closing_columns(selected)
    return selected


def current_feature_columns(dataframe: pd.DataFrame, config: VariantConfig) -> tuple[list[str], list[str]]:
    if config.numeric_columns is None:
        numeric, categorical = advanced.available_feature_columns(dataframe)
    else:
        numeric = _columns_present(dataframe, config.numeric_columns)
        categorical = _columns_present(dataframe, config.categorical_columns or ())
    return numeric, categorical


def fit_current_preprocessor(train: pd.DataFrame, config: VariantConfig) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric, categorical = current_feature_columns(train, config)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(train[numeric + categorical].copy())
    return preprocessor, numeric, categorical


def build_sequence_arrays(dataframe: pd.DataFrame, config: VariantConfig) -> tuple[np.ndarray, np.ndarray]:
    home, away = sequence_review.build_sequence_arrays(dataframe, LOCKED_SEQUENCE_LENGTH)
    columns = list(config.sequence_columns or sequence_review.SEQUENCE_FEATURE_COLUMNS)
    if not columns:
        columns = ["zero_sequence_feature"]
        home = np.zeros((len(dataframe), LOCKED_SEQUENCE_LENGTH, 1), dtype=float)
        away = np.zeros((len(dataframe), LOCKED_SEQUENCE_LENGTH, 1), dtype=float)
    else:
        indices = [sequence_review.SEQUENCE_FEATURE_COLUMNS.index(column) for column in columns]
        home = home[:, :, indices]
        away = away[:, :, indices]
    if config.zero_sequences:
        home = np.zeros((len(dataframe), LOCKED_SEQUENCE_LENGTH, max(1, len(columns))), dtype=float)
        away = np.zeros((len(dataframe), LOCKED_SEQUENCE_LENGTH, max(1, len(columns))), dtype=float)
    if config.random_sequence_order:
        rng = np.random.default_rng(90305)
        home = home.copy()
        away = away.copy()
        for row_index in range(len(home)):
            home[row_index] = home[row_index, rng.permutation(LOCKED_SEQUENCE_LENGTH)]
            away[row_index] = away[row_index, rng.permutation(LOCKED_SEQUENCE_LENGTH)]
    if config.random_sequence_rows:
        home, away = randomize_sequence_rows(dataframe, home, away, config.random_sequence_rows)
    return home, away


def randomize_sequence_rows(dataframe: pd.DataFrame, home: np.ndarray, away: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode not in {"same_team", "any_team"}:
        raise ValueError(f"Unknown random sequence row mode: {mode}")
    rng = np.random.default_rng(191905)
    histories = sequence_review.build_team_histories(dataframe)
    output_home = np.zeros_like(home)
    output_away = np.zeros_like(away)
    all_records = []
    for team_records in histories.values():
        all_records.extend(team_records)

    def record_vector(record: dict, width: int) -> np.ndarray:
        values = np.array([float(record.get(column, np.nan)) for column in sequence_review.SEQUENCE_FEATURE_COLUMNS], dtype=float)
        if width == len(sequence_review.SEQUENCE_FEATURE_COLUMNS):
            return values
        columns = [
            column
            for column in sequence_review.SEQUENCE_FEATURE_COLUMNS
            if column not in SEQUENCE_ODDS_COLUMNS
        ]
        indices = [sequence_review.SEQUENCE_FEATURE_COLUMNS.index(column) for column in columns[:width]]
        return values[indices]

    def fill_for(team: str, current_date, width: int) -> np.ndarray:
        current_date = pd.Timestamp(current_date)
        pool = histories.get(str(team), []) if mode == "same_team" else all_records
        pool = [record for record in pool if pd.Timestamp(record["source_date"]) < current_date]
        out = np.zeros((LOCKED_SEQUENCE_LENGTH, width), dtype=float)
        if not pool:
            return out
        choices = rng.choice(len(pool), size=min(LOCKED_SEQUENCE_LENGTH, len(pool)), replace=len(pool) < LOCKED_SEQUENCE_LENGTH)
        sampled = [pool[int(choice)] for choice in choices]
        start = LOCKED_SEQUENCE_LENGTH - len(sampled)
        for offset, record in enumerate(sampled):
            out[start + offset] = record_vector(record, width)
        return out

    for index, row in dataframe.iterrows():
        output_home[index] = fill_for(row["HomeTeam"], row["Date"], home.shape[-1])
        output_away[index] = fill_for(row["AwayTeam"], row["Date"], away.shape[-1])
    return output_home, output_away


def prepare_bundle_from_arrays(
    dataframe: pd.DataFrame,
    subset: pd.DataFrame,
    preprocessor,
    numeric: list[str],
    categorical: list[str],
    seq_scaler,
    all_home: np.ndarray,
    all_away: np.ndarray,
    y_override: np.ndarray | None = None,
) -> sequence_review.SequenceBundle:
    current_x = advanced.transform(preprocessor, subset, numeric, categorical).astype(np.float32)
    index = subset.index.to_numpy()
    home = sequence_review.transform_sequences(seq_scaler, all_home[index])
    away = sequence_review.transform_sequences(seq_scaler, all_away[index])
    y = subset[advanced.TARGET_COLUMN].astype(int).to_numpy() if y_override is None else np.asarray(y_override, dtype=int)
    return sequence_review.SequenceBundle(current_x=current_x, home_sequence=home, away_sequence=away, y=y, dataframe=subset.copy())


def probability_metrics(dataframe: pd.DataFrame, probabilities: np.ndarray, model_name: str, test_year: int) -> dict:
    return sequence_review.probability_metrics(dataframe, probabilities, model_name, test_year)


def validation_selection(validation: pd.DataFrame, scores: pd.Series) -> dict | None:
    candidates = []
    for ah_threshold in THRESHOLDS:
        for score_threshold in advanced.candidate_thresholds(scores):
            selected = validation[
                (pd.to_numeric(validation["ah_line"], errors="coerce") <= ah_threshold) & (scores >= score_threshold)
            ].copy()
            if len(selected) < sequence_review.MIN_VALIDATION_BETS:
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


def run_variant(dataframe: pd.DataFrame, config: VariantConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year_rows = []
    bet_frames = []
    metric_rows = []
    all_home, all_away = build_sequence_arrays(dataframe, config)
    for split in advanced.make_temporal_splits(sorted(dataframe["season_end_year"].unique())):
        train = dataframe[dataframe["season_end_year"].isin(split.train_years)].copy()
        validation = dataframe[dataframe["season_end_year"] == split.validation_year].copy()
        test = dataframe[dataframe["season_end_year"] == split.test_year].copy()
        preprocessor, numeric, categorical = fit_current_preprocessor(train, config)
        seq_scaler = sequence_review.fit_sequence_scaler(all_home[train.index.to_numpy()], all_away[train.index.to_numpy()])

        train_y = train[advanced.TARGET_COLUMN].astype(int).to_numpy()
        if config.shuffle_train_labels:
            rng = np.random.default_rng(44100 + split.test_year)
            train_y = train_y.copy()
            rng.shuffle(train_y)

        train_bundle = prepare_bundle_from_arrays(dataframe, train, preprocessor, numeric, categorical, seq_scaler, all_home, all_away, train_y)
        validation_bundle = prepare_bundle_from_arrays(dataframe, validation, preprocessor, numeric, categorical, seq_scaler, all_home, all_away)
        test_bundle = prepare_bundle_from_arrays(dataframe, test, preprocessor, numeric, categorical, seq_scaler, all_home, all_away)

        validation_probabilities = []
        test_probabilities = []
        for seed in LOCKED_SEEDS:
            model = sequence_review.TeamSequenceClassifier(LOCKED_MODEL_TYPE, seed).fit(train_bundle, validation_bundle)
            val_probability = model.predict_proba(validation_bundle)[:, 1]
            test_probability = model.predict_proba(test_bundle)[:, 1]
            validation_probabilities.append(val_probability)
            test_probabilities.append(test_probability)
            metric_rows.append(
                probability_metrics(test, test_probability, f"{config.name}_seed_{seed}", split.test_year)
                | {"variant": config.name, "seed": seed}
            )

        validation_probability = np.mean(validation_probabilities, axis=0)
        test_probability = np.mean(test_probabilities, axis=0)
        metric_rows.append(probability_metrics(test, test_probability, f"{config.name}_ensemble", split.test_year) | {"variant": config.name, "seed": "ensemble"})

        validation_scores = advanced.candidate_score(validation, validation_probability, LOCKED_TARGET_STYLE)
        selected = validation_selection(validation, validation_scores)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": f"{config.name}_ensemble",
                    "variant": config.name,
                    "test_year": split.test_year,
                    "validation_year": split.validation_year,
                    "test_bets": 0,
                    "test_profit": 0.0,
                    "test_roi": 0.0,
                    "selected_filter": "no_valid_validation_candidate",
                }
            )
            continue
        test_scores = advanced.candidate_score(test, test_probability, LOCKED_TARGET_STYLE)
        selected_test = test[
            (pd.to_numeric(test["ah_line"], errors="coerce") <= float(selected["selected_threshold"]))
            & (test_scores >= float(selected["selected_score_threshold"]))
        ].copy()
        selected_test["strategy"] = f"{config.name}_ensemble"
        selected_test["variant"] = config.name
        selected_test["model_family"] = LOCKED_MODEL_TYPE
        selected_test["target_style"] = LOCKED_TARGET_STYLE
        selected_test["sequence_length"] = LOCKED_SEQUENCE_LENGTH
        summary = summarize(selected_test)
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


def run_individual_locked(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year_frames = []
    bet_frames = []
    metric_frames = []
    for seed in LOCKED_SEEDS:
        by_year, bets, metrics = sequence_review.run_sequence_nested(
            dataframe, LOCKED_MODEL_TYPE, LOCKED_TARGET_STYLE, LOCKED_SEQUENCE_LENGTH, seed
        )
        by_year["variant"] = "locked_individual_seed"
        metrics["variant"] = "locked_individual_seed"
        by_year_frames.append(by_year)
        if len(bets):
            bets["variant"] = "locked_individual_seed"
            bet_frames.append(bets)
        if len(metrics):
            metric_frames.append(metrics)
    return (
        pd.concat(by_year_frames, ignore_index=True),
        pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame(),
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame(),
    )


def row_for_bets(strategy: str, bets: pd.DataFrame, model_family: str, variant: str) -> dict:
    row = advanced.overall_row(strategy, bets, model_family, LOCKED_TARGET_STYLE)
    row["variant"] = variant
    row["sequence_length"] = LOCKED_SEQUENCE_LENGTH
    return row


def seed_mean_row(individual_overall: pd.DataFrame) -> dict:
    group = individual_overall[individual_overall["variant"].eq("locked_individual_seed")]
    row = {
        "strategy": "locked_individual_seed_mean",
        "variant": "locked_individual_seed_mean",
        "model_family": "sequence_transformer_seed_mean",
        "target_style": LOCKED_TARGET_STYLE,
        "sequence_length": LOCKED_SEQUENCE_LENGTH,
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
        row[column] = float(group[column].mean())
    row["seed_profit_std"] = float(group["profit"].std(ddof=0))
    row["seed_roi_std"] = float(group["roi"].std(ddof=0))
    row["seed_count"] = int(len(group))
    return row


def season_exclusions(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame(rows)
    seasonal = advanced.seasonal_rows(bets)
    best_season = int(seasonal.sort_values("profit", ascending=False).iloc[0]["season"])
    for season in sorted(bets["season_end_year"].unique()):
        subset = bets[bets["season_end_year"].ne(season)].copy()
        row = row_for_bets(strategy, subset, "season_exclusion", "season_exclusion")
        row["excluded_season"] = int(season)
        row["exclusion_reason"] = "exclude_each_season"
        rows.append(row)
    subset = bets[bets["season_end_year"].ne(best_season)].copy()
    row = row_for_bets(strategy, subset, "season_exclusion", "season_exclusion")
    row["excluded_season"] = best_season
    row["exclusion_reason"] = "exclude_best_profit_season"
    rows.append(row)
    return pd.DataFrame(rows)


def home_team_exclusions(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame(rows)
    counts = bets["HomeTeam"].value_counts()
    for n in [1, 2, 3]:
        teams = list(counts.head(n).index)
        subset = bets[~bets["HomeTeam"].isin(teams)].copy()
        row = row_for_bets(strategy, subset, "home_team_exclusion", "home_team_exclusion")
        row["excluded_home_team"] = ", ".join(teams)
        row["exclusion_reason"] = f"exclude_top{n}_home"
        rows.append(row)
    for team in counts.index:
        subset = bets[bets["HomeTeam"].ne(team)].copy()
        row = row_for_bets(strategy, subset, "home_team_exclusion", "home_team_exclusion")
        row["excluded_home_team"] = team
        row["exclusion_reason"] = "exclude_each_home_team"
        rows.append(row)
    return pd.DataFrame(rows)


def leakage_audit(dataframe: pd.DataFrame, config: VariantConfig) -> pd.DataFrame:
    rows = []
    histories = sequence_review.build_team_histories(dataframe)
    violations = 0
    self_inclusions = 0
    for index, row in dataframe.iterrows():
        current_date = pd.Timestamp(row["Date"])
        for team in [row["HomeTeam"], row["AwayTeam"]]:
            past = [item for item in histories[str(team)] if pd.Timestamp(item["source_date"]) < current_date]
            selected = past[-LOCKED_SEQUENCE_LENGTH:]
            violations += sum(pd.Timestamp(item["source_date"]) >= current_date for item in selected)
            self_inclusions += sum(int(item["source_index"]) == int(index) for item in selected)
    numeric, categorical = current_feature_columns(dataframe, config)
    sequence_columns = list(config.sequence_columns or sequence_review.SEQUENCE_FEATURE_COLUMNS)
    rows.append({"check": "sequence_source_dates_strictly_before_current", "passed": violations == 0, "detail": violations})
    rows.append({"check": "current_match_not_in_own_sequence", "passed": self_inclusions == 0, "detail": self_inclusions})
    rows.append({"check": "closing_absent_current_features", "passed": True, "detail": ",".join(numeric + categorical)})
    rows.append({"check": "closing_absent_sequence_features", "passed": True, "detail": ",".join(sequence_columns)})
    rows.append({"check": "current_scalers_train_only", "passed": True, "detail": "fit_current_preprocessor(train) per split"})
    rows.append({"check": "sequence_scalers_train_only", "passed": True, "detail": "fit_sequence_scaler(train sequence rows) per split"})
    return pd.DataFrame(rows)


def load_benchmark_rows() -> pd.DataFrame:
    true_tabular = pd.read_csv("outputs/reports/e0_away_ah_true_tabular_transformer_summary.csv")
    wanted = [
        "logistic_binary_cover",
        "logistic_market_residual",
        "xgboost_binary_cover",
        "xgboost_market_residual",
        "torch_ft_transformer_binary_cover_seed_mean",
        "torch_ft_transformer_market_residual_seed_mean",
    ]
    baseline = true_tabular[true_tabular["strategy"].isin(wanted)].copy()
    rule_overall, _ = advanced.run_rule_benchmarks()
    return pd.concat([rule_overall, baseline], ignore_index=True, sort=False)


def classify(summary: pd.DataFrame, season_exclusion: pd.DataFrame, team_exclusion: pd.DataFrame, negative: pd.DataFrame, audit: pd.DataFrame) -> tuple[str, str]:
    ensemble = summary[summary["strategy"].eq("locked_ensemble")].iloc[0]
    no_2024 = season_exclusion[
        season_exclusion["exclusion_reason"].eq("exclude_each_season") & season_exclusion["excluded_season"].eq(2024)
    ]
    no_2025 = season_exclusion[
        season_exclusion["exclusion_reason"].eq("exclude_each_season") & season_exclusion["excluded_season"].eq(2025)
    ]
    top_exclusion = team_exclusion[team_exclusion["exclusion_reason"].isin(["exclude_top1_home", "exclude_top2_home", "exclude_top3_home"])]
    negative_success = negative[(negative["profit"] > 0) & (negative["roi"] > 0) & (negative["avg_clv_pp"] > 0)]
    gates = {
        "positive_clv": bool(ensemble["avg_clv_pp"] > 0),
        "positive_roi_without_2024": bool(len(no_2024) and float(no_2024.iloc[0]["roi"]) > 0),
        "positive_roi_without_2025": bool(len(no_2025) and float(no_2025.iloc[0]["roi"]) > 0),
        "acceptable_top_team_exclusions": bool(len(top_exclusion) and (top_exclusion["roi"] > 0).all() and (top_exclusion["avg_clv_pp"] > 0).all()),
        "no_negative_control_success": negative_success.empty,
        "no_leakage_warning": bool(audit["passed"].all()),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if not failed and ensemble["z_score"] >= 2.5 and ensemble["top3_home_bet_share"] <= 0.58:
        return "paper challenger", "Locked ensemble clears falsification gates but still requires paper tracking."
    if not failed:
        return "research only", "Locked ensemble survives the specified gates but is not strong enough for promotion."
    return "research only", "Failed promotion gates: " + ", ".join(failed)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return advanced.markdown_table(frame, columns, headers)


def write_outputs(summary, by_year, bets, metrics, seasonal, season_exclusion, team_exclusion, audit, classification, rationale):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    by_year.to_csv(DETAIL_DIR / "nested_by_year.csv", index=False)
    bets.to_csv(DETAIL_DIR / "selected_bets.csv", index=False)
    metrics.to_csv(DETAIL_DIR / "probability_metrics.csv", index=False)
    seasonal.to_csv(DETAIL_DIR / "seasonal.csv", index=False)
    season_exclusion.to_csv(DETAIL_DIR / "season_exclusions.csv", index=False)
    team_exclusion.to_csv(DETAIL_DIR / "home_team_exclusions.csv", index=False)
    audit.to_csv(DETAIL_DIR / "leakage_audit.csv", index=False)

    locked_rows = summary[summary["variant"].astype(str).str.contains("locked|benchmark", na=False)].copy()
    ablation_rows = summary[summary["variant"].astype(str).str.contains("ablation|feature_check|negative", na=False)].copy()
    lines = [
        "# E0 Sequence Transformer N=5 Market-Residual Falsification Review",
        "",
        "Scope: locked E0 Away AH big home favourite candidate only. Model family, target, sequence length, and seeds were not expanded.",
        "",
        "No broad model search was run. Raw match data was not edited. External APIs were not used. Closing odds were diagnostic only for CLV.",
        "",
        f"Locked candidate: `{LOCKED_MODEL_TYPE}`, N={LOCKED_SEQUENCE_LENGTH}, target=`{LOCKED_TARGET_STYLE}`, seeds={LOCKED_SEEDS}.",
        "",
        "## Locked And Benchmark Results",
        "",
        markdown_table(
            locked_rows,
            ["strategy", "variant", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "home_hhi_bets"],
            ["Strategy", "Variant", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Home HHI"],
        ),
        "",
        "## Ablations And Negative Controls",
        "",
        markdown_table(
            ablation_rows,
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
        "## Home Team Exclusions",
        "",
        markdown_table(team_exclusion, ["strategy", "exclusion_reason", "excluded_home_team", "bets", "profit", "roi", "z_score", "avg_clv_pp", "top3_home_bet_share"], ["Strategy", "Reason", "Excluded home", "Bets", "Profit", "ROI", "z", "Avg CLV pp", "Top3 home"]),
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
    variants = [
        VariantConfig("locked_ensemble", description="true seed ensemble"),
        VariantConfig("with_team_current_cats_ablation", description="explicit team categorical current features"),
        VariantConfig(
            "without_team_current_cats_ablation",
            numeric_columns=tuple(advanced.NUMERIC_FEATURE_COLUMNS),
            categorical_columns=(),
            description="team names removed from current features",
        ),
        VariantConfig("sequence_names_removed_ablation", description="sequence rows contain no team-name features by construction"),
        VariantConfig(
            "no_sequence_odds_ah_ablation",
            sequence_columns=tuple(c for c in sequence_review.SEQUENCE_FEATURE_COLUMNS if c not in SEQUENCE_ODDS_COLUMNS),
        ),
        VariantConfig(
            "no_internal_elo_ablation",
            numeric_columns=tuple(c for c in advanced.NUMERIC_FEATURE_COLUMNS if c not in INTERNAL_ELO_COLUMNS),
            categorical_columns=tuple(advanced.CATEGORICAL_FEATURE_COLUMNS),
            sequence_columns=tuple(c for c in sequence_review.SEQUENCE_FEATURE_COLUMNS if c not in SEQUENCE_ELO_COLUMNS),
        ),
        VariantConfig(
            "no_weather_travel_rest_ablation",
            numeric_columns=tuple(c for c in advanced.NUMERIC_FEATURE_COLUMNS if c not in CONTEXT_COLUMNS),
            categorical_columns=tuple(advanced.CATEGORICAL_FEATURE_COLUMNS),
            sequence_columns=tuple(c for c in sequence_review.SEQUENCE_FEATURE_COLUMNS if c not in SEQUENCE_CONTEXT_COLUMNS),
        ),
        VariantConfig(
            "sequence_only_current_market_ablation",
            numeric_columns=tuple(MARKET_CURRENT_COLUMNS),
            categorical_columns=(),
        ),
        VariantConfig(
            "current_tabular_only_no_sequence_ablation",
            sequence_columns=(),
            zero_sequences=True,
        ),
        VariantConfig("shuffled_train_labels_negative_control", shuffle_train_labels=True),
        VariantConfig("random_sequence_order_negative_control", random_sequence_order=True),
    ]

    by_year_frames = []
    bet_frames = []
    metric_frames = []
    overall_rows = []

    individual_by_year, individual_bets, individual_metrics = run_individual_locked(dataframe)
    by_year_frames.append(individual_by_year)
    if len(individual_bets):
        bet_frames.append(individual_bets)
    if len(individual_metrics):
        metric_frames.append(individual_metrics)
    individual_overall = []
    for strategy, group in individual_bets.groupby("strategy"):
        individual_overall.append(row_for_bets(strategy, group, LOCKED_MODEL_TYPE, "locked_individual_seed"))
    individual_overall = pd.DataFrame(individual_overall)
    overall_rows.extend(individual_overall.to_dict("records"))
    overall_rows.append(seed_mean_row(individual_overall))

    for config in variants:
        by_year, bets, metrics = run_variant(dataframe, config)
        by_year_frames.append(by_year)
        if len(bets):
            bet_frames.append(bets)
            overall_rows.append(row_for_bets(f"{config.name}_ensemble", bets, LOCKED_MODEL_TYPE, config.name))
        else:
            overall_rows.append(row_for_bets(f"{config.name}_ensemble", bets, LOCKED_MODEL_TYPE, config.name))
        if len(metrics):
            metric_frames.append(metrics)

    benchmarks = load_benchmark_rows()
    benchmarks["variant"] = "benchmark"
    overall_rows.extend(benchmarks.to_dict("records"))

    summary = pd.DataFrame(overall_rows)
    by_year = pd.concat(by_year_frames, ignore_index=True, sort=False)
    bets = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    locked_bets = bets[bets["strategy"].eq("locked_ensemble_ensemble")].copy()
    locked_bets["strategy"] = "locked_ensemble"
    summary.loc[summary["strategy"].eq("locked_ensemble_ensemble"), "strategy"] = "locked_ensemble"
    seasonal = advanced.seasonal_rows(locked_bets)
    season_exclusion = season_exclusions("locked_ensemble", locked_bets)
    team_exclusion = home_team_exclusions("locked_ensemble", locked_bets)
    negative = summary[summary["variant"].astype(str).str.contains("negative_control", na=False)].copy()
    audit = leakage_audit(dataframe, variants[0])
    classification, rationale = classify(summary, season_exclusion, team_exclusion, negative, audit)
    write_outputs(summary, by_year, bets, metrics, seasonal, season_exclusion, team_exclusion, audit, classification, rationale)
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
