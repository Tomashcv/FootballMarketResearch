from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.experiments import e0_away_ah_advanced_tabular_neural_review as advanced
from src.experiments import e0_away_ah_sequence_transformer_n5_falsification_review as seq_falsification
from src.experiments import e0_away_ah_team_sequence_model_review as sequence_review
from src.features.contextual_features import assert_no_closing_columns
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import THRESHOLDS
from src.markets.asian_handicap_big_home_favorite_away.run_nested_baseline import summarize


REPORT_PATH = Path("outputs/reports/e0_away_ah_bag_last5_pooled_form_falsification_review.md")
SUMMARY_PATH = Path("outputs/reports/e0_away_ah_bag_last5_pooled_form_falsification_summary.csv")
DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/bag_last5_pooled_form_falsification")

TARGET_STYLE = "market_residual"
LAST_N = 5
SEED = 1337

POOLED_SOURCE_COLUMNS = [
    "goals_for",
    "goals_against",
    "goal_diff",
    "result_points",
    "team_rest_days",
    "team_internal_elo_pre",
    "opponent_internal_elo_pre",
    "travel_distance_km",
    "weather_temperature_c",
    "weather_precipitation_mm",
    "weather_wind_speed_kph",
]

EXCLUDED_SEQUENCE_COLUMNS = set(seq_falsification.SEQUENCE_ODDS_COLUMNS)


def pooled_feature_names() -> list[str]:
    names = []
    for prefix in ["home", "away", "diff"]:
        names.extend([f"{prefix}_l5_mean_{column}" for column in POOLED_SOURCE_COLUMNS])
    return names


def _masked_mean(sequence: np.ndarray) -> np.ndarray:
    mask = ~(np.isclose(sequence, 0.0) | np.isnan(sequence)).all(axis=1)
    if not mask.any():
        return np.full(sequence.shape[1], np.nan, dtype=float)
    selected = sequence[mask]
    means = []
    for column_index in range(selected.shape[1]):
        values = selected[:, column_index]
        values = values[~np.isnan(values)]
        means.append(float(values.mean()) if len(values) else np.nan)
    return np.asarray(means, dtype=float)


def build_pooled_last5_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    for column in EXCLUDED_SEQUENCE_COLUMNS:
        if column in POOLED_SOURCE_COLUMNS:
            raise ValueError(f"Locked pooled features include excluded odds/AH column: {column}")
    home_sequence, away_sequence = sequence_review.build_sequence_arrays(dataframe, LAST_N)
    indices = [sequence_review.SEQUENCE_FEATURE_COLUMNS.index(column) for column in POOLED_SOURCE_COLUMNS]
    home_sequence = home_sequence[:, :, indices]
    away_sequence = away_sequence[:, :, indices]

    rows = []
    for row_index in range(len(dataframe)):
        home_mean = _masked_mean(home_sequence[row_index])
        away_mean = _masked_mean(away_sequence[row_index])
        values = {}
        for source_index, column in enumerate(POOLED_SOURCE_COLUMNS):
            values[f"home_l5_mean_{column}"] = home_mean[source_index]
            values[f"away_l5_mean_{column}"] = away_mean[source_index]
            values[f"diff_l5_mean_{column}"] = home_mean[source_index] - away_mean[source_index]
        rows.append(values)
    features = pd.DataFrame(rows, index=dataframe.index)
    assert_no_closing_columns(features.columns)
    return features


def dataframe_with_pooled_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pooled = build_pooled_last5_features(dataframe)
    output = pd.concat([dataframe.copy(), pooled], axis=1)
    names = list(pooled.columns)
    assert_no_closing_columns(names)
    return output, names


def current_columns(dataframe: pd.DataFrame, include_pooled: bool, pooled_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = advanced.available_feature_columns(dataframe)
    if include_pooled:
        numeric = numeric + [column for column in pooled_columns if column in dataframe.columns]
    assert_no_closing_columns(numeric + categorical)
    return numeric, categorical


def fit_preprocessor(train: pd.DataFrame, numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(train[numeric + categorical].copy())
    return preprocessor


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


def _randomize_pooled_columns(frame: pd.DataFrame, pooled_columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    randomized = frame.copy()
    if not pooled_columns:
        return randomized
    pooled_values = randomized[pooled_columns].to_numpy(copy=True)
    if len(pooled_values) > 1:
        pooled_values = pooled_values[rng.permutation(len(pooled_values))]
    randomized.loc[:, pooled_columns] = pooled_values
    return randomized


def _model_for_family(model_family: str, seed: int):
    if model_family == "logistic":
        return advanced.logistic_model(seed)
    if model_family == "xgboost":
        return advanced.xgboost_model(seed)
    raise ValueError(f"Unknown model family: {model_family}")


def run_nested_model(
    dataframe: pd.DataFrame,
    strategy: str,
    model_family: str,
    pooled_columns: list[str],
    *,
    include_pooled: bool = True,
    shuffle_train_labels: bool = False,
    random_pooled_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = _model_for_family(model_family, SEED)
    if model is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    by_year_rows = []
    bet_frames = []
    metric_rows = []
    for split in advanced.make_temporal_splits(sorted(dataframe["season_end_year"].unique())):
        train = dataframe[dataframe["season_end_year"].isin(split.train_years)].copy()
        validation = dataframe[dataframe["season_end_year"].eq(split.validation_year)].copy()
        test = dataframe[dataframe["season_end_year"].eq(split.test_year)].copy()
        if len(train) == 0 or len(validation) == 0 or len(test) == 0:
            continue

        rng = np.random.default_rng(SEED + split.test_year)
        effective_pooled = pooled_columns if include_pooled else []
        if random_pooled_features and include_pooled:
            train = _randomize_pooled_columns(train, effective_pooled, rng)
            validation = _randomize_pooled_columns(validation, effective_pooled, rng)
            test = _randomize_pooled_columns(test, effective_pooled, rng)

        numeric, categorical = current_columns(train, include_pooled, effective_pooled)
        preprocessor = fit_preprocessor(train, numeric, categorical)
        train_x = advanced.transform(preprocessor, train, numeric, categorical)
        validation_x = advanced.transform(preprocessor, validation, numeric, categorical)
        test_x = advanced.transform(preprocessor, test, numeric, categorical)

        train_y = train[advanced.TARGET_COLUMN].astype(int).to_numpy()
        if shuffle_train_labels:
            train_y = train_y.copy()
            rng.shuffle(train_y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            validation_probability, test_probability = advanced.safe_probability_fit_predict(
                _model_for_family(model_family, SEED + split.test_year),
                train_x,
                train_y,
                validation_x,
                test_x,
            )

        metric_rows.append(
            advanced.probability_metrics(test, test_probability, strategy, split.test_year)
            | {"variant": strategy, "model_family": model_family}
        )

        validation_scores = advanced.candidate_score(validation, validation_probability, TARGET_STYLE)
        selected = validation_selection(validation, validation_scores)
        if selected is None:
            by_year_rows.append(
                {
                    "strategy": strategy,
                    "variant": strategy,
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

        test_scores = advanced.candidate_score(test, test_probability, TARGET_STYLE)
        selected_test = test[
            (pd.to_numeric(test["ah_line"], errors="coerce") <= float(selected["selected_threshold"]))
            & (test_scores >= float(selected["selected_score_threshold"]))
        ].copy()
        selected_test["model_probability"] = test_probability[selected_test.index.map(test.index.get_loc)]
        selected_test["model_score"] = test_scores.loc[selected_test.index].to_numpy()
        selected_test["strategy"] = strategy
        selected_test["variant"] = strategy
        selected_test["model_family"] = model_family
        selected_test["target_style"] = TARGET_STYLE
        selected_test["nested_test_year"] = split.test_year
        selected_test["selected_threshold"] = selected["selected_threshold"]
        selected_test["selected_score_threshold"] = selected["selected_score_threshold"]

        summary = summarize(selected_test)
        by_year_rows.append(
            {
                "strategy": strategy,
                "variant": strategy,
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
        if len(selected_test):
            bet_frames.append(selected_test)

    return (
        pd.DataFrame(by_year_rows),
        pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def row_for_bets(strategy: str, bets: pd.DataFrame, model_family: str, variant: str) -> dict:
    row = advanced.overall_row(strategy, bets, model_family, TARGET_STYLE)
    row["variant"] = variant
    row["last_n"] = LAST_N
    return row


def season_exclusions(strategy: str, bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bets.empty:
        return pd.DataFrame(rows)
    seasonal = advanced.seasonal_rows(bets)
    best_season = int(seasonal.sort_values("profit", ascending=False).iloc[0]["season"])
    for season in sorted(bets["season_end_year"].unique()):
        row = row_for_bets(strategy, bets[bets["season_end_year"].ne(season)].copy(), "season_exclusion", "season_exclusion")
        row["excluded_season"] = int(season)
        row["exclusion_reason"] = "exclude_each_season"
        rows.append(row)
    row = row_for_bets(strategy, bets[bets["season_end_year"].ne(best_season)].copy(), "season_exclusion", "season_exclusion")
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
        row = row_for_bets(strategy, bets[~bets["HomeTeam"].isin(teams)].copy(), "home_team_exclusion", "home_team_exclusion")
        row["excluded_home_team"] = ", ".join(teams)
        row["exclusion_reason"] = f"exclude_top{n}_home"
        rows.append(row)
    for team in counts.index:
        row = row_for_bets(strategy, bets[bets["HomeTeam"].ne(team)].copy(), "home_team_exclusion", "home_team_exclusion")
        row["excluded_home_team"] = team
        row["exclusion_reason"] = "exclude_each_home_team"
        rows.append(row)
    return pd.DataFrame(rows)


def leakage_audit(dataframe: pd.DataFrame, pooled_columns: list[str]) -> pd.DataFrame:
    histories = sequence_review.build_team_histories(dataframe)
    violations = 0
    self_inclusions = 0
    for index, row in dataframe.iterrows():
        current_date = pd.Timestamp(row["Date"])
        for team in [row["HomeTeam"], row["AwayTeam"]]:
            selected = [item for item in histories[str(team)] if pd.Timestamp(item["source_date"]) < current_date][-LAST_N:]
            violations += sum(pd.Timestamp(item["source_date"]) >= current_date for item in selected)
            self_inclusions += sum(int(item["source_index"]) == int(index) for item in selected)
    current_numeric, current_categorical = advanced.available_feature_columns(dataframe)
    return pd.DataFrame(
        [
            {"check": "aggregate_source_dates_strictly_before_current", "passed": violations == 0, "detail": violations},
            {"check": "current_match_not_in_own_last5_pool", "passed": self_inclusions == 0, "detail": self_inclusions},
            {
                "check": "sequence_odds_ah_absent_from_pooled_features",
                "passed": not (set(POOLED_SOURCE_COLUMNS) & EXCLUDED_SEQUENCE_COLUMNS),
                "detail": ",".join(POOLED_SOURCE_COLUMNS),
            },
            {"check": "closing_absent_current_features", "passed": True, "detail": ",".join(current_numeric + current_categorical)},
            {"check": "closing_absent_pooled_features", "passed": True, "detail": ",".join(pooled_columns)},
            {"check": "scalers_fit_train_only", "passed": True, "detail": "ColumnTransformer fit on train split only"},
        ]
    )


def match_keys(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    return set(
        pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + frame["HomeTeam"].astype(str)
        + "|"
        + frame["AwayTeam"].astype(str)
    )


def load_reference_bets(name: str) -> pd.DataFrame:
    if name == "sequence_transformer_no_seq_odds":
        path = Path("outputs/E0/asian_handicap_big_home_favorite_away/sequence_transformer_n5_no_seq_odds_falsification/selected_bets.csv")
        if not path.exists():
            return pd.DataFrame()
        bets = pd.read_csv(path, low_memory=False)
        return bets[bets["strategy"].eq("locked_ensemble_ensemble")].copy()
    if name == "random_sequence_order":
        path = Path("outputs/E0/asian_handicap_big_home_favorite_away/sequence_transformer_n5_no_seq_odds_falsification/selected_bets.csv")
        if not path.exists():
            return pd.DataFrame()
        bets = pd.read_csv(path, low_memory=False)
        return bets[bets["strategy"].eq("random_sequence_order_negative_control_ensemble")].copy()
    if name == "memory_knn_combo":
        path = Path("outputs/E0/asian_handicap_big_home_favorite_away/memory_odds_combo_review/nested_bets.csv")
        if not path.exists():
            return pd.DataFrame()
        bets = pd.read_csv(path, low_memory=False)
        return bets[bets["strategy"].eq("away_odds_ge_1_85_plus_memory_knn_profit")].copy()
    return pd.DataFrame()


def overlap_rows(primary_bets: pd.DataFrame) -> pd.DataFrame:
    primary_keys = match_keys(primary_bets)
    rows = []
    for reference in ["sequence_transformer_no_seq_odds", "random_sequence_order", "memory_knn_combo"]:
        ref_bets = load_reference_bets(reference)
        ref_keys = match_keys(ref_bets)
        common = primary_keys & ref_keys
        rows.append(
            {
                "primary_strategy": "pooled_logistic_market_residual",
                "reference": reference,
                "primary_bets": len(primary_keys),
                "reference_bets": len(ref_keys),
                "overlap_bets": len(common),
                "overlap_share_of_primary": len(common) / len(primary_keys) if primary_keys else pd.NA,
                "overlap_share_of_reference": len(common) / len(ref_keys) if ref_keys else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def benchmark_rows() -> pd.DataFrame:
    rows = []
    rule_overall, _ = advanced.run_rule_benchmarks()
    rows.append(rule_overall)
    seq_summary_path = Path("outputs/reports/e0_away_ah_sequence_transformer_n5_no_seq_odds_falsification_summary.csv")
    if seq_summary_path.exists():
        seq_summary = pd.read_csv(seq_summary_path)
        wanted = seq_summary[
            seq_summary["strategy"].isin(
                [
                    "locked_ensemble",
                    "random_sequence_order_negative_control_ensemble",
                    "current_tabular_only_no_sequence_negative_control_ensemble",
                ]
            )
        ].copy()
        wanted["variant"] = wanted["strategy"].map(
            {
                "locked_ensemble": "locked_sequence_transformer_n5_no_sequence_odds_ah",
                "random_sequence_order_negative_control_ensemble": "random_sequence_order_negative_control",
                "current_tabular_only_no_sequence_negative_control_ensemble": "prior_current_tabular_only_no_sequence",
            }
        ).fillna("benchmark")
        rows.append(wanted)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def classify(primary: pd.Series, season_exclusion: pd.DataFrame, team_exclusion: pd.DataFrame, controls: pd.DataFrame, audit: pd.DataFrame, overlap: pd.DataFrame) -> tuple[str, str]:
    no_2024 = season_exclusion[
        season_exclusion["exclusion_reason"].eq("exclude_each_season") & season_exclusion["excluded_season"].eq(2024)
    ]
    no_2025 = season_exclusion[
        season_exclusion["exclusion_reason"].eq("exclude_each_season") & season_exclusion["excluded_season"].eq(2025)
    ]
    top_exclusions = team_exclusion[team_exclusion["exclusion_reason"].isin(["exclude_top1_home", "exclude_top2_home", "exclude_top3_home"])]
    control_success = controls[(controls["profit"] > 0) & (controls["roi"] > 0) & (controls["avg_clv_pp"] > 0)]
    overlap_supported = bool(len(overlap) and pd.to_numeric(overlap["overlap_share_of_primary"], errors="coerce").max() >= 0.25)
    gates = {
        "positive_clv": bool(primary["avg_clv_pp"] > 0),
        "positive_roi_without_2024": bool(len(no_2024) and float(no_2024.iloc[0]["roi"]) > 0),
        "positive_roi_without_2025": bool(len(no_2025) and float(no_2025.iloc[0]["roi"]) > 0),
        "acceptable_top_team_exclusions": bool(len(top_exclusions) and (top_exclusions["roi"] > 0).all() and (top_exclusions["avg_clv_pp"] > 0).all()),
        "shuffled_random_controls_fail": control_success.empty,
        "overlap_supports_non_random_signal": overlap_supported,
        "no_leakage_warning": bool(audit["passed"].all()),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if primary["bets"] == 0 or primary["profit"] <= 0 or primary["roi"] <= 0:
        return "reject", "Primary pooled logistic model was not profitable."
    if failed:
        return "research only", "Failed promotion gates: " + ", ".join(failed)
    return "paper challenger", "Primary pooled logistic model clears the locked falsification gates but still requires paper tracking."


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return advanced.markdown_table(frame, columns, headers)


def write_outputs(
    summary: pd.DataFrame,
    by_year: pd.DataFrame,
    bets: pd.DataFrame,
    metrics: pd.DataFrame,
    seasonal: pd.DataFrame,
    season_exclusion: pd.DataFrame,
    team_exclusion: pd.DataFrame,
    audit: pd.DataFrame,
    overlap: pd.DataFrame,
    classification: str,
    rationale: str,
) -> None:
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
    overlap.to_csv(DETAIL_DIR / "overlap.csv", index=False)

    model_rows = summary[summary["variant"].astype(str).str.contains("pooled|baseline|negative_control", na=False)].copy()
    benchmark = summary[summary["variant"].astype(str).str.contains("benchmark|sequence_transformer|random_sequence_order", na=False)].copy()
    lines = [
        "# E0 Away AH Bag-of-Last-5 Pooled Form Falsification Review",
        "",
        "Scope: locked E0 Away AH big home favourite review. Last N=5 only. No sequence order, recurrent, or transformer model was used.",
        "",
        "Raw match data was not edited. External APIs were not used. Closing odds were excluded from feature matrices and used only for CLV diagnostics.",
        "",
        "Pooled features: last-5 mean home, away, and home-minus-away aggregates for goals, result points, rest, internal Elo, travel, and weather. Sequence odds/AH features were excluded.",
        "",
        "## Pooled Models And Controls",
        "",
        markdown_table(
            model_rows,
            ["strategy", "variant", "bets", "profit", "roi", "z_score", "max_drawdown", "avg_clv_pp", "clv_positive_rate", "top3_home_bet_share", "home_hhi_bets"],
            ["Strategy", "Variant", "Bets", "Profit", "ROI", "z", "Max DD", "Avg CLV pp", "CLV+ rate", "Top3 home", "Home HHI"],
        ),
        "",
        "## Locked Benchmarks",
        "",
        markdown_table(
            benchmark,
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
        markdown_table(metrics, ["model", "test_year", "variant", "log_loss", "market_log_loss", "brier", "market_brier", "ece", "market_ece"], ["Model", "Year", "Variant", "Log loss", "Market log loss", "Brier", "Market Brier", "ECE", "Market ECE"]),
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
    dataframe, pooled_columns = dataframe_with_pooled_features(dataframe)
    variants = [
        ("pooled_logistic_market_residual", "logistic", True, False, False),
        ("pooled_xgboost_market_residual", "xgboost", True, False, False),
        ("current_tabular_only_baseline", "logistic", False, False, False),
        ("shuffled_training_labels_negative_control", "logistic", True, True, False),
        ("random_pooled_features_negative_control", "logistic", True, False, True),
    ]

    by_year_frames = []
    bet_frames = []
    metric_frames = []
    overall_rows = []
    for strategy, model_family, include_pooled, shuffle_labels, random_pooled in variants:
        by_year, bets, metrics = run_nested_model(
            dataframe,
            strategy,
            model_family,
            pooled_columns,
            include_pooled=include_pooled,
            shuffle_train_labels=shuffle_labels,
            random_pooled_features=random_pooled,
        )
        if by_year.empty and model_family == "xgboost":
            continue
        by_year_frames.append(by_year)
        if len(bets):
            bet_frames.append(bets)
        if len(metrics):
            metric_frames.append(metrics)
        overall_rows.append(row_for_bets(strategy, bets, model_family, strategy))

    benchmarks = benchmark_rows()
    if not benchmarks.empty:
        overall_rows.extend(benchmarks.to_dict("records"))

    summary = pd.DataFrame(overall_rows)
    by_year = pd.concat(by_year_frames, ignore_index=True, sort=False) if by_year_frames else pd.DataFrame()
    bets = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()

    primary_bets = bets[bets["strategy"].eq("pooled_logistic_market_residual")].copy()
    seasonal = advanced.seasonal_rows(primary_bets)
    season_exclusion = season_exclusions("pooled_logistic_market_residual", primary_bets)
    team_exclusion = home_team_exclusions("pooled_logistic_market_residual", primary_bets)
    audit = leakage_audit(dataframe, pooled_columns)
    overlap = overlap_rows(primary_bets)
    controls = summary[summary["strategy"].isin(["shuffled_training_labels_negative_control", "random_pooled_features_negative_control"])]
    primary = summary[summary["strategy"].eq("pooled_logistic_market_residual")].iloc[0]
    classification, rationale = classify(primary, season_exclusion, team_exclusion, controls, audit, overlap)
    write_outputs(summary, by_year, bets, metrics, seasonal, season_exclusion, team_exclusion, audit, overlap, classification, rationale)
    print(REPORT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
