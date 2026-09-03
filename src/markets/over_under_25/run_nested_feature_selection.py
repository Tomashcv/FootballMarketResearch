import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.metrics import binary_classification_metrics
from src.common.odds_utils import probability_to_logit
from src.common.paths import get_market_features_path
from src.common.paths import get_nested_by_year_path
from src.common.paths import get_nested_predictions_path


MARKET_NAME = "over_under_25"

REGULARIZATION_C_VALUES = [0.03, 0.07, 0.15, 0.25]
SHRINKAGE_ALPHA_VALUES = [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]


FEATURE_SETS = {
    "market_only_logreg": [
        "market_logit_over"
    ],

    "market_plus_overall_form_5": [
        "market_logit_over",
        "home_overall_goals_for_last_5",
        "home_overall_goals_against_last_5",
        "away_overall_goals_for_last_5",
        "away_overall_goals_against_last_5",
        "home_overall_total_goals_last_5",
        "away_overall_total_goals_last_5",
        "overall_goals_for_last_5_diff",
        "overall_goals_against_last_5_diff",
        "overall_total_goals_last_5_diff",
    ],

    "market_plus_home_away_context_5": [
        "market_logit_over",
        "home_home_goals_for_last_5",
        "home_home_goals_against_last_5",
        "home_home_total_goals_last_5",
        "home_home_over_25_rate_last_5",
        "away_away_goals_for_last_5",
        "away_away_goals_against_last_5",
        "away_away_total_goals_last_5",
        "away_away_over_25_rate_last_5",
        "home_away_context_goals_for_last_5_diff",
        "home_away_context_goals_against_last_5_diff",
        "home_away_context_total_goals_last_5_diff",
        "home_away_context_over_25_rate_last_5_diff",
    ],

    "market_plus_home_away_context_10": [
        "market_logit_over",
        "home_home_goals_for_last_10",
        "home_home_goals_against_last_10",
        "home_home_total_goals_last_10",
        "home_home_over_25_rate_last_10",
        "away_away_goals_for_last_10",
        "away_away_goals_against_last_10",
        "away_away_total_goals_last_10",
        "away_away_over_25_rate_last_10",
        "home_away_context_goals_for_last_10_diff",
        "home_away_context_goals_against_last_10_diff",
        "home_away_context_total_goals_last_10_diff",
        "home_away_context_over_25_rate_last_10_diff",
    ],

    "market_plus_matchup_expected_goals_5": [
        "market_logit_over",
        "expected_home_goals_simple_5",
        "expected_away_goals_simple_5",
        "expected_total_goals_simple_5",
        "expected_total_goals_simple_5_minus_2_5",
        "expected_total_goals_simple_5_minus_league_avg",
        "home_attack_vs_away_defense_5",
        "away_attack_vs_home_defense_5",
        "combined_home_away_over_rate_5",
        "combined_home_away_total_goals_5",
    ],

    "market_plus_matchup_expected_goals_10": [
        "market_logit_over",
        "expected_home_goals_simple_10",
        "expected_away_goals_simple_10",
        "expected_total_goals_simple_10",
        "expected_total_goals_simple_10_minus_2_5",
        "expected_total_goals_simple_10_minus_league_avg",
        "home_attack_vs_away_defense_10",
        "away_attack_vs_home_defense_10",
        "combined_home_away_over_rate_10",
        "combined_home_away_total_goals_10",
    ],

    "market_plus_full_v2": [
        "market_logit_over",

        "home_overall_goals_for_last_5",
        "home_overall_goals_against_last_5",
        "home_overall_total_goals_last_5",
        "home_overall_over_25_rate_last_5",
        "away_overall_goals_for_last_5",
        "away_overall_goals_against_last_5",
        "away_overall_total_goals_last_5",
        "away_overall_over_25_rate_last_5",

        "home_home_goals_for_last_5",
        "home_home_goals_against_last_5",
        "home_home_total_goals_last_5",
        "home_home_over_25_rate_last_5",
        "away_away_goals_for_last_5",
        "away_away_goals_against_last_5",
        "away_away_total_goals_last_5",
        "away_away_over_25_rate_last_5",

        "expected_home_goals_simple_5",
        "expected_away_goals_simple_5",
        "expected_total_goals_simple_5",
        "expected_total_goals_simple_5_minus_2_5",
        "expected_total_goals_simple_5_minus_league_avg",
        "combined_home_away_over_rate_5",
        "combined_home_away_total_goals_5",

        "home_overall_goals_for_last_10",
        "home_overall_goals_against_last_10",
        "away_overall_goals_for_last_10",
        "away_overall_goals_against_last_10",

        "home_home_goals_for_last_10",
        "home_home_goals_against_last_10",
        "away_away_goals_for_last_10",
        "away_away_goals_against_last_10",

        "expected_total_goals_simple_10",
        "expected_total_goals_simple_10_minus_2_5",
        "expected_total_goals_simple_10_minus_league_avg",

        "home_matches_played_before",
        "away_matches_played_before",
        "home_home_matches_played_before",
        "away_away_matches_played_before",
        "home_days_since_last_match",
        "away_days_since_last_match",
        "days_since_last_match_diff",

        "league_avg_goals_so_far",
        "league_over_25_rate_so_far",
        "market_over_probability_minus_league_over_rate",
        "market_over_probability_minus_0_5",
    ],
}


def clip_probability(probability):
    return min(max(float(probability), 0.000001), 0.999999)


def add_market_logit(dataframe):
    dataframe["market_probability_over"] = dataframe["market_probability_over"].apply(clip_probability)
    dataframe["market_logit_over"] = dataframe["market_probability_over"].apply(probability_to_logit)

    return dataframe


def check_feature_columns(dataframe):
    missing_columns = []

    for feature_set_name in FEATURE_SETS:
        for column in FEATURE_SETS[feature_set_name]:
            if column not in dataframe.columns:
                missing_columns.append(column)

    missing_columns = sorted(list(set(missing_columns)))

    if len(missing_columns) > 0:
        print("Colunas em falta:")
        for column in missing_columns:
            print("-", column)

        raise ValueError("Há feature columns em falta.")


def make_model(c_value):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=3000,
                    C=float(c_value),
                    solver="lbfgs"
                )
            ),
        ]
    )

    return model


def blend_with_market(raw_model_probabilities, market_probabilities, alpha):
    return (
        (1.0 - float(alpha)) * market_probabilities
        + float(alpha) * raw_model_probabilities
    )


def train_and_predict(train_dataframe, test_dataframe, feature_columns, c_value, alpha):
    x_train = train_dataframe[feature_columns].astype(float)
    y_train = train_dataframe["target_over_25"].astype(int)

    x_test = test_dataframe[feature_columns].astype(float)

    market_probabilities = test_dataframe["market_probability_over"].astype(float).values

    if float(alpha) == 0.0:
        return market_probabilities

    model = make_model(c_value)
    model.fit(x_train, y_train)

    raw_model_probabilities = model.predict_proba(x_test)[:, 1]
    final_probabilities = blend_with_market(raw_model_probabilities, market_probabilities, alpha)

    final_probabilities = np.clip(final_probabilities, 0.000001, 0.999999)

    return final_probabilities


def evaluate_candidate_on_validation(dataframe, feature_set_name, c_value, alpha, validation_year):
    feature_columns = FEATURE_SETS[feature_set_name]

    train_dataframe = dataframe[dataframe["season_end_year"] < validation_year].copy()
    validation_dataframe = dataframe[dataframe["season_end_year"] == validation_year].copy()

    if len(train_dataframe) == 0 or len(validation_dataframe) == 0:
        return None

    if train_dataframe["target_over_25"].nunique() < 2:
        return None

    model_probabilities = train_and_predict(
        train_dataframe,
        validation_dataframe,
        feature_columns,
        c_value,
        alpha
    )

    y_true = validation_dataframe["target_over_25"].astype(int).values
    market_probabilities = validation_dataframe["market_probability_over"].astype(float).values

    market_metrics = binary_classification_metrics(y_true, market_probabilities)
    model_metrics = binary_classification_metrics(y_true, model_probabilities)

    return {
        "feature_set": feature_set_name,
        "c_value": c_value,
        "alpha": alpha,
        "validation_year": validation_year,
        "validation_matches": len(validation_dataframe),

        "market_accuracy": market_metrics["accuracy"],
        "model_accuracy": model_metrics["accuracy"],
        "market_log_loss": market_metrics["log_loss"],
        "model_log_loss": model_metrics["log_loss"],
        "delta_log_loss": model_metrics["log_loss"] - market_metrics["log_loss"],

        "market_brier": market_metrics["brier"],
        "model_brier": model_metrics["brier"],
        "delta_brier": model_metrics["brier"] - market_metrics["brier"],

        "market_ece": market_metrics["ece"],
        "model_ece": model_metrics["ece"],
        "delta_ece": model_metrics["ece"] - market_metrics["ece"],
    }


def select_candidate_before_test_year(dataframe, test_year, min_train_years):
    years = sorted(dataframe["season_end_year"].dropna().unique().tolist())

    validation_results = []

    for validation_year in years:
        if validation_year >= test_year:
            continue

        train_years = [year for year in years if year < validation_year]

        if len(train_years) < min_train_years:
            continue

        for feature_set_name in FEATURE_SETS:
            for c_value in REGULARIZATION_C_VALUES:
                for alpha in SHRINKAGE_ALPHA_VALUES:
                    result = evaluate_candidate_on_validation(
                        dataframe,
                        feature_set_name,
                        c_value,
                        alpha,
                        validation_year
                    )

                    if result is not None:
                        validation_results.append(result)

    if len(validation_results) == 0:
        return None, None

    validation_results_dataframe = pd.DataFrame(validation_results)

    grouped = validation_results_dataframe.groupby(
        ["feature_set", "c_value", "alpha"],
        as_index=False
    ).agg(
        validation_matches=("validation_matches", "sum"),
        validation_delta_log_loss=("delta_log_loss", "mean"),
        validation_delta_brier=("delta_brier", "mean"),
        validation_delta_ece=("delta_ece", "mean"),
    )

    grouped = grouped.sort_values(
        [
            "validation_delta_log_loss",
            "validation_delta_brier",
            "validation_delta_ece",
            "alpha",
        ],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    grouped["validation_rank"] = grouped.index + 1

    selected = grouped.iloc[0].to_dict()

    return selected, grouped


def evaluate_test_year(dataframe, test_year, selected_candidate):
    selected_feature_set = selected_candidate["feature_set"]
    selected_c_value = selected_candidate["c_value"]
    selected_alpha = selected_candidate["alpha"]

    feature_columns = FEATURE_SETS[selected_feature_set]

    train_dataframe = dataframe[dataframe["season_end_year"] < test_year].copy()
    test_dataframe = dataframe[dataframe["season_end_year"] == test_year].copy()

    model_probabilities = train_and_predict(
        train_dataframe,
        test_dataframe,
        feature_columns,
        selected_c_value,
        selected_alpha
    )

    y_true = test_dataframe["target_over_25"].astype(int).values
    market_probabilities = test_dataframe["market_probability_over"].astype(float).values

    market_metrics = binary_classification_metrics(y_true, market_probabilities)
    model_metrics = binary_classification_metrics(y_true, model_probabilities)

    result = {
        "test_year": test_year,
        "selected_feature_set": selected_feature_set,
        "selected_c_value": selected_c_value,
        "selected_alpha": selected_alpha,
        "test_matches": len(test_dataframe),

        "market_accuracy": market_metrics["accuracy"],
        "model_accuracy": model_metrics["accuracy"],
        "market_log_loss": market_metrics["log_loss"],
        "model_log_loss": model_metrics["log_loss"],
        "delta_log_loss": model_metrics["log_loss"] - market_metrics["log_loss"],

        "market_brier": market_metrics["brier"],
        "model_brier": model_metrics["brier"],
        "delta_brier": model_metrics["brier"] - market_metrics["brier"],

        "market_ece": market_metrics["ece"],
        "model_ece": model_metrics["ece"],
        "delta_ece": model_metrics["ece"] - market_metrics["ece"],
    }

    predictions_dataframe = test_dataframe.copy()
    predictions_dataframe["selected_feature_set"] = selected_feature_set
    predictions_dataframe["selected_c_value"] = selected_c_value
    predictions_dataframe["selected_alpha"] = selected_alpha
    predictions_dataframe["model_probability_over"] = model_probabilities
    predictions_dataframe["model_probability_under"] = 1.0 - predictions_dataframe["model_probability_over"]
    predictions_dataframe["market_probability_under"] = 1.0 - predictions_dataframe["market_probability_over"]

    return result, predictions_dataframe


def weighted_average(dataframe, value_column, weight_column):
    total_weight = dataframe[weight_column].sum()

    if total_weight == 0:
        return None

    return (dataframe[value_column] * dataframe[weight_column]).sum() / total_weight


def print_overall_results(by_year_dataframe):
    total_matches = by_year_dataframe["test_matches"].sum()

    market_log_loss = weighted_average(by_year_dataframe, "market_log_loss", "test_matches")
    model_log_loss = weighted_average(by_year_dataframe, "model_log_loss", "test_matches")

    market_brier = weighted_average(by_year_dataframe, "market_brier", "test_matches")
    model_brier = weighted_average(by_year_dataframe, "model_brier", "test_matches")

    market_ece = weighted_average(by_year_dataframe, "market_ece", "test_matches")
    model_ece = weighted_average(by_year_dataframe, "model_ece", "test_matches")

    print("")
    print("=== NESTED FEATURE SELECTION OVERALL ===")
    print("test_matches:", int(total_matches))
    print("market_log_loss:", round(float(market_log_loss), 6))
    print("model_log_loss:", round(float(model_log_loss), 6))
    print("delta_log_loss:", round(float(model_log_loss - market_log_loss), 6))
    print("market_brier:", round(float(market_brier), 6))
    print("model_brier:", round(float(model_brier), 6))
    print("delta_brier:", round(float(model_brier - market_brier), 6))
    print("market_ece:", round(float(market_ece), 6))
    print("model_ece:", round(float(model_ece), 6))
    print("delta_ece:", round(float(model_ece - market_ece), 6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = MARKET_NAME

    input_path = get_market_features_path(league_code, market_name)
    by_year_output_path = get_nested_by_year_path(league_code, market_name)
    predictions_output_path = get_nested_predictions_path(league_code, market_name)

    dataframe = pd.read_csv(input_path, low_memory=False)
    dataframe = add_market_logit(dataframe)

    dataframe = dataframe.dropna(
        subset=["season_end_year", "target_over_25", "market_probability_over"]
    ).copy()

    dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)
    dataframe["target_over_25"] = dataframe["target_over_25"].astype(int)

    check_feature_columns(dataframe)

    years = sorted(dataframe["season_end_year"].unique().tolist())

    print("Anos disponíveis:", years)
    print("Jogos:", len(dataframe))
    print("Feature sets:", list(FEATURE_SETS.keys()))
    print("C values:", REGULARIZATION_C_VALUES)
    print("Shrinkage alpha values:", SHRINKAGE_ALPHA_VALUES)

    by_year_results = []
    all_predictions = []

    for test_year in years:
        previous_years = [year for year in years if year < test_year]

        if len(previous_years) < args.min_train_years + 1:
            continue

        print("")
        print(f"=== NESTED FEATURE SELECTION BEFORE TEST YEAR {test_year} ===")

        selected_candidate, validation_table = select_candidate_before_test_year(
            dataframe,
            test_year,
            args.min_train_years
        )

        if selected_candidate is None:
            print("Sem validação suficiente.")
            continue

        print(validation_table.head(20).to_string(index=False))
        print("")
        print("Selected candidate:")
        print("feature_set:", selected_candidate["feature_set"])
        print("c_value:", selected_candidate["c_value"])
        print("alpha:", selected_candidate["alpha"])

        test_result, predictions_dataframe = evaluate_test_year(
            dataframe,
            test_year,
            selected_candidate
        )

        print("test delta_log_loss:", round(float(test_result["delta_log_loss"]), 6))
        print("test delta_brier:", round(float(test_result["delta_brier"]), 6))
        print("test delta_ece:", round(float(test_result["delta_ece"]), 6))

        by_year_results.append(test_result)
        all_predictions.append(predictions_dataframe)

    if len(by_year_results) == 0:
        raise ValueError("Não houve anos suficientes para nested feature selection.")

    by_year_dataframe = pd.DataFrame(by_year_results)
    predictions_dataframe = pd.concat(all_predictions, ignore_index=True)

    by_year_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)

    by_year_dataframe.to_csv(by_year_output_path, index=False)
    predictions_dataframe.to_csv(predictions_output_path, index=False)

    print("")
    print("=== NESTED FEATURE SELECTION BY YEAR ===")
    print(by_year_dataframe.to_string(index=False))

    print_overall_results(by_year_dataframe)

    print("")
    print("Selected alpha counts:")
    print(by_year_dataframe["selected_alpha"].value_counts().sort_index().to_string())

    print("")
    print("Selected feature set counts:")
    print(by_year_dataframe["selected_feature_set"].value_counts().to_string())

    print("")
    print("Ficheiros guardados:")
    print(by_year_output_path)
    print(predictions_output_path)


if __name__ == "__main__":
    main()
