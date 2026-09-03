import argparse
import itertools

import numpy as np
import pandas as pd

from src.common.metrics import binary_classification_metrics
from src.common.odds_utils import probability_to_logit
from src.common.paths import get_market_features_path
from src.common.paths import get_market_predictions_dir

try:
    import xgboost as xgb
except ImportError as error:
    raise ImportError(
        "xgboost não está instalado. Corre: python -m pip install xgboost"
    ) from error


MARKET_NAME = "over_under_25"


ALPHA_VALUES = [0.0, 0.10, 0.20, 0.35, 0.50]
MAX_DEPTH_VALUES = [1, 2]
LEARNING_RATE_VALUES = [0.03]
NUM_BOOST_ROUND_VALUES = [80]
LAMBDA_VALUES = [10.0, 25.0]


FEATURE_SETS = {
    "xgb_home_away_context_5": [
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

        "expected_home_goals_simple_5",
        "expected_away_goals_simple_5",
        "expected_total_goals_simple_5",
        "expected_total_goals_simple_5_minus_2_5",
        "expected_total_goals_simple_5_minus_league_avg",

        "combined_home_away_over_rate_5",
        "combined_home_away_total_goals_5",

        "league_avg_goals_so_far",
        "league_over_25_rate_so_far",
        "market_over_probability_minus_league_over_rate",
        "market_over_probability_minus_0_5",
    ],

    "xgb_home_away_context_10": [
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

        "expected_home_goals_simple_10",
        "expected_away_goals_simple_10",
        "expected_total_goals_simple_10",
        "expected_total_goals_simple_10_minus_2_5",
        "expected_total_goals_simple_10_minus_league_avg",

        "combined_home_away_over_rate_10",
        "combined_home_away_total_goals_10",

        "league_avg_goals_so_far",
        "league_over_25_rate_so_far",
        "market_over_probability_minus_league_over_rate",
        "market_over_probability_minus_0_5",
    ],

    "xgb_full_v2": [
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

        "overall_goals_for_last_5_diff",
        "overall_goals_against_last_5_diff",
        "overall_total_goals_last_5_diff",
        "overall_over_25_rate_last_5_diff",

        "home_away_context_goals_for_last_5_diff",
        "home_away_context_goals_against_last_5_diff",
        "home_away_context_total_goals_last_5_diff",
        "home_away_context_over_25_rate_last_5_diff",

        "expected_home_goals_simple_5",
        "expected_away_goals_simple_5",
        "expected_total_goals_simple_5",
        "expected_total_goals_simple_5_minus_2_5",
        "expected_total_goals_simple_5_minus_league_avg",
        "home_attack_vs_away_defense_5",
        "away_attack_vs_home_defense_5",
        "combined_home_away_over_rate_5",
        "combined_home_away_total_goals_5",

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
        "matches_played_in_league_so_far",

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


def blend_with_market(raw_model_probabilities, market_probabilities, alpha):
    blended = (
        (1.0 - float(alpha)) * market_probabilities
        + float(alpha) * raw_model_probabilities
    )

    blended = np.clip(blended, 0.000001, 0.999999)

    return blended


def make_xgb_params(max_depth, learning_rate, lambda_value):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(max_depth),
        "eta": float(learning_rate),
        "lambda": float(lambda_value),
        "alpha": 0.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 8.0,
        "tree_method": "hist",
        "seed": 42,
    }

    return params


def train_xgboost_base_margin(
    train_dataframe,
    test_dataframe,
    feature_columns,
    max_depth,
    learning_rate,
    num_boost_round,
    lambda_value
):
    x_train = train_dataframe[feature_columns].astype(float)
    y_train = train_dataframe["target_over_25"].astype(int)

    x_test = test_dataframe[feature_columns].astype(float)

    train_base_margin = train_dataframe["market_logit_over"].astype(float).values
    test_base_margin = test_dataframe["market_logit_over"].astype(float).values

    train_matrix = xgb.DMatrix(
        x_train,
        label=y_train,
        base_margin=train_base_margin,
        feature_names=feature_columns
    )

    test_matrix = xgb.DMatrix(
        x_test,
        base_margin=test_base_margin,
        feature_names=feature_columns
    )

    params = make_xgb_params(
        max_depth=max_depth,
        learning_rate=learning_rate,
        lambda_value=lambda_value
    )

    model = xgb.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=int(num_boost_round),
        verbose_eval=False
    )

    raw_probabilities = model.predict(test_matrix)
    raw_probabilities = np.clip(raw_probabilities, 0.000001, 0.999999)

    return raw_probabilities


def predict_candidate(
    train_dataframe,
    test_dataframe,
    feature_set_name,
    max_depth,
    learning_rate,
    num_boost_round,
    lambda_value,
    alpha
):
    market_probabilities = test_dataframe["market_probability_over"].astype(float).values

    if float(alpha) == 0.0:
        return market_probabilities

    feature_columns = FEATURE_SETS[feature_set_name]

    raw_model_probabilities = train_xgboost_base_margin(
        train_dataframe=train_dataframe,
        test_dataframe=test_dataframe,
        feature_columns=feature_columns,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_boost_round=num_boost_round,
        lambda_value=lambda_value
    )

    final_probabilities = blend_with_market(
        raw_model_probabilities,
        market_probabilities,
        alpha
    )

    return final_probabilities


def evaluate_probabilities(dataframe, probabilities):
    y_true = dataframe["target_over_25"].astype(int).values
    market_probabilities = dataframe["market_probability_over"].astype(float).values

    market_metrics = binary_classification_metrics(y_true, market_probabilities)
    model_metrics = binary_classification_metrics(y_true, probabilities)

    result = {
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

    return result


def evaluate_candidate_on_validation(dataframe, candidate, validation_year):
    train_dataframe = dataframe[dataframe["season_end_year"] < validation_year].copy()
    validation_dataframe = dataframe[dataframe["season_end_year"] == validation_year].copy()

    if len(train_dataframe) == 0 or len(validation_dataframe) == 0:
        return None

    if train_dataframe["target_over_25"].nunique() < 2:
        return None

    probabilities = predict_candidate(
        train_dataframe=train_dataframe,
        test_dataframe=validation_dataframe,
        feature_set_name=candidate["feature_set"],
        max_depth=candidate["max_depth"],
        learning_rate=candidate["learning_rate"],
        num_boost_round=candidate["num_boost_round"],
        lambda_value=candidate["lambda_value"],
        alpha=candidate["alpha"]
    )

    result = evaluate_probabilities(validation_dataframe, probabilities)

    result["feature_set"] = candidate["feature_set"]
    result["max_depth"] = candidate["max_depth"]
    result["learning_rate"] = candidate["learning_rate"]
    result["num_boost_round"] = candidate["num_boost_round"]
    result["lambda_value"] = candidate["lambda_value"]
    result["alpha"] = candidate["alpha"]
    result["validation_year"] = validation_year
    result["validation_matches"] = len(validation_dataframe)

    return result


def build_candidates():
    candidates = []

    # Mercado puro. Serve como opção segura.
    candidates.append({
        "feature_set": "market_only",
        "max_depth": 0,
        "learning_rate": 0.0,
        "num_boost_round": 0,
        "lambda_value": 0.0,
        "alpha": 0.0,
    })

    for feature_set_name in FEATURE_SETS:
        for max_depth in MAX_DEPTH_VALUES:
            for learning_rate in LEARNING_RATE_VALUES:
                for num_boost_round in NUM_BOOST_ROUND_VALUES:
                    for lambda_value in LAMBDA_VALUES:
                        for alpha in ALPHA_VALUES:
                            if float(alpha) == 0.0:
                                continue

                            candidates.append({
                                "feature_set": feature_set_name,
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                                "num_boost_round": num_boost_round,
                                "lambda_value": lambda_value,
                                "alpha": alpha,
                            })

    return candidates


def select_candidate_before_test_year(dataframe, test_year, min_train_years, min_validation_improvement):
    years = sorted(dataframe["season_end_year"].dropna().unique().tolist())
    candidates = build_candidates()

    validation_results = []

    for validation_year in years:
        if validation_year >= test_year:
            continue

        train_years = [year for year in years if year < validation_year]

        if len(train_years) < min_train_years:
            continue

        print(f"  validation_year={validation_year}")

        for candidate in candidates:
            result = evaluate_candidate_on_validation(dataframe, candidate, validation_year)

            if result is not None:
                validation_results.append(result)

    if len(validation_results) == 0:
        return None, None

    validation_dataframe = pd.DataFrame(validation_results)

    grouped = validation_dataframe.groupby(
        [
            "feature_set",
            "max_depth",
            "learning_rate",
            "num_boost_round",
            "lambda_value",
            "alpha",
        ],
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

    best_candidate = grouped.iloc[0].to_dict()

    if float(best_candidate["validation_delta_log_loss"]) > -abs(float(min_validation_improvement)):
        selected_candidate = {
            "feature_set": "market_only",
            "max_depth": 0,
            "learning_rate": 0.0,
            "num_boost_round": 0,
            "lambda_value": 0.0,
            "alpha": 0.0,
            "validation_matches": best_candidate["validation_matches"],
            "validation_delta_log_loss": 0.0,
            "validation_delta_brier": 0.0,
            "validation_delta_ece": 0.0,
            "validation_rank": 0,
        }
    else:
        selected_candidate = best_candidate

    return selected_candidate, grouped


def evaluate_test_year(dataframe, test_year, selected_candidate):
    train_dataframe = dataframe[dataframe["season_end_year"] < test_year].copy()
    test_dataframe = dataframe[dataframe["season_end_year"] == test_year].copy()

    probabilities = predict_candidate(
        train_dataframe=train_dataframe,
        test_dataframe=test_dataframe,
        feature_set_name=selected_candidate["feature_set"],
        max_depth=selected_candidate["max_depth"],
        learning_rate=selected_candidate["learning_rate"],
        num_boost_round=selected_candidate["num_boost_round"],
        lambda_value=selected_candidate["lambda_value"],
        alpha=selected_candidate["alpha"]
    )

    result = evaluate_probabilities(test_dataframe, probabilities)

    result["test_year"] = test_year
    result["selected_feature_set"] = selected_candidate["feature_set"]
    result["selected_max_depth"] = selected_candidate["max_depth"]
    result["selected_learning_rate"] = selected_candidate["learning_rate"]
    result["selected_num_boost_round"] = selected_candidate["num_boost_round"]
    result["selected_lambda_value"] = selected_candidate["lambda_value"]
    result["selected_alpha"] = selected_candidate["alpha"]
    result["validation_delta_log_loss"] = selected_candidate["validation_delta_log_loss"]
    result["test_matches"] = len(test_dataframe)

    predictions_dataframe = test_dataframe.copy()
    predictions_dataframe["selected_feature_set"] = selected_candidate["feature_set"]
    predictions_dataframe["selected_max_depth"] = selected_candidate["max_depth"]
    predictions_dataframe["selected_learning_rate"] = selected_candidate["learning_rate"]
    predictions_dataframe["selected_num_boost_round"] = selected_candidate["num_boost_round"]
    predictions_dataframe["selected_lambda_value"] = selected_candidate["lambda_value"]
    predictions_dataframe["selected_alpha"] = selected_candidate["alpha"]
    predictions_dataframe["model_probability_over"] = probabilities
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
    print("=== XGBOOST MARKET-MARGIN OVERALL ===")
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
    parser.add_argument("--min-validation-improvement", type=float, default=0.00025)
    args = parser.parse_args()

    league_code = args.league.upper()
    market_name = MARKET_NAME

    input_path = get_market_features_path(league_code, market_name)
    predictions_dir = get_market_predictions_dir(league_code, market_name)

    by_year_output_path = predictions_dir / "xgboost_market_margin_by_year.csv"
    predictions_output_path = predictions_dir / "xgboost_market_margin_predictions.csv"
    candidates_output_path = predictions_dir / "xgboost_market_margin_candidates_last.csv"

    dataframe = pd.read_csv(input_path, low_memory=False)
    dataframe = add_market_logit(dataframe)

    dataframe = dataframe.dropna(
        subset=["season_end_year", "target_over_25", "market_probability_over", "market_logit_over"]
    ).copy()

    dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)
    dataframe["target_over_25"] = dataframe["target_over_25"].astype(int)

    check_feature_columns(dataframe)

    years = sorted(dataframe["season_end_year"].unique().tolist())

    print("Anos disponíveis:", years)
    print("Jogos:", len(dataframe))
    print("Feature sets:", list(FEATURE_SETS.keys()))
    print("Alpha values:", ALPHA_VALUES)
    print("Max depth values:", MAX_DEPTH_VALUES)
    print("Lambda values:", LAMBDA_VALUES)
    print("Min validation improvement:", args.min_validation_improvement)

    by_year_results = []
    all_predictions = []
    last_validation_table = None

    for test_year in years:
        previous_years = [year for year in years if year < test_year]

        if len(previous_years) < args.min_train_years + 1:
            continue

        print("")
        print(f"=== XGBOOST MARKET-MARGIN BEFORE TEST YEAR {test_year} ===")

        selected_candidate, validation_table = select_candidate_before_test_year(
            dataframe=dataframe,
            test_year=test_year,
            min_train_years=args.min_train_years,
            min_validation_improvement=args.min_validation_improvement
        )

        if selected_candidate is None:
            print("Sem validação suficiente.")
            continue

        last_validation_table = validation_table

        print("")
        print("Top validation candidates:")
        print(validation_table.head(15).to_string(index=False))

        print("")
        print("Selected candidate:")
        print("feature_set:", selected_candidate["feature_set"])
        print("max_depth:", selected_candidate["max_depth"])
        print("learning_rate:", selected_candidate["learning_rate"])
        print("num_boost_round:", selected_candidate["num_boost_round"])
        print("lambda:", selected_candidate["lambda_value"])
        print("alpha:", selected_candidate["alpha"])
        print("validation_delta_log_loss:", selected_candidate["validation_delta_log_loss"])

        test_result, predictions_dataframe = evaluate_test_year(
            dataframe=dataframe,
            test_year=test_year,
            selected_candidate=selected_candidate
        )

        print("test delta_log_loss:", round(float(test_result["delta_log_loss"]), 6))
        print("test delta_brier:", round(float(test_result["delta_brier"]), 6))
        print("test delta_ece:", round(float(test_result["delta_ece"]), 6))

        by_year_results.append(test_result)
        all_predictions.append(predictions_dataframe)

    if len(by_year_results) == 0:
        raise ValueError("Não houve anos suficientes para XGBoost market-margin.")

    by_year_dataframe = pd.DataFrame(by_year_results)
    predictions_dataframe = pd.concat(all_predictions, ignore_index=True)

    predictions_dir.mkdir(parents=True, exist_ok=True)

    by_year_dataframe.to_csv(by_year_output_path, index=False)
    predictions_dataframe.to_csv(predictions_output_path, index=False)

    if last_validation_table is not None:
        last_validation_table.to_csv(candidates_output_path, index=False)

    print("")
    print("=== XGBOOST MARKET-MARGIN BY YEAR ===")
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
    print(candidates_output_path)


if __name__ == "__main__":
    main()
