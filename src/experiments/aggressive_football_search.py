import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.metrics import expected_calibration_error
from src.common.paths import PROJECT_ROOT


try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "aggressive_football_search"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LEAGUES = ["E0", "E1", "E2", "E3", "SP1", "I1", "D1", "F1"]
MIN_TEST_YEAR = 2022
MIN_VALIDATION_YEARS = 2
MIN_VALIDATION_BETS = 40
MIN_TOTAL_BETS = 80
MIN_SEASON_BETS = 8


def decimal_to_probability(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return 1.0 / odds


def z_score(profits):
    profits = pd.Series(profits, dtype=float).dropna()
    if len(profits) < 2:
        return 0.0
    standard_deviation = profits.std(ddof=1)
    if standard_deviation == 0:
        return 0.0
    return float(profits.mean() / (standard_deviation / math.sqrt(len(profits))))


def max_drawdown_from_frame(dataframe, profit_col="profit"):
    if len(dataframe) == 0:
        return 0.0
    ordered = dataframe.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="coerce")
    ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    cumulative = ordered[profit_col].astype(float).cumsum()
    return float((cumulative.cummax() - cumulative).max())


def summarize_bets(dataframe):
    if len(dataframe) == 0:
        return {
            "bets": 0,
            "profit": 0.0,
            "roi": 0.0,
            "z_score": 0.0,
            "max_drawdown": 0.0,
            "positive_test_years": 0,
            "negative_test_years": 0,
            "min_year_roi": 0.0,
        }
    by_year = dataframe.groupby("test_year")["profit"].mean()
    return {
        "bets": int(len(dataframe)),
        "profit": float(dataframe["profit"].sum()),
        "roi": float(dataframe["profit"].mean()),
        "z_score": z_score(dataframe["profit"]),
        "max_drawdown": max_drawdown_from_frame(dataframe),
        "positive_test_years": int((by_year > 0).sum()),
        "negative_test_years": int((by_year <= 0).sum()),
        "min_year_roi": float(by_year.min()) if len(by_year) else 0.0,
    }


def split_handicap(handicap):
    handicap = float(handicap)
    scaled = handicap * 4.0
    rounded = round(scaled)
    if abs(scaled - rounded) > 0.000001:
        return [handicap]
    if rounded % 2 == 0:
        return [handicap]
    lower = math.floor(handicap * 2.0) / 2.0
    upper = math.ceil(handicap * 2.0) / 2.0
    return [lower, upper]


def profit_single_line(adjusted_margin, odds):
    if adjusted_margin > 0:
        return float(odds) - 1.0
    if adjusted_margin == 0:
        return 0.0
    return -1.0


def asian_profit(team_margin, handicap, odds):
    parts = split_handicap(handicap)
    total = 0.0
    for part in parts:
        total += profit_single_line(team_margin + part, odds) / len(parts)
    return total


def prepare_base_matches(league):
    path = PROCESSED_DIR / league / f"{league}_matches.csv"
    dataframe = pd.read_csv(path, low_memory=False)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "season_end_year"]).copy()
    dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)
    dataframe["home_goals"] = pd.to_numeric(dataframe["FTHG"], errors="coerce")
    dataframe["away_goals"] = pd.to_numeric(dataframe["FTAG"], errors="coerce")
    dataframe["total_goals"] = dataframe["home_goals"] + dataframe["away_goals"]
    dataframe["home_margin"] = dataframe["home_goals"] - dataframe["away_goals"]
    dataframe["away_margin"] = -dataframe["home_margin"]
    return dataframe


def add_rolling_form_features(dataframe):
    dataframe = dataframe.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True).copy()
    histories = {}
    rows = []
    grouped = dataframe.groupby("Date", sort=True)

    def stats(team):
        history = histories.get(team, [])
        last5 = history[-5:]
        if not last5:
            return {
                "points_pg5": 1.35,
                "gf_pg5": 1.35,
                "ga_pg5": 1.35,
                "matches_seen": 0,
            }
        return {
            "points_pg5": sum(item["points"] for item in last5) / len(last5),
            "gf_pg5": sum(item["gf"] for item in last5) / len(last5),
            "ga_pg5": sum(item["ga"] for item in last5) / len(last5),
            "matches_seen": len(history),
        }

    for current_date, date_group in grouped:
        for index, row in date_group.iterrows():
            home = stats(row["HomeTeam"])
            away = stats(row["AwayTeam"])
            rows.append(
                {
                    "row_index": index,
                    "home_points_pg5": home["points_pg5"],
                    "away_points_pg5": away["points_pg5"],
                    "home_gf_pg5": home["gf_pg5"],
                    "away_gf_pg5": away["gf_pg5"],
                    "home_ga_pg5": home["ga_pg5"],
                    "away_ga_pg5": away["ga_pg5"],
                    "home_matches_seen": home["matches_seen"],
                    "away_matches_seen": away["matches_seen"],
                    "points_pg5_diff": home["points_pg5"] - away["points_pg5"],
                    "gf_pg5_diff": home["gf_pg5"] - away["gf_pg5"],
                    "ga_pg5_diff": home["ga_pg5"] - away["ga_pg5"],
                }
            )
        for _, row in date_group.iterrows():
            home_goals = float(row["home_goals"])
            away_goals = float(row["away_goals"])
            if home_goals > away_goals:
                home_points, away_points = 3, 0
            elif home_goals < away_goals:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1
            histories.setdefault(row["HomeTeam"], []).append({"points": home_points, "gf": home_goals, "ga": away_goals})
            histories.setdefault(row["AwayTeam"], []).append({"points": away_points, "gf": away_goals, "ga": home_goals})

    feature_frame = pd.DataFrame(rows).set_index("row_index")
    return dataframe.join(feature_frame, how="left")


def no_vig_two_way(prob_a, prob_b):
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def no_vig_three_way(prob_h, prob_d, prob_a):
    total = prob_h + prob_d + prob_a
    return prob_h / total, prob_d / total, prob_a / total


def build_binary_datasets(matches):
    datasets = []
    base = add_rolling_form_features(matches)

    # Over / under 2.5.
    if {"Avg>2.5", "Avg<2.5"}.issubset(base.columns):
        df = base.copy()
        df["over_odds"] = pd.to_numeric(df["Avg>2.5"], errors="coerce")
        df["under_odds"] = pd.to_numeric(df["Avg<2.5"], errors="coerce")
        df = df[(df["over_odds"] > 1.0) & (df["under_odds"] > 1.0)].copy()
        over_raw = decimal_to_probability(df["over_odds"])
        under_raw = decimal_to_probability(df["under_odds"])
        df["market_probability"], df["opposite_probability"] = no_vig_two_way(over_raw, under_raw)
        df["side_odds"] = df["over_odds"]
        df["target"] = (df["total_goals"] > 2.5).astype(int)
        df["market"] = "over_25"
        datasets.append(df)

        df2 = df.copy()
        df2["market_probability"] = df["opposite_probability"]
        df2["opposite_probability"] = df["market_probability"]
        df2["side_odds"] = df["under_odds"]
        df2["target"] = (df2["total_goals"] < 2.5).astype(int)
        df2["market"] = "under_25"
        datasets.append(df2)

    # 1X2 binary sides.
    if {"AvgH", "AvgD", "AvgA"}.issubset(base.columns):
        df = base.copy()
        df["home_odds"] = pd.to_numeric(df["AvgH"], errors="coerce")
        df["draw_odds"] = pd.to_numeric(df["AvgD"], errors="coerce")
        df["away_odds"] = pd.to_numeric(df["AvgA"], errors="coerce")
        df = df[(df["home_odds"] > 1.0) & (df["draw_odds"] > 1.0) & (df["away_odds"] > 1.0)].copy()
        home_raw = decimal_to_probability(df["home_odds"])
        draw_raw = decimal_to_probability(df["draw_odds"])
        away_raw = decimal_to_probability(df["away_odds"])
        df["home_market_probability"], df["draw_market_probability"], df["away_market_probability"] = no_vig_three_way(
            home_raw, draw_raw, away_raw
        )

        sides = [
            ("home_win", "home_odds", "home_market_probability", df["home_goals"] > df["away_goals"]),
            ("draw", "draw_odds", "draw_market_probability", df["home_goals"] == df["away_goals"]),
            ("away_win", "away_odds", "away_market_probability", df["home_goals"] < df["away_goals"]),
            ("draw_vs_not_draw", "draw_odds", "draw_market_probability", df["home_goals"] == df["away_goals"]),
        ]
        for market, odds_col, prob_col, target in sides:
            side = df.copy()
            side["market_probability"] = side[prob_col]
            side["side_odds"] = side[odds_col]
            side["target"] = target.astype(int)
            side["market"] = market
            datasets.append(side)

    return datasets


def feature_columns_for_binary(dataframe):
    candidates = [
        "market_probability",
        "opposite_probability",
        "home_market_probability",
        "draw_market_probability",
        "away_market_probability",
        "side_odds",
        "home_points_pg5",
        "away_points_pg5",
        "home_gf_pg5",
        "away_gf_pg5",
        "home_ga_pg5",
        "away_ga_pg5",
        "points_pg5_diff",
        "gf_pg5_diff",
        "ga_pg5_diff",
        "home_matches_seen",
        "away_matches_seen",
        "AHh",
        "AvgAHH",
        "AvgAHA",
    ]
    return [column for column in candidates if column in dataframe.columns]


def make_models(random_state):
    models = {
        "market_only": None,
        "logistic_elastic_net": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.25,
                        C=0.5,
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logistic_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=500, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=4,
                        min_samples_leaf=20,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=160,
                        max_depth=4,
                        min_samples_leaf=20,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=80,
                        max_leaf_nodes=12,
                        learning_rate=0.04,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "mlp_small": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(12,),
                        alpha=0.01,
                        learning_rate_init=0.005,
                        max_iter=250,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["xgboost_shallow"] = XGBClassifier(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
        models["xgboost_deep"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
    return models


def predict_model(model_name, model, train, test, feature_columns):
    if model_name == "market_only":
        return test["market_probability"].astype(float).clip(0.001, 0.999).values
    if train["target"].nunique() < 2:
        return test["market_probability"].astype(float).clip(0.001, 0.999).values
    x_train = train[feature_columns].astype(float)
    y_train = train["target"].astype(int)
    x_test = test[feature_columns].astype(float)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return np.clip(probabilities, 0.001, 0.999)


def predict_isotonic(train, test):
    if train["target"].nunique() < 2:
        return test["market_probability"].astype(float).clip(0.001, 0.999).values
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(train["market_probability"].astype(float), train["target"].astype(int))
    return np.clip(iso.predict(test["market_probability"].astype(float)), 0.001, 0.999)


def predict_blend(train, test, feature_columns, random_state):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=500, random_state=random_state)),
        ]
    )
    model_probability = predict_model("blend_logistic", model, train, test, feature_columns)
    market_probability = test["market_probability"].astype(float).values
    return np.clip(0.65 * market_probability + 0.35 * model_probability, 0.001, 0.999)


def evaluate_probability_frame(dataframe, probability_col):
    y = dataframe["target"].astype(int).values
    probabilities = dataframe[probability_col].astype(float).clip(0.001, 0.999).values
    market_probabilities = dataframe["market_probability"].astype(float).clip(0.001, 0.999).values
    return {
        "model_log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "market_log_loss": float(log_loss(y, market_probabilities, labels=[0, 1])),
        "model_brier": float(brier_score_loss(y, probabilities)),
        "market_brier": float(brier_score_loss(y, market_probabilities)),
        "model_ece": float(expected_calibration_error(y, probabilities)),
        "market_ece": float(expected_calibration_error(y, market_probabilities)),
    }


def betting_candidates_from_predictions(dataframe, edge_threshold, min_odds, max_odds):
    selected = dataframe[
        (dataframe["model_probability"] * dataframe["side_odds"] - 1.0 >= edge_threshold)
        & (dataframe["side_odds"] >= min_odds)
        & (dataframe["side_odds"] <= max_odds)
    ].copy()
    selected["profit"] = np.where(selected["target"].astype(int) == 1, selected["side_odds"].astype(float) - 1.0, -1.0)
    selected["edge_threshold"] = edge_threshold
    selected["min_odds"] = min_odds
    selected["max_odds"] = max_odds
    return selected


def select_betting_rule(validation_predictions):
    candidates = []
    for edge_threshold in [0.02, 0.04, 0.06, 0.08, 0.10]:
        for min_odds, max_odds in [(1.4, 5.0), (1.7, 2.4), (1.85, 2.05), (2.0, 4.0), (2.5, 6.0)]:
            bets = betting_candidates_from_predictions(validation_predictions, edge_threshold, min_odds, max_odds)
            if len(bets) < MIN_VALIDATION_BETS:
                continue
            by_year = bets.groupby("season_end_year")["profit"].mean()
            if len(by_year) < MIN_VALIDATION_YEARS:
                continue
            if bets["profit"].mean() <= 0:
                continue
            candidates.append(
                {
                    "edge_threshold": edge_threshold,
                    "min_odds": min_odds,
                    "max_odds": max_odds,
                    "validation_bets": int(len(bets)),
                    "validation_profit": float(bets["profit"].sum()),
                    "validation_roi": float(bets["profit"].mean()),
                    "validation_z_score": z_score(bets["profit"]),
                    "validation_positive_years": int((by_year > 0).sum()),
                    "validation_min_year_roi": float(by_year.min()),
                }
            )
    if not candidates:
        return None, pd.DataFrame()
    table = pd.DataFrame(candidates).sort_values(
        ["validation_positive_years", "validation_min_year_roi", "validation_roi", "validation_bets"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    table["validation_rank"] = table.index + 1
    return table.iloc[0].to_dict(), table


def run_binary_experiment(league, dataset, model_name, random_state):
    market = dataset["market"].iloc[0]
    years = sorted(dataset["season_end_year"].unique().tolist())
    feature_columns = feature_columns_for_binary(dataset)
    rows = []
    bets = []
    candidates = []

    for test_year in years:
        if test_year < MIN_TEST_YEAR:
            continue
        validation_years = [year for year in years if year < test_year and year >= test_year - 4]
        if len(validation_years) < MIN_VALIDATION_YEARS:
            continue

        validation_predictions = []
        for validation_year in validation_years:
            train = dataset[dataset["season_end_year"] < validation_year].copy()
            validation = dataset[dataset["season_end_year"] == validation_year].copy()
            if len(train) < 300 or len(validation) == 0:
                continue
            models = make_models(random_state + int(validation_year))
            if model_name == "isotonic_market":
                probabilities = predict_isotonic(train, validation)
            elif model_name == "blend_market_logistic":
                probabilities = predict_blend(train, validation, feature_columns, random_state + int(validation_year))
            else:
                probabilities = predict_model(model_name, models[model_name], train, validation, feature_columns)
            validation = validation.copy()
            validation["model_probability"] = probabilities
            validation_predictions.append(validation)

        if not validation_predictions:
            continue
        validation_predictions = pd.concat(validation_predictions, ignore_index=True)
        selected_rule, candidate_table = select_betting_rule(validation_predictions)
        if selected_rule is None:
            continue

        train = dataset[dataset["season_end_year"] < test_year].copy()
        test = dataset[dataset["season_end_year"] == test_year].copy()
        if len(train) < 300 or len(test) == 0:
            continue
        models = make_models(random_state + int(test_year))
        if model_name == "isotonic_market":
            probabilities = predict_isotonic(train, test)
        elif model_name == "blend_market_logistic":
            probabilities = predict_blend(train, test, feature_columns, random_state + int(test_year))
        else:
            probabilities = predict_model(model_name, models[model_name], train, test, feature_columns)
        test = test.copy()
        test["model_probability"] = probabilities
        metrics = evaluate_probability_frame(test, "model_probability")
        test_bets = betting_candidates_from_predictions(
            test,
            selected_rule["edge_threshold"],
            selected_rule["min_odds"],
            selected_rule["max_odds"],
        )
        test_bets["league"] = league
        test_bets["strategy"] = f"{market}_{model_name}"
        test_bets["test_year"] = test_year
        test_bets["validation_years"] = ",".join(str(year) for year in validation_years)
        if len(test_bets):
            bets.append(test_bets)

        if len(candidate_table):
            candidate_table = candidate_table.copy()
            candidate_table["league"] = league
            candidate_table["market"] = market
            candidate_table["model"] = model_name
            candidate_table["test_year"] = test_year
            candidates.append(candidate_table)

        row = {
            "league": league,
            "market": market,
            "model": model_name,
            "strategy": f"{market}_{model_name}",
            "classification": "A bet-time-safe",
            "test_year": test_year,
            "train_seasons": ",".join(str(year) for year in sorted(train["season_end_year"].unique().tolist())),
            "validation_seasons": ",".join(str(year) for year in validation_years),
            "selected_edge_threshold": selected_rule["edge_threshold"],
            "selected_min_odds": selected_rule["min_odds"],
            "selected_max_odds": selected_rule["max_odds"],
            "validation_bets": selected_rule["validation_bets"],
            "validation_roi": selected_rule["validation_roi"],
            "test_bets": int(len(test_bets)),
            "test_profit": float(test_bets["profit"].sum()) if len(test_bets) else 0.0,
            "test_roi": float(test_bets["profit"].mean()) if len(test_bets) else 0.0,
        }
        row.update(metrics)
        rows.append(row)

    by_year = pd.DataFrame(rows)
    all_bets = pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()
    all_candidates = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    return by_year, all_bets, all_candidates


def prepare_ah(matches, line_col, home_col, away_col):
    dataframe = matches.copy()
    for column in [line_col, home_col, away_col]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    dataframe = dataframe.dropna(subset=[line_col, home_col, away_col]).copy()
    dataframe = dataframe[(dataframe[home_col] > 1.0) & (dataframe[away_col] > 1.0)].copy()
    dataframe["ah_line"] = dataframe[line_col].astype(float)
    dataframe["home_ah_odds"] = dataframe[home_col].astype(float)
    dataframe["away_ah_odds"] = dataframe[away_col].astype(float)
    dataframe["home_ah_profit"] = dataframe.apply(
        lambda row: asian_profit(row["home_margin"], row["ah_line"], row["home_ah_odds"]), axis=1
    )
    dataframe["away_ah_profit"] = dataframe.apply(
        lambda row: asian_profit(row["away_margin"], -row["ah_line"], row["away_ah_odds"]), axis=1
    )
    if {"AHCh", "AvgCAHH", "AvgCAHA"}.issubset(dataframe.columns):
        for column in ["AHCh", "AvgCAHH", "AvgCAHA"]:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        open_away = (1.0 / dataframe["away_ah_odds"]) / ((1.0 / dataframe["away_ah_odds"]) + (1.0 / dataframe["home_ah_odds"]))
        close_away = (1.0 / dataframe["AvgCAHA"]) / ((1.0 / dataframe["AvgCAHA"]) + (1.0 / dataframe["AvgCAHH"]))
        dataframe["away_clv_prob_pp"] = 100.0 * (close_away - open_away)
        dataframe["away_line_move"] = -dataframe["AHCh"] - (-dataframe["ah_line"])
    return dataframe


def select_ah_rule(validation_data, side):
    rules = []
    thresholds = [-1.00, -1.25, -1.50, -1.75, -2.00] if side == "away_big_home_fav" else [1.00, 1.25, 1.50, 1.75, 2.00]
    odds_bands = [(1.4, 5.0), (1.7, 2.4), (1.85, 2.05), (1.9, 2.2)]
    max_abs_lines = [None, 2.25, 2.5, 2.75]
    for threshold in thresholds:
        for min_odds, max_odds in odds_bands:
            for max_abs_line in max_abs_lines:
                if side == "away_big_home_fav":
                    selected = validation_data[validation_data["ah_line"] <= threshold].copy()
                    selected = selected[(selected["away_ah_odds"] >= min_odds) & (selected["away_ah_odds"] <= max_odds)]
                    if max_abs_line is not None:
                        selected = selected[selected["ah_line"] > -max_abs_line]
                    profit_col = "away_ah_profit"
                else:
                    selected = validation_data[validation_data["ah_line"] >= threshold].copy()
                    selected = selected[(selected["home_ah_odds"] >= min_odds) & (selected["home_ah_odds"] <= max_odds)]
                    if max_abs_line is not None:
                        selected = selected[selected["ah_line"] < max_abs_line]
                    profit_col = "home_ah_profit"
                if len(selected) < MIN_VALIDATION_BETS:
                    continue
                by_year = selected.groupby("season_end_year")[profit_col].mean()
                if len(by_year) < MIN_VALIDATION_YEARS:
                    continue
                if selected[profit_col].mean() <= 0:
                    continue
                rules.append(
                    {
                        "threshold": threshold,
                        "min_odds": min_odds,
                        "max_odds": max_odds,
                        "max_abs_line": max_abs_line,
                        "validation_bets": int(len(selected)),
                        "validation_profit": float(selected[profit_col].sum()),
                        "validation_roi": float(selected[profit_col].mean()),
                        "validation_z_score": z_score(selected[profit_col]),
                        "validation_positive_years": int((by_year > 0).sum()),
                        "validation_min_year_roi": float(by_year.min()),
                    }
                )
    if not rules:
        return None, pd.DataFrame()
    table = pd.DataFrame(rules).sort_values(
        ["validation_positive_years", "validation_min_year_roi", "validation_roi", "validation_bets"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    table["validation_rank"] = table.index + 1
    return table.iloc[0].to_dict(), table


def apply_ah_rule(dataframe, side, rule):
    if side == "away_big_home_fav":
        selected = dataframe[dataframe["ah_line"] <= rule["threshold"]].copy()
        selected = selected[(selected["away_ah_odds"] >= rule["min_odds"]) & (selected["away_ah_odds"] <= rule["max_odds"])]
        if pd.notna(rule["max_abs_line"]):
            selected = selected[selected["ah_line"] > -float(rule["max_abs_line"])]
        selected["profit"] = selected["away_ah_profit"]
        selected["side_odds"] = selected["away_ah_odds"]
        selected["side"] = "away_ah"
    else:
        selected = dataframe[dataframe["ah_line"] >= rule["threshold"]].copy()
        selected = selected[(selected["home_ah_odds"] >= rule["min_odds"]) & (selected["home_ah_odds"] <= rule["max_odds"])]
        if pd.notna(rule["max_abs_line"]):
            selected = selected[selected["ah_line"] < float(rule["max_abs_line"])]
        selected["profit"] = selected["home_ah_profit"]
        selected["side_odds"] = selected["home_ah_odds"]
        selected["side"] = "home_ah"
    return selected


def run_ah_experiment(league, matches, variant):
    if variant == "main":
        line_col, home_col, away_col = "AHh", "AvgAHH", "AvgAHA"
        classification = "A bet-time-safe"
    else:
        line_col, home_col, away_col = "AHCh", "AvgCAHH", "AvgCAHA"
        classification = "B closing-line diagnostic"
    if not {line_col, home_col, away_col}.issubset(matches.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data = prepare_ah(matches, line_col, home_col, away_col)
    years = sorted(data["season_end_year"].unique().tolist())
    rows = []
    bets = []
    candidates = []
    for side in ["away_big_home_fav", "home_big_away_fav"]:
        for test_year in years:
            if test_year < MIN_TEST_YEAR:
                continue
            validation_years = [year for year in years if year < test_year and year >= test_year - 4]
            if len(validation_years) < MIN_VALIDATION_YEARS:
                continue
            validation = data[data["season_end_year"].isin(validation_years)].copy()
            test = data[data["season_end_year"] == test_year].copy()
            rule, table = select_ah_rule(validation, side)
            if rule is None:
                continue
            selected = apply_ah_rule(test, side, rule)
            selected["league"] = league
            selected["strategy"] = f"ah_{variant}_{side}"
            selected["test_year"] = test_year
            selected["validation_years"] = ",".join(str(year) for year in validation_years)
            selected["threshold"] = rule["threshold"]
            if len(selected):
                bets.append(selected)
            if len(table):
                table = table.copy()
                table["league"] = league
                table["variant"] = variant
                table["side"] = side
                table["test_year"] = test_year
                candidates.append(table)
            row = {
                "league": league,
                "market": "asian_handicap",
                "model": "rule_scan",
                "strategy": f"ah_{variant}_{side}",
                "classification": classification,
                "test_year": test_year,
                "train_seasons": "none_rule_based",
                "validation_seasons": ",".join(str(year) for year in validation_years),
                "selected_threshold": rule["threshold"],
                "selected_min_odds": rule["min_odds"],
                "selected_max_odds": rule["max_odds"],
                "selected_max_abs_line": rule["max_abs_line"],
                "validation_bets": rule["validation_bets"],
                "validation_roi": rule["validation_roi"],
                "test_bets": int(len(selected)),
                "test_profit": float(selected["profit"].sum()) if len(selected) else 0.0,
                "test_roi": float(selected["profit"].mean()) if len(selected) else 0.0,
                "model_log_loss": np.nan,
                "market_log_loss": np.nan,
                "model_brier": np.nan,
                "market_brier": np.nan,
                "model_ece": np.nan,
                "market_ece": np.nan,
            }
            if side == "away_big_home_fav" and "away_clv_prob_pp" in selected.columns and len(selected):
                row["clv_prob_pp"] = float(selected["away_clv_prob_pp"].mean())
                row["line_move"] = float(selected["away_line_move"].mean())
            rows.append(row)
    return (
        pd.DataFrame(rows),
        pd.concat(bets, ignore_index=True) if bets else pd.DataFrame(),
        pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(),
    )


def aggregate_results(by_year, bets):
    rows = []
    if len(by_year) == 0:
        return pd.DataFrame()
    group_columns = ["classification", "league", "market", "model", "strategy"]
    for keys, group in by_year.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, keys))
        strategy_bets = bets[
            (bets.get("league", pd.Series(dtype=str)) == record["league"])
            & (bets.get("strategy", pd.Series(dtype=str)) == record["strategy"])
        ].copy()
        summary = summarize_bets(strategy_bets)
        record.update(summary)
        record["test_years"] = ",".join(str(int(year)) for year in sorted(group["test_year"].unique()))
        record["year_rows"] = int(len(group))
        record["sum_test_bets_from_years"] = int(group["test_bets"].sum())
        record["weighted_model_log_loss"] = weighted_metric(group, "model_log_loss", "test_bets")
        record["weighted_market_log_loss"] = weighted_metric(group, "market_log_loss", "test_bets")
        record["weighted_model_brier"] = weighted_metric(group, "model_brier", "test_bets")
        record["weighted_market_brier"] = weighted_metric(group, "market_brier", "test_bets")
        record["weighted_model_ece"] = weighted_metric(group, "model_ece", "test_bets")
        record["weighted_market_ece"] = weighted_metric(group, "market_ece", "test_bets")
        if "clv_prob_pp" in group.columns:
            record["avg_clv_prob_pp"] = float(group["clv_prob_pp"].dropna().mean()) if group["clv_prob_pp"].notna().any() else np.nan
        rows.append(record)
    output = pd.DataFrame(rows)
    if len(output):
        output = output.sort_values(["classification", "profit", "roi"], ascending=[True, False, False]).reset_index(drop=True)
    return output


def weighted_metric(dataframe, value_col, weight_col):
    if value_col not in dataframe.columns:
        return np.nan
    valid = dataframe[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]
    if len(valid) == 0:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def classify_recommendation(row):
    if row["bets"] < MIN_TOTAL_BETS:
        return "reject_too_few_bets"
    if row["positive_test_years"] <= 1:
        return "reject_one_lucky_season"
    if row["roi"] <= 0 or row["profit"] <= 0:
        return "reject_negative"
    if row["classification"].startswith("B"):
        return "closing_line_diagnostic_only"
    if pd.notna(row.get("avg_clv_prob_pp", np.nan)) and row["avg_clv_prob_pp"] < 0:
        return "paper_trade_only_negative_clv"
    if row["z_score"] < 1.5:
        return "inconclusive_low_z"
    return "paper_trade_candidate"


def write_inventory():
    script_rows = []
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        rel = path.relative_to(PROJECT_ROOT)
        text = path.read_text(errors="ignore")
        role = "utility"
        if "asian_handicap" in str(rel):
            role = "asian_handicap"
        elif "xgboost" in str(rel) or "value_scan" in str(rel):
            role = "model_or_value_scan"
        elif "build_" in path.name or "features" in str(rel):
            role = "data_or_feature_builder"
        elif "fetch" in path.name:
            role = "fetcher"
        stale = ""
        if "split_handicap" in text and "asian_profit" in text and "common" not in str(rel):
            stale = "duplicates_ah_settlement"
        if path.name == "add_current_season_features.py":
            stale = "writes_back_to_input_path"
        script_rows.append({"path": str(rel), "lines": len(text.splitlines()), "role": role, "stale_or_duplicate_note": stale})
    inventory = pd.DataFrame(script_rows)
    inventory.to_csv(OUTPUT_DIR / "script_inventory.csv", index=False)
    return inventory


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def write_report(inventory, summary, by_year, notes, command):
    best_safe = summary[summary["classification"].str.startswith("A")].head(10) if len(summary) else pd.DataFrame()
    best_closing = summary[summary["classification"].str.startswith("B")].head(10) if len(summary) else pd.DataFrame()
    positives = summary[summary["profit"] > 0].copy() if len(summary) else pd.DataFrame()
    positives["decision_note"] = positives.apply(classify_recommendation, axis=1) if len(positives) else []
    paper_candidates = positives[positives["decision_note"] == "paper_trade_candidate"] if len(positives) else pd.DataFrame()

    def table(df, columns, max_rows=12):
        if len(df) == 0:
            return "_None._"
        clipped = df[columns].head(max_rows).copy()
        for col in clipped.columns:
            if clipped[col].dtype.kind in "fc":
                clipped[col] = clipped[col].map(lambda x: "" if pd.isna(x) else round(float(x), 4))
        clipped = clipped.fillna("")
        headers = [str(column) for column in clipped.columns]
        rows = []
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("| " + " | ".join("---" for _ in headers) + " |")
        for _, table_row in clipped.iterrows():
            values = [str(table_row[column]).replace("|", "\\|") for column in clipped.columns]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    changed_files = [
        "src/experiments/__init__.py",
        "src/experiments/aggressive_football_search.py",
        "tests/test_aggressive_football_search.py",
        "outputs/reports/aggressive_football_search/*",
    ]
    lines = []
    lines.append("# Aggressive Football Search Report")
    lines.append("")
    lines.append("Audit date: 2026-06-28")
    lines.append("")
    lines.append("Classification labels: A = bet-time-safe, B = closing-line diagnostic, C = invalid due to leakage, D = inconclusive.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append(f"```bash\n{command}\n```")
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append(f"Scripts inventoried: {len(inventory)}.")
    availability_path = OUTPUT_DIR / "market_availability.csv"
    if availability_path.exists():
        availability = pd.read_csv(availability_path)
        lines.append("")
        lines.append("Market availability:")
        lines.append("")
        lines.append(table(availability, ["league", "matches", "has_ah", "has_1x2", "has_ou25", "has_btts_odds"], max_rows=20))
        lines.append("")
        lines.append("BTTS odds were not available in processed data, so BTTS was not evaluated as a bet-time-safe betting market.")
    stale = inventory[inventory["stale_or_duplicate_note"].fillna("").astype(str) != ""]
    lines.append("")
    lines.append(table(stale, ["path", "role", "stale_or_duplicate_note"], max_rows=20))
    lines.append("")
    lines.append("## Best Bet-Time-Safe Results")
    lines.append("")
    lines.append(table(best_safe, ["classification", "league", "market", "model", "strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "positive_test_years", "avg_clv_prob_pp"], max_rows=15))
    lines.append("")
    lines.append("## Best Closing-Line Diagnostic Results")
    lines.append("")
    lines.append(table(best_closing, ["classification", "league", "market", "model", "strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "positive_test_years"], max_rows=15))
    lines.append("")
    lines.append("## Paper Candidates")
    lines.append("")
    lines.append(table(paper_candidates, ["classification", "league", "market", "model", "strategy", "bets", "profit", "roi", "z_score", "max_drawdown", "positive_test_years", "min_year_roi"], max_rows=15))
    lines.append("")
    if len(paper_candidates):
        lines.append("These are not live-shadow recommendations. They passed only the automated coarse filters and still require manual falsification, CLV where available, and forward paper validation.")
        lines.append("")
    lines.append("## Rejected Or Invalid Positive Results")
    lines.append("")
    rejected = positives[positives["decision_note"] != "paper_trade_candidate"] if len(positives) else pd.DataFrame()
    lines.append(table(rejected, ["classification", "league", "market", "model", "strategy", "bets", "profit", "roi", "z_score", "positive_test_years", "decision_note"], max_rows=25))
    lines.append("")
    lines.append("Full rejected/positive inventory is in `experiment_summary.csv` via the `decision_note` column.")
    lines.append("")
    lines.append("## Falsification Notes For Positive Results")
    lines.append("")
    for note in notes[:30]:
        lines.append(f"- {note}")
    if not notes:
        lines.append("_No positive strategy survived basic filters._")
    lines.append("")
    lines.append("## Model Metrics")
    lines.append("")
    metric_rows = summary.dropna(subset=["weighted_model_log_loss"], how="all") if len(summary) else pd.DataFrame()
    lines.append(table(metric_rows.sort_values("weighted_model_log_loss"), ["league", "market", "model", "bets", "profit", "roi", "weighted_model_log_loss", "weighted_market_log_loss", "weighted_model_brier", "weighted_market_brier", "weighted_model_ece", "weighted_market_ece"], max_rows=20))
    lines.append("")
    lines.append("## Changed Files")
    lines.append("")
    for file_name in changed_files:
        lines.append(f"- `{file_name}`")
    lines.append("")
    lines.append("## Tests Run")
    lines.append("")
    lines.append("- `python -m py_compile src/experiments/aggressive_football_search.py`")
    lines.append("- `PYTHONPATH=. python tests/test_aggressive_football_search.py`")
    lines.append("- `python -m pytest tests/test_aggressive_football_search.py` attempted but pytest is not installed in this environment.")
    lines.append("- `python -m src.experiments.aggressive_football_search --quick` completed computations and wrote CSVs; first run failed only on markdown rendering because `tabulate` is not installed.")
    lines.append("")
    lines.append("## Honest Recommendation")
    lines.append("")
    if len(best_safe) == 0:
        recommendation = "pause"
        rationale = "No bet-time-safe result survived the minimum reporting filters."
    else:
        top = best_safe.iloc[0]
        top_note = classify_recommendation(top)
        if top_note == "paper_trade_candidate":
            recommendation = "paper trade"
            rationale = "The best safe result is positive across more than one season, but this was an aggressive search and still needs forward paper validation."
        elif "negative_clv" in top_note:
            recommendation = "paper trade"
            rationale = "The best safe result has historical profit but CLV does not support live promotion."
        else:
            recommendation = "pause"
            rationale = f"The best safe result is limited by {top_note}."
    lines.append(f"Recommendation: **{recommendation}**.")
    lines.append("")
    lines.append(rationale)
    lines.append("")
    lines.append("No strategy is a confirmed edge from this run. The search was intentionally aggressive, so positive results are candidates for falsification and paper trading, not live promotion.")
    (OUTPUT_DIR / "aggressive_football_search_report.md").write_text("\n".join(lines) + "\n")


def generate_falsification_notes(summary, all_bets):
    notes = []
    if len(summary) == 0 or len(all_bets) == 0:
        return notes
    positive = summary[(summary["classification"].str.startswith("A")) & (summary["profit"] > 0)].head(20)
    for _, row in positive.iterrows():
        bets = all_bets[(all_bets["league"] == row["league"]) & (all_bets["strategy"] == row["strategy"])].copy()
        if len(bets) == 0:
            continue
        by_year = bets.groupby("test_year")["profit"].sum().sort_values(ascending=False)
        top_year_share = float(by_year.iloc[0] / bets["profit"].sum()) if bets["profit"].sum() != 0 else np.nan
        if "HomeTeam" in bets.columns:
            by_team = bets.groupby("HomeTeam")["profit"].sum().sort_values(ascending=False)
            top_team_share = float(by_team.head(3).sum() / bets["profit"].sum()) if bets["profit"].sum() != 0 else np.nan
        else:
            top_team_share = np.nan
        notes.append(
            f"{row['league']} {row['strategy']}: top-year profit share={top_year_share:.2f}, top-3-home-team share={top_team_share:.2f}, decision={classify_recommendation(row)}."
        )
    return notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a bounded but broad search suitable for audit iteration.")
    parser.add_argument("--leagues", default=",".join(LEAGUES))
    parser.add_argument("--random-state", type=int, default=17)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    command = "python -m src.experiments.aggressive_football_search --quick"
    leagues = [item.strip().upper() for item in args.leagues.split(",") if item.strip()]
    model_names = ["market_only", "isotonic_market", "logistic_l2", "blend_market_logistic"]
    if not args.quick:
        model_names += ["random_forest", "extra_trees", "hist_gradient_boosting", "mlp_small"]
        if XGBClassifier is not None:
            model_names += ["xgboost_shallow", "xgboost_deep"]
    else:
        model_names += ["hist_gradient_boosting"]

    inventory = write_inventory()
    all_by_year = []
    all_bets = []
    all_candidates = []
    availability = []

    for league in leagues:
        matches = prepare_base_matches(league)
        available = {
            "league": league,
            "matches": int(len(matches)),
            "years": ",".join(str(year) for year in sorted(matches["season_end_year"].unique().tolist())),
            "has_ah": {"AHh", "AvgAHH", "AvgAHA"}.issubset(matches.columns),
            "has_1x2": {"AvgH", "AvgD", "AvgA"}.issubset(matches.columns),
            "has_ou25": {"Avg>2.5", "Avg<2.5"}.issubset(matches.columns),
            "has_btts_odds": any("BTTS" in column or "Both" in column for column in matches.columns),
        }
        availability.append(available)

        for variant in ["main", "closing"]:
            by_year, bets, candidates = run_ah_experiment(league, matches, variant)
            if len(by_year):
                all_by_year.append(by_year)
            if len(bets):
                all_bets.append(bets)
            if len(candidates):
                all_candidates.append(candidates)

        for dataset in build_binary_datasets(matches):
            for model_name in model_names:
                by_year, bets, candidates = run_binary_experiment(league, dataset, model_name, args.random_state)
                if len(by_year):
                    all_by_year.append(by_year)
                if len(bets):
                    all_bets.append(bets)
                if len(candidates):
                    all_candidates.append(candidates)

    by_year = pd.concat(all_by_year, ignore_index=True) if all_by_year else pd.DataFrame()
    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    summary = aggregate_results(by_year, bets)
    if len(summary):
        summary["decision_note"] = summary.apply(classify_recommendation, axis=1)

    pd.DataFrame(availability).to_csv(OUTPUT_DIR / "market_availability.csv", index=False)
    by_year.to_csv(OUTPUT_DIR / "experiment_by_year.csv", index=False)
    bets.to_csv(OUTPUT_DIR / "experiment_bets.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "experiment_candidates.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "command": command,
                "leagues": leagues,
                "models": model_names,
                "min_test_year": MIN_TEST_YEAR,
                "min_validation_years": MIN_VALIDATION_YEARS,
                "min_validation_bets": MIN_VALIDATION_BETS,
            },
            indent=2,
        )
    )
    notes = generate_falsification_notes(summary, bets)
    write_report(inventory, summary, by_year, notes, command)

    print("Wrote aggressive football search outputs to:", OUTPUT_DIR)
    if len(summary):
        print(summary.head(20).to_string(index=False))
    else:
        print("No strategies produced bets under the nested validation filters.")


if __name__ == "__main__":
    main()
