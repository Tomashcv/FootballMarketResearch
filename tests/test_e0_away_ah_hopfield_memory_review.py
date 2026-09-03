import math

import numpy as np
import pandas as pd

from src.experiments.e0_away_ah_hopfield_memory_review import compute_memory_scores_for_year
from src.experiments.e0_away_ah_hopfield_memory_review import retrieve_memory_value


def test_empty_memory_handled_safely():
    value = retrieve_memory_value(
        np.empty((0, 2)),
        np.array([]),
        np.array([1.0, 0.0]),
        method="hopfield",
        beta=2,
    )

    assert math.isnan(value)


def test_retrieval_deterministic():
    memory_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    memory_values = np.array([0.25, 1.0])
    query_vector = np.array([0.0, 1.0])

    first = retrieve_memory_value(memory_vectors, memory_values, query_vector, method="hopfield", beta=5)
    second = retrieve_memory_value(memory_vectors, memory_values, query_vector, method="hopfield", beta=5)

    assert first == second


def test_beta_changes_output():
    memory_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    memory_values = np.array([0.0, 1.0])
    query_vector = np.array([0.0, 1.0])

    beta_1 = retrieve_memory_value(memory_vectors, memory_values, query_vector, method="hopfield", beta=1)
    beta_10 = retrieve_memory_value(memory_vectors, memory_values, query_vector, method="hopfield", beta=10)

    assert beta_10 != beta_1
    assert beta_10 > beta_1


def test_scaler_fitted_only_on_past_data():
    dataframe = pd.DataFrame(
        {
            "season_end_year": [2020, 2020, 2021],
            "feature_x": [0.0, 2.0, 1000.0],
            "memory_value_profit": [0.0, 1.0, 0.0],
        },
        index=[10, 11, 12],
    )
    variant = {"method": "hopfield", "value_column": "memory_value_profit", "beta": 1}

    _, scaler = compute_memory_scores_for_year(dataframe, 2021, variant, ["feature_x"])

    assert list(scaler["fit_index"]) == [10, 11]
    assert scaler["means"]["feature_x"] == 1.0


def test_no_future_leakage_in_memory():
    dataframe = pd.DataFrame(
        {
            "season_end_year": [2020, 2021, 2022],
            "feature_x": [1.0, 2.0, 3.0],
            "memory_value_profit": [0.1, 0.2, 0.3],
        },
        index=[100, 101, 102],
    )
    variant = {"method": "knn", "value_column": "memory_value_profit", "beta": None}

    scores, scaler = compute_memory_scores_for_year(dataframe, 2021, variant, ["feature_x"])

    assert list(scaler["fit_index"]) == [100]
    assert list(scores.index) == [101]
    assert scores.loc[101] == 0.1
