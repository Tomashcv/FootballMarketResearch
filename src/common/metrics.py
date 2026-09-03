import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


def expected_calibration_error(y_true, y_probability, n_bins=10):
    y_true_array = np.asarray(y_true)
    y_probability_array = np.asarray(y_probability)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_index in range(n_bins):
        left_edge = bin_edges[bin_index]
        right_edge = bin_edges[bin_index + 1]

        if bin_index == n_bins - 1:
            in_bin = (y_probability_array >= left_edge) & (y_probability_array <= right_edge)
        else:
            in_bin = (y_probability_array >= left_edge) & (y_probability_array < right_edge)

        count = int(np.sum(in_bin))

        if count == 0:
            continue

        average_confidence = float(np.mean(y_probability_array[in_bin]))
        average_accuracy = float(np.mean(y_true_array[in_bin]))
        bin_weight = count / len(y_true_array)

        ece += bin_weight * abs(average_confidence - average_accuracy)

    return ece


def binary_classification_metrics(y_true, y_probability):
    predictions = []

    for probability in y_probability:
        if probability >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "log_loss": log_loss(y_true, y_probability, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_probability),
        "ece": expected_calibration_error(y_true, y_probability),
    }
