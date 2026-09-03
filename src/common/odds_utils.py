import math


def decimal_to_implied_probability(decimal_odd):
    if decimal_odd is None:
        return None

    odd = float(decimal_odd)

    if odd <= 1.0:
        return None

    return 1.0 / odd


def normalize_two_way_probabilities(probability_a, probability_b):
    if probability_a is None or probability_b is None:
        return None, None

    total = probability_a + probability_b

    if total <= 0:
        return None, None

    normalized_a = probability_a / total
    normalized_b = probability_b / total

    return normalized_a, normalized_b


def probability_to_logit(probability):
    clipped_probability = min(max(float(probability), 0.000001), 0.999999)
    return math.log(clipped_probability / (1.0 - clipped_probability))


def calculate_ev(model_probability, decimal_odd):
    return float(model_probability) * float(decimal_odd) - 1.0


def calculate_minimum_acceptable_odds(model_probability):
    probability = float(model_probability)

    if probability <= 0:
        return None

    return 1.0 / probability
