# Research status

## Premier League Asian Handicap candidate

A rule-based Premier League Asian Handicap candidate remains the clearest paper-only strategy in the snapshot:

- away Asian Handicap when the home line is at or below -1.25;
- 261 nested out-of-sample bets;
- +25.525 units;
- +9.78% ROI;
- z-score 1.79;
- positive test years from 2022 through 2025;
- a smaller closing-line sample also remained positive.

This is a historical research candidate, not a confirmed live edge.

## V3 machine-learning family

The exact V3 reproduction produced:

- 12,760 OOS prediction rows;
- 989 historical bets;
- +5.68% ROI;
- z-score 1.82.

The later clean 2025/26 validation was much weaker:

- 3,026 prediction rows;
- 248 bets;
- +0.52% ROI;
- z-score 0.084.

The correct interpretation is therefore neutral/inconclusive rather than “historically profitable model confirmed”.

## V3 Next

The challenger improved market log loss and Brier by small amounts on 15,786 predictions, but the associated value rule produced only 1.13% ROI with z-score 0.45 and a bootstrap interval that crossed zero.

Frozen decision: `v3_next_predictive_gain_no_value`.

## V4 and V5

V4 ended with:

`v4_no_predictive_or_price_movement_signal`

V5 Betfair BASIC price-path research ended with:

`v5_betfair_no_price_movement_signal`

For V5, the selected model's MAE was slightly worse than the no-movement baseline and the bootstrap probability of improvement was 0.020.

## External feature blocks

Controlled Transfermarkt, ClubElo and Understat audits were rejected when they failed the predeclared predictive-improvement requirement. Those negative results remain visible because they constrain future research.

## Current classification

The repository is research-only. Paper tracking is permitted for frozen candidates; live execution and persistent-edge claims are not.
