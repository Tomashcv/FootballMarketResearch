# V3 Next Research Plan

## What this patch changes

The frozen V3 paper candidate remains unchanged. The patch creates a separate challenger pipeline with strict chronology:

`historical train -> tune year -> calibration/rule year -> untouched test year`

The tune year selects model family, feature block and complexity. The next year selects probability shrinkage/temperature and, separately, a predeclared value rule. Only then is the following year scored.

## Why this is a better optimization path

The current exact V3 is a fixed 35-round XGBoost residual model with 220 features and no explicit post-model calibration. The strongest realistic improvements are therefore not simply “a larger neural network”:

1. **Market anchoring with optional fallback.** Every challenger starts from no-vig market probabilities. When it cannot improve the tune/calibration market, the fold falls back to the market rather than forcing model bets.
2. **Native missing-value trees and stronger regularization.** Transfermarkt and other external blocks have non-random missingness. Native missing branches are compared with the legacy median-imputed setup.
3. **Recency and league balancing.** Old matches and high-volume leagues no longer have to carry equal influence. These choices are tuned before the test year only.
4. **Feature-block ablation.** The search compares all 220 features, external-only residual features, Elo core and a broader context core. This can remove noisy or duplicated market columns.
5. **Probability shrinkage and temperature scaling.** Model probabilities are blended back toward the market on a separate calibration year. This directly attacks the overconfidence that often destroys betting edges.
6. **Conservative nested value selection.** Odds/edge thresholds are selected only on the calibration year from a fixed grid, with minimum volume and league-concentration constraints.
7. **Cluster bootstrap and concentration reporting.** A positive ROI is not promoted merely because one season or league was exceptional.

## Commands

Smoke test:

```bash
python scripts/run_v3_next_research.py --quick
```

Full nested run:

```bash
python scripts/run_v3_next_research.py
```

Audit locally available external data:

```bash
python scripts/audit_v3_next_external_data.py
```

Run unit tests:

```bash
python -m pytest -q tests/test_v3_next_modeling.py
```

## Outputs

The main run writes to `outputs/reports/v3_next_research/`:

- `v3_next_model_selection.csv`
- `v3_next_calibration.csv`
- `v3_next_rule_grid_validation.csv`
- `v3_next_fold_summary.csv`
- `v3_next_test_predictions.csv`
- `v3_next_selected_bets.csv`
- `v3_next_overall_summary.csv`
- `v3_next_by_year.csv`
- `v3_next_by_league.csv`
- `v3_next_report.md`
- `v3_next_decision.md`

## Promotion rule

Do not replace the frozen V3 automatically. A challenger must first:

- improve aggregate out-of-sample log loss and Brier versus the market;
- improve a majority of test folds or at least avoid dependence on one year;
- show positive value after nested rule selection;
- avoid excessive season and league concentration;
- survive a fresh forward season or frozen paper challenger period.

Possible decisions are:

- `v3_next_data_not_ready`
- `v3_next_no_predictive_gain`
- `v3_next_predictive_gain_no_value`
- `v3_next_value_candidate_research_only`
- `v3_next_challenger_for_frozen_forward_test_research_only`

No decision is a confirmed-edge claim.
