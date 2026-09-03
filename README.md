# Football Market Research

A leakage-aware football analytics and machine-learning research platform for testing whether pre-match information adds predictive value beyond betting-market probabilities.

The project combines canonical fixture identity, point-in-time feature engineering, nested temporal validation, market-baseline comparisons, robustness/falsification studies and paper-only tracking. The main objective is not to maximize an in-sample betting return; it is to determine which signals survive chronology, market baselines and clean validation.

## Highlights

- 229 Python research, modeling, data-engineering and test modules in the curated public snapshot.
- Canonical match IDs and explicit source/entity alias registries for cross-provider joins.
- Point-in-time features from market prices, internal Elo, ClubElo, Transfermarkt, Understat and contextual data.
- Strict temporal rules: historical train -> tune -> calibration/rule selection -> untouched test.
- Market probabilities are treated as a baseline that models must beat, not merely as another feature.
- Nested validation, calibration checks, bootstrap uncertainty and concentration diagnostics.
- Explicit falsification studies for neural, sequence, memory and contextual feature families.
- Paper-only pipelines with deterministic IDs, input manifests and settlement logic.
- Negative and inconclusive results are preserved rather than optimized away.
- Full audited source snapshot: **165 tests passed, 2 skipped locally**; the two skips require `pyarrow`, which is installed by CI.

## Research pipeline

<p align="center">
  <img src="docs/assets/research_pipeline.svg" alt="Football Market Research pipeline" width="100%">
</p>

## What the evidence currently says

The repository contains several distinct research lines. Their results should not be collapsed into one “winning model”.

| Research line | Frozen evidence | Current interpretation |
| --- | --- | --- |
| Premier League Asian Handicap rule | 261 nested out-of-sample bets, +25.525 units, +9.78% ROI, z=1.79; all 2022–2025 test years positive. Closing-line sample: 105 bets, +10.14% ROI. | **Paper-trading candidate only.** No real-money or persistent-edge claim. |
| Exact V3 ML reproduction | 12,760 OOS predictions, 989 bets, +5.68% historical ROI, z=1.82. | Historical research signal; not treated as confirmation. |
| Clean 2025/26 V3 validation | 3,026 predictions, 248 bets, +0.52% ROI, z=0.084. | **Neutral / inconclusive.** The historical return did not reproduce convincingly. |
| V3 Next challenger | 15,786 predictions; slightly better log loss and Brier than market, but 1.13% ROI, z=0.45 and a wide bootstrap interval crossing zero. | Predictive gain did not convert into robust value. |
| V4 price-discovery research | Final decision: `v4_no_predictive_or_price_movement_signal`. | Rejected. |
| V5 Betfair BASIC price-path research | Selected model MAE 0.010508 vs no-movement 0.010489; bootstrap probability of lower MAE 0.020. | **No price-movement signal.** |
| Controlled external feature blocks | Transfermarkt, ClubElo and Understat controlled audits were rejected when they failed to improve both log loss and Brier. | Negative results retained. |

Selected frozen summaries are available in [`results/selected/`](results/selected/).

## Why the market baseline matters

A football model can be statistically interesting while still being economically useless. This project therefore compares predictive models directly with no-vig market probabilities.

For probabilistic markets, a candidate is expected to improve metrics such as:

- log loss;
- Brier score;
- calibration error;

before a value or threshold search is allowed.

This ordering prevents a weak model from generating an apparently profitable rule through repeated threshold searching.

## Temporal validation design

The challenger research uses an explicit chronology:

```text
historical training
        |
        v
model / feature tuning year
        |
        v
calibration + rule-selection year
        |
        v
untouched test year
```

Model family, feature block and complexity are chosen before the test period. Probability calibration and decision-rule selection are also separated from the final test year.

The pipeline reports season and league concentration so that one exceptional slice cannot silently dominate the headline result.

## Data engineering

The project evolved from individual league CSV experiments into a provider-aware research system.

### Canonical identity

`canonical_match_id` is a deterministic fixture key. Team identity is handled separately through locked entity/alias registries. Fuzzy mappings are candidates only until explicitly reviewed.

### Point-in-time feature blocks

Feature blocks are keyed by canonical match ID and must satisfy temporal-safety rules. Examples include:

- market no-vig probabilities and bookmaker disagreement;
- internal Elo;
- ClubElo strictly prior to match date;
- Transfermarkt historical squad/value features;
- lagged Understat information;
- schedule, travel and weather context;
- missingness, staleness and history-count flags.

Same-match outcomes, scores, current-fixture post-match statistics and future source rows are forbidden as model features.

### Source boundaries

Raw provider files are not redistributed in this public portfolio repository. The code retains acquisition, canonicalization, mapping and audit logic, while tests use synthetic fixtures where provider content would otherwise be required.

## Modeling and falsification

The repository contains research across:

- regularized logistic regression;
- XGBoost market-residual models;
- calibrated / shrunk probability challengers;
- advanced tabular neural models;
- sequence models and transformers;
- memory / contextual model experiments;
- bookmaker-disagreement features;
- market-only and feature-block ablations.

An experiment name in the repository does **not** imply success. Several neural, sequence, external-data and price-path hypotheses were deliberately falsified or rejected.

## Paper-only execution boundary

The project includes paper pipelines for frozen candidates, including:

- deterministic paper-bet IDs;
- idempotent ledger updates;
- raw-input snapshot manifests;
- settlement from later local results;
- read-only HTML/report artifacts.

There is no broker integration and no live order placement in this repository.

## Repository structure

```text
src/                    core data, feature, modeling, research and paper modules
scripts/                reproducible research/audit entry points
tests/                  offline test suite using synthetic public fixtures
configs/                frozen candidate and research configuration
config/                 league / market configuration
docs/                   architecture, contracts and research methodology
results/selected/       curated frozen aggregate evidence
data/processed/v4/      derived feature-group schema required by tests
```

## Validation

Python 3.12+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m compileall -q src scripts
python -m pytest -q
```

During publication audit, the reconstructed source snapshot completed:

```text
165 passed
2 skipped
```

The two local skips were optional Parquet tests because `pyarrow` was unavailable in the audit runtime. `pyarrow` is part of `requirements.txt`, so GitHub Actions installs it before running the suite.

## Known research boundary

The strongest historical returns in this repository are **not presented as evidence of a persistent betting edge**.

In particular, the exact V3 historical result was followed by a clean 2025/26 validation with near-zero statistical evidence. The project deliberately keeps both results visible.

A stale 2025-completion audit was excluded from the public release because its decision text contradicted the files it had detected. The authoritative public 2025/26 artifact is the neutral/inconclusive validation under `results/selected/`.

## Data and publication boundary

This portfolio release contains original code, methodology, synthetic test fixtures and selected aggregate research summaries.

It excludes:

- raw football datasets and provider archives;
- local caches and generated full prediction tables;
- model binaries;
- API credentials and `.env` files;
- provider-specific bulk data;
- local machine metadata;
- paper-trading ledgers containing nonessential generated rows.

See [Publication boundary](docs/PUBLICATION_BOUNDARY.md).

## Selected documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Research status](docs/RESEARCH_STATUS.md)
- [Publication boundary](docs/PUBLICATION_BOUNDARY.md)
- [Canonical ID contract](docs/research_contracts/CANONICAL_ID_CONTRACT.md)
- [Entity registry contract](docs/research_contracts/ENTITY_REGISTRY_CONTRACT.md)
- [Feature block contract](docs/research_contracts/FEATURE_BLOCK_CONTRACT.md)
- [Super CSV contract](docs/research_contracts/SUPER_CSV_CONTRACT.md)
- [V3 challenger research plan](docs/research_contracts/V3_NEXT_RESEARCH_PLAN.md)

## License

The software and original documentation in this repository are licensed under the [MIT License](LICENSE).

Third-party datasets, odds, provider content, APIs, trademarks and other third-party materials are not covered by this license and remain subject to their respective terms.

## Disclaimer

Independent research project for software, data-engineering and statistical experimentation. It is not betting advice and is not affiliated with or endorsed by any football league, club, bookmaker or data provider.
