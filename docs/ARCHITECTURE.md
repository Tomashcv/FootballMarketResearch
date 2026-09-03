# Architecture

## Research layers

Football Market Research is organized as a sequence of evidence-preserving layers.

1. **Source ingestion** — local/provider datasets are treated as immutable evidence.
2. **Canonical identity** — fixture and team identities are normalized before feature joins.
3. **Point-in-time feature blocks** — each source family is transformed independently with date and leakage checks.
4. **Market-specific research tables** — targets, no-vig market probabilities and allowed feature blocks are joined under an explicit contract.
5. **Nested temporal modeling** — model/feature tuning, calibration/rule selection and untouched testing occur in separate time periods.
6. **Robustness and falsification** — concentration, bootstrap, ablation and alternate-model checks determine whether a result is promoted.
7. **Paper-only tracking** — frozen candidates may be monitored with deterministic ledgers; live order placement is outside the system.

## Dependency direction

```text
raw/provider data
      |
      v
canonical match + entity registry
      |
      v
point-in-time feature blocks
      |
      v
market-specific research tables
      |
      v
nested validation + market comparison
      |
      v
robustness / falsification
      |
      v
paper-only monitoring
```

## Identity model

`canonical_match_id` identifies fixtures. `team_id` identifies clubs or national teams independently of league and season.

Provider-facing aliases are reviewed before joins. Fuzzy matching is never treated as an automatic approval mechanism.

## Temporal safety

The core invariant is that every predictive feature must be available before the match being scored.

Examples:

- rolling match features use previous matches only;
- ClubElo lookups are strictly before match date;
- post-match Understat information is used only after lagging into later fixtures;
- model preprocessing is fitted inside training folds;
- the untouched test period cannot influence model, calibration or threshold selection.

## Research vs execution

This repository deliberately ends at paper tracking. It contains no broker/order API and no real-money execution layer.
