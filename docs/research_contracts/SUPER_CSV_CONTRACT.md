# Super CSV Contract

A super CSV is a market-specific research dataset keyed by `canonical_match_id`.

## One Market Per File

Each super CSV should contain one market target and that market's odds/probability columns.

Examples:

- BTTS: `target_btts_yes`, `BTTSY`, `BTTSN`, no-vig BTTS probabilities.
- O/U 2.5: `target_over_2_5`, `O25`, `U25`, no-vig O/U 2.5 probabilities.

Do not mix unrelated market probabilities as model features.

## Paired Odds Requirement

Research-ready market rows must have:

- target present
- both paired odds present
- both paired odds greater than `1`
- `canonical_match_id` present
- no duplicate `canonical_match_id`

Do not impute market odds.

## No-Vig Probabilities

Market no-vig probabilities are allowed as the market baseline and as market-specific probability features. Raw odds are retained for audit/settlement but are forbidden as model features unless explicitly documented otherwise.

## Provenance

Super CSVs should preserve:

- `primary_source`
- `source`
- `source_match_id`
- `source_league_slug`
- canonical competition fields
- `odds_timing_flag`

## Feature Allowlist

Allowed model features are documented per market in report allowlists. Current approved categories include:

- target-specific no-vig market probabilities
- date-safe Footiqo rolling features
- ClubElo locked date-safe features
- Understat locked lagged rolling features
- missingness/staleness/history flags
- league one-hot columns when marked safe

## Forbidden Columns

Forbidden as model features:

- identifiers
- source IDs
- team names
- targets
- raw odds
- same-match scores/results
- same-match post-match stats
- current fixture xG
- unrelated market odds/probabilities
- alias IDs and source file paths

Forbidden columns may remain in files for audit and provenance.

## Research-Ready Levels

- `research_ready/`: Footiqo market CSVs after fixed availability filters.
- `research_ready_plus/clubelo/`: research-ready Footiqo plus locked ClubElo feature block.
- `research_ready_plus/clubelo_understat/`: ClubElo-enhanced files plus locked lagged Understat feature block.

## Research-Only Classification

Footiqo odds timing remains unknown. Therefore all current super CSVs are `research_only`. They are not betting signals and do not imply confirmed edge.
