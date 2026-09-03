# Feature Block Contract

A feature block is a standalone processed table keyed by `canonical_match_id`. It adds one source or feature family to the canonical match universe without building a final modeling table by itself.

## Required Key

Every feature block must include:

- `canonical_match_id`

`canonical_match_id` must be unique unless a report explicitly documents a different grain. For match-level feature blocks, row count should equal the canonical match registry or the documented prototype scope.

## Temporal Safety

Feature values must be available before the match being predicted.

Rules:

- No future source rows.
- No same-day source rows unless timestamp and source documentation prove pre-kickoff availability.
- Rolling features must be computed from previous matches only.
- Median imputers or other preprocessing must be fit inside train folds only during predictive audits.

## Leakage Checks

Each feature block should report:

- row count preservation
- duplicate `canonical_match_id`
- future-date violations
- same-match post-match stat leakage
- rejected/manual-review alias use
- coverage by league and season
- missingness and staleness

## Missingness And Staleness

Feature blocks should retain unmatched rows and provide flags such as:

- found flags
- both-found flags
- history counts
- latest-days-ago fields
- source-max-date staleness flags

Missing rows should not be silently dropped.

## Provenance

Feature blocks may include source provenance columns, but these are audit columns, not model features.

Examples:

- source file path
- alias ID
- latest contributing date
- source league name

## Allowed Feature Categories

- Date-safe ratings.
- Lagged rolling team features.
- Coverage, missingness, and staleness flags.
- Home-minus-away lagged differences.

## Forbidden Feature Categories

- Current fixture target or outcome.
- Same-match score/result or post-match stats.
- Current fixture xG.
- Rejected or manual-review alias joins.
- Source IDs, team names, alias IDs, and source file paths as model features.
- Future data.

## ClubElo Policy

- Use latest ClubElo rating strictly before match date.
- Do not use same-day ratings unless source/timestamp proves they are before kickoff.
- Use locked aliases only.
- Keep missing rows and flags.

## Understat Policy

- Understat match/team stats are post-match data.
- They are usable only as lagged historical features.
- Contributing rows must satisfy `understat_date < match_date`.
- Current fixture xG and current fixture stats are forbidden.
- Rejected aliases are forbidden.
- Manual-review aliases are forbidden.
- Rows after the Understat source max date must remain flagged with staleness warnings.
