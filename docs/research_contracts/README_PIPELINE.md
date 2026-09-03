# Football Data Pipeline

This project is organized around stable match IDs, locked entity aliases, date-safe feature blocks, and market-specific research CSVs. The current approved state is research-only: Footiqo odds timing/source documentation remains unknown, predictive audits did not show robust gain over the market baseline, and no confirmed edge is claimed.

## Data Architecture

- `data/raw/` and `data/raw_external/`: immutable source inputs. Do not edit, move, overwrite, or delete raw files during pipeline rebuilds.
- `data/interim/`: audited intermediate outputs used to bridge raw sources into canonical tables.
- `data/processed/`: canonical registries, feature blocks, and research CSVs.
- `outputs/reports/`: validation reports, leakage checks, audit summaries, and decisions.

The pipeline should always promote data in one direction: raw or raw external inputs to interim audit tables, then processed canonical artifacts, then reports. Raw files are evidence, not workspace scratch files.

## Canonical Match IDs

Matches are keyed by `canonical_match_id`, an int64-compatible ID with format `TLLLYYYYGGGG`.

- `T`: competition type.
- `LLL`: competition code within that type.
- `YYYY`: season start year or tournament year.
- `GGGG`: deterministic match sequence within competition and season/year.

The prototype currently covers Footiqo top-5 domestic club leagues only, using competition type `1`.

## Entity And Alias Registry

Teams are keyed by `team_id`. A `team_id` is an internal integer and must not encode league. Clubs and national teams are entities; source names are aliases.

The locked entity registry is:

- `data/processed/entity_registry/teams_v1_locked.csv`
- `data/processed/entity_registry/team_aliases_v1_locked.csv`
- `data/processed/entity_registry/competitions_v1_locked.csv`
- `data/processed/entity_registry/matches_v1_locked.csv`
- `data/processed/entity_registry/source_match_map_v1_locked.csv`

Every source-specific team name must map through `team_aliases_v1_locked.csv` before a feature block joins to canonical matches. Fuzzy matches are candidates only until manually approved.

## Source Match Map

`source_match_map` links source match IDs and source team names to canonical match IDs and entity IDs. It is the audit bridge between source records and canonical registry rows. It should preserve source provenance and make one-to-one assumptions explicit.

## Feature Blocks

A feature block is a table keyed by `canonical_match_id`. It should preserve canonical row count unless the report clearly documents an intentional scope difference. Feature blocks must include missingness and staleness flags when coverage is incomplete or source freshness varies.

Current approved feature blocks:

- ClubElo locked block: latest rating strictly before match date; no same-day ratings unless proven pre-kickoff.
- Understat locked lagged block: post-match data used only as historical rolling features with `understat_date < match_date`.

## Super CSVs

Super CSVs are market-specific research datasets. They include a single market target, paired market odds, market no-vig probabilities, allowed feature blocks, provenance columns, and forbidden audit columns.

- `research_ready/`: Footiqo market datasets after fixed availability filters.
- `research_ready_plus/clubelo/`: Footiqo plus locked ClubElo features.
- `research_ready_plus/clubelo_understat/`: Footiqo plus locked ClubElo and locked lagged Understat features.

Super CSVs are not betting signals. They are research inputs.

## Odds Timing Limitation

Footiqo odds timing remains unknown. All datasets remain `research_only`. Unknown timing prevents production betting claims even if a predictive audit shows improvement.

## Leakage Rules

Allowed:

- Market-specific no-vig probabilities for the market being studied.
- Date-safe rolling Footiqo features.
- ClubElo ratings strictly before the match date.
- Understat lagged rolling features from matches before the match date.
- Missingness, coverage, history, and staleness flags.

Forbidden as model features:

- `canonical_match_id`, source IDs, source file paths, alias IDs, and team names.
- Targets.
- Raw odds as model features, except retained for audit/settlement.
- Same-match scores/results and same-match post-match stats.
- Current fixture xG or Understat stats.
- Unrelated market odds/probabilities.
- Rejected or manual-review aliases.
- Future data.

## Adding New Sources

Recommended order:

1. Inventory and classify source files.
2. Create source-specific schema and leakage reports.
3. Map source competitions and teams through locked registries.
4. Add aliases as candidates, then manually lock them.
5. Build a standalone feature block keyed by `canonical_match_id`.
6. Validate temporal safety and coverage.
7. Merge into research CSVs only after feature block validation passes.
8. Run predictive audits before any value-diagnostic discussion.

No confirmed edge is claimed anywhere in this pipeline.
