# Entity Registry Contract

The entity registry gives clubs, teams, competitions, and source aliases stable keys.

## Locked Tables

- `data/processed/entity_registry/teams_v1_locked.csv`
- `data/processed/entity_registry/team_aliases_v1_locked.csv`
- `data/processed/entity_registry/competitions_v1_locked.csv`
- `data/processed/entity_registry/matches_v1_locked.csv`
- `data/processed/entity_registry/source_match_map_v1_locked.csv`

Original non-locked v1 files are retained for audit history and should not be deleted.

## Team ID Policy

- `team_id` is an internal integer ID.
- `team_id` must not encode league.
- One club has one `team_id` across competitions.
- One national team has one `team_id` across competitions.
- Valid `team_type` values are `club` and `national_team`.
- Source names are aliases, not separate teams.

## Alias Policy

Every source-specific team name must map through `team_aliases_v1_locked.csv` before joining feature blocks.

Important columns:

- `source`: source system, such as `footiqo`, `clubelo`, or `understat`.
- `alias_name`: source-facing alias.
- `alias_normalized`: deterministic normalized alias.
- `team_id`: internal entity ID.
- `confidence`: audit confidence, not proof by itself.
- `alias_status`: status such as `approved_exact`, `approved_obvious_alias`, `needs_manual_review`, or `rejected`.
- `approved_for_research`: true only when the alias is allowed for research joins.
- `manual_review_required`: true when a human decision is still needed.

## Rejected Aliases

Rejected aliases must not be used in feature blocks, source joins, or model features.

Example: Understat alias `Dijon` was a false fuzzy candidate for `Gijon`, Spain. Dijon is a French club and must not map to Gijon. The rejected alias remains in the locked alias table for audit history with `approved_for_research=false`.

## No Silent Fuzzy Approval

Fuzzy matching can generate candidates, but no fuzzy alias can be silently approved. A candidate must be explicitly approved or rejected before it can affect a research feature block.

## Source Match Map

`source_match_map_v1_locked.csv` links:

- `canonical_match_id`
- source match IDs
- source league/team names
- source home/away `team_id`
- source home/away alias IDs where available

It is the controlled bridge between raw source identities and canonical match identities.
