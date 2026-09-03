# Canonical ID Contract

`canonical_match_id` is the stable primary key for canonical match rows.

## Format

`canonical_match_id` uses integer int64-compatible format:

```text
TLLLYYYYGGGG
```

- `T`: competition type.
- `LLL`: three-digit competition code inside that competition type.
- `YYYY`: season start year or tournament year.
- `GGGG`: deterministic match sequence inside competition and season/year.

Do not use competition type `0`.

## Competition Type Values

- `1`: domestic club leagues
- `2`: international club competitions
- `3`: domestic club cups
- `4`: club supercups / single-match club trophies
- `5`: national team competitions

## Components

Persist these component columns alongside `canonical_match_id`:

- `competition_type`
- `competition_code`
- `season_start_year`
- `match_sequence`

## Match Sequence Ordering

Within each competition and season/year, assign `match_sequence` deterministically using:

1. `match_datetime`
2. `home_team_normalized` or `home_team_id`
3. `away_team_normalized` or `away_team_id`
4. `source_match_id` if available

`match_sequence` starts at `1`, has no gaps, and must not exceed `9999`.

## Team IDs Are Separate

`team_id` is a separate entity key. It must not be encoded into `canonical_match_id` because:

- Teams can move between competitions.
- Clubs must retain one identity across cups, leagues, and international competitions.
- National teams need stable IDs across tournaments.
- Match IDs identify fixtures; team IDs identify entities.

`team_id` must not encode league, country, or season.

## Examples

Premier League 2024/25 first match:

- `T = 1`
- `LLL = 001`
- `YYYY = 2024`
- `GGGG = 0001`
- `canonical_match_id = 100120240001`

Champions League 2024/25 first match:

- `T = 2`
- `LLL = 001`
- `YYYY = 2024`
- `GGGG = 0001`
- `canonical_match_id = 200120240001`

World Cup 2022 first match:

- `T = 5`
- `LLL = 001`
- `YYYY = 2022`
- `GGGG = 0001`
- `canonical_match_id = 500120220001`

These examples show the contract shape. Actual sequence values must be assigned only after sorting the audited canonical fixture set.
