from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FOOTIQO = ROOT / "data/interim/footiqo/footiqo_top5_linked_audit_table.csv"
INPUT_COMPETITION_DRAFT = ROOT / "data/processed/match_registry/competition_registry_draft.csv"
REGISTRY_DIR = ROOT / "data/processed/match_registry"
REPORT_DIR = ROOT / "outputs/reports/match_registry"

TOP5_SLUGS = [
    "england_premier_league",
    "spain_laliga",
    "germany_bundesliga",
    "italy_serie_a",
    "france_ligue_1",
]

REQUIRED_INPUT_COLUMNS = [
    "id",
    "source_league_slug",
    "match_datetime",
    "matchDate",
    "Country",
    "League",
    "Season",
    "season_start_year",
    "homeTeam",
    "awayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]


def normalize_team(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def validate_input(df: pd.DataFrame, competition: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    rows.append(
        {
            "check_name": "required_input_columns_present",
            "status": "pass" if not missing else "fail",
            "details": "" if not missing else f"Missing: {missing}",
        }
    )
    observed_slugs = sorted(df["source_league_slug"].dropna().unique()) if "source_league_slug" in df else []
    rows.append(
        {
            "check_name": "only_top5_footiqo_leagues",
            "status": "pass" if observed_slugs == sorted(TOP5_SLUGS) else "fail",
            "details": f"observed={observed_slugs}",
        }
    )
    comp_slugs = sorted(competition["competition_slug"].dropna().unique())
    rows.append(
        {
            "check_name": "competition_registry_has_top5",
            "status": "pass" if set(TOP5_SLUGS).issubset(set(comp_slugs)) else "fail",
            "details": f"registry_slugs={comp_slugs}",
        }
    )
    return rows


def build_locked_competition_registry(draft: pd.DataFrame) -> pd.DataFrame:
    registry = draft[draft["competition_slug"].isin(TOP5_SLUGS)].copy()
    registry = registry.sort_values(["competition_type", "competition_code"])
    registry = registry[
        [
            "competition_type",
            "competition_code",
            "competition_slug",
            "competition_name",
            "country",
            "scope",
            "notes",
        ]
    ].copy()
    registry["competition_type"] = registry["competition_type"].astype("int64")
    registry["competition_code"] = registry["competition_code"].astype("int64").map(lambda x: f"{x:03d}")
    registry["notes"] = (
        "Prototype locked Footiqo top-5 domestic league code. "
        "Use only for canonical registry prototype until full competition audit."
    )
    return registry


def result_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def build_registry(df: pd.DataFrame, competition: pd.DataFrame) -> pd.DataFrame:
    comp = competition.set_index("competition_slug")
    work = df.copy()
    work["match_datetime"] = pd.to_datetime(work["match_datetime"], errors="coerce")
    work["home_team_normalized"] = work["homeTeam"].map(normalize_team)
    work["away_team_normalized"] = work["awayTeam"].map(normalize_team)
    work["source_match_id_sort"] = work["id"].astype(str)
    work = work.sort_values(
        [
            "source_league_slug",
            "season_start_year",
            "match_datetime",
            "home_team_normalized",
            "away_team_normalized",
            "source_match_id_sort",
        ],
        kind="mergesort",
    ).copy()
    work["match_sequence"] = (
        work.groupby(["source_league_slug", "season_start_year"], sort=False).cumcount() + 1
    ).astype("int64")
    work["competition_type"] = work["source_league_slug"].map(comp["competition_type"]).astype("int64")
    work["competition_code_int"] = work["source_league_slug"].map(comp["competition_code"]).astype("int64")
    work["competition_code"] = work["competition_code_int"].map(lambda x: f"{x:03d}")
    work["competition_slug"] = work["source_league_slug"]
    work["country"] = work["source_league_slug"].map(comp["country"])
    work["league_name"] = work["source_league_slug"].map(comp["competition_name"])
    work["season_start_year"] = work["season_start_year"].astype("int64")
    work["home_goals"] = work["FTHG"].astype("int64")
    work["away_goals"] = work["FTAG"].astype("int64")
    work["result_1x2"] = [
        result_from_goals(h, a) for h, a in zip(work["home_goals"], work["away_goals"])
    ]
    id_strings = [
        f"{t}{c:03d}{y:04d}{s:04d}"
        for t, c, y, s in zip(
            work["competition_type"],
            work["competition_code_int"],
            work["season_start_year"],
            work["match_sequence"],
        )
    ]
    work["canonical_match_id"] = pd.Series(id_strings, index=work.index).astype("int64")
    work["primary_source"] = "footiqo"
    out = work[
        [
            "canonical_match_id",
            "competition_type",
            "competition_code",
            "competition_slug",
            "season_start_year",
            "Season",
            "match_sequence",
            "match_datetime",
            "country",
            "league_name",
            "homeTeam",
            "awayTeam",
            "home_team_normalized",
            "away_team_normalized",
            "home_goals",
            "away_goals",
            "result_1x2",
            "primary_source",
        ]
    ].copy()
    out = out.rename(
        columns={
            "Season": "season_label",
            "homeTeam": "home_team_raw",
            "awayTeam": "away_team_raw",
        }
    )
    out["match_datetime"] = out["match_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def build_source_map(df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    key_cols_left = [
        "source_league_slug",
        "season_start_year",
        "match_datetime",
        "home_team_normalized",
        "away_team_normalized",
    ]
    source = df.copy()
    source["match_datetime"] = pd.to_datetime(source["match_datetime"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    source["home_team_normalized"] = source["homeTeam"].map(normalize_team)
    source["away_team_normalized"] = source["awayTeam"].map(normalize_team)
    source["season_start_year"] = source["season_start_year"].astype("int64")
    merged = source.merge(
        registry[
            [
                "canonical_match_id",
                "competition_slug",
                "season_start_year",
                "match_datetime",
                "home_team_normalized",
                "away_team_normalized",
            ]
        ],
        left_on=key_cols_left,
        right_on=[
            "competition_slug",
            "season_start_year",
            "match_datetime",
            "home_team_normalized",
            "away_team_normalized",
        ],
        how="left",
        validate="one_to_one",
    )
    manual_review = merged["canonical_match_id"].isna() | merged["id"].duplicated(keep=False)
    out = pd.DataFrame(
        {
            "canonical_match_id": merged["canonical_match_id"].astype("Int64"),
            "source": "footiqo",
            "source_match_id": merged["id"],
            "source_league_slug": merged["source_league_slug"],
            "source_match_datetime": merged["match_datetime"],
            "source_season": merged["Season"],
            "source_home_team": merged["homeTeam"],
            "source_away_team": merged["awayTeam"],
            "mapping_method": "source_primary_exact_id",
            "mapping_confidence": 1.0,
            "manual_review_required": manual_review,
        }
    )
    return out.sort_values(["canonical_match_id", "source_match_id"]).reset_index(drop=True)


def add_validation(rows: list[dict[str, object]], name: str, passed: bool, details: str = "") -> None:
    rows.append({"check_name": name, "status": "pass" if passed else "fail", "details": details})


def validate_outputs(
    input_df: pd.DataFrame,
    registry: pd.DataFrame,
    source_map: pd.DataFrame,
    competition: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, object]] = []
    id_str = registry["canonical_match_id"].astype(str)
    add_validation(rows, "canonical_match_id_int64_compatible", pd.api.types.is_integer_dtype(registry["canonical_match_id"]))
    add_validation(rows, "canonical_match_id_no_missing", registry["canonical_match_id"].notna().all())
    add_validation(rows, "canonical_match_id_no_duplicates", not registry["canonical_match_id"].duplicated().any())
    add_validation(rows, "canonical_match_id_12_digits", id_str.str.fullmatch(r"\d{12}").all())
    add_validation(rows, "canonical_match_id_starts_with_1", id_str.str.startswith("1").all())
    slug_to_code = competition.set_index("competition_slug")["competition_code"].astype(str).to_dict()
    code_match = registry["competition_code"].eq(registry["competition_slug"].map(slug_to_code)).all()
    add_validation(rows, "competition_code_matches_competition_slug", bool(code_match))
    season_check = input_df["season_start_year"].astype("int64").eq(
        input_df["Season"].astype(str).str[:4].astype("int64")
    ).all()
    add_validation(rows, "season_start_year_correct_from_season", bool(season_check))
    gaps_ok = True
    starts_ok = True
    max_ok = registry["match_sequence"].max() <= 9999
    for _, group in registry.groupby(["competition_type", "competition_code", "season_start_year"]):
        seq = sorted(group["match_sequence"].astype(int).tolist())
        starts_ok = starts_ok and (seq[0] == 1)
        gaps_ok = gaps_ok and (seq == list(range(1, len(seq) + 1)))
    add_validation(rows, "match_sequence_starts_at_1_per_competition_season", starts_ok)
    add_validation(rows, "match_sequence_has_no_gaps_per_competition_season", gaps_ok)
    add_validation(rows, "match_sequence_no_value_over_9999", bool(max_ok))
    duplicate_component = registry.duplicated(
        ["competition_type", "competition_code", "season_start_year", "match_sequence"]
    ).any()
    add_validation(
        rows,
        "no_duplicate_competition_type_code_season_sequence",
        not duplicate_component,
    )
    duplicate_match_key = registry.duplicated(
        ["competition_slug", "match_datetime", "home_team_normalized", "away_team_normalized"]
    ).any()
    add_validation(
        rows,
        "no_duplicate_competition_slug_datetime_home_away",
        not duplicate_match_key,
    )
    add_validation(rows, "row_count_equals_input_row_count", len(registry) == len(input_df), f"registry={len(registry)}, input={len(input_df)}")
    add_validation(
        rows,
        "source_map_row_count_equals_registry_row_count",
        len(source_map) == len(registry),
        f"source_map={len(source_map)}, registry={len(registry)}",
    )
    add_validation(
        rows,
        "source_map_no_manual_review_rows",
        not source_map["manual_review_required"].astype(bool).any(),
        f"manual_review_rows={int(source_map['manual_review_required'].astype(bool).sum())}",
    )
    validation = pd.DataFrame(rows)
    ready_good_checks = [
        "canonical_match_id_no_duplicates",
        "canonical_match_id_12_digits",
        "canonical_match_id_starts_with_1",
        "match_sequence_starts_at_1_per_competition_season",
        "match_sequence_has_no_gaps_per_competition_season",
        "match_sequence_no_value_over_9999",
        "row_count_equals_input_row_count",
        "source_map_row_count_equals_registry_row_count",
        "source_map_no_manual_review_rows",
    ]
    failed = validation[validation["status"] != "pass"]
    if not failed.empty:
        decision = "canonical_registry_prototype_failed"
    elif validation[validation["check_name"].isin(ready_good_checks)]["status"].eq("pass").all():
        decision = "canonical_registry_prototype_ready_good"
    else:
        decision = "canonical_registry_prototype_ready_needs_review"
    return validation, decision


def build_examples(registry: pd.DataFrame) -> pd.DataFrame:
    examples = []
    for (slug, season), group in registry.groupby(["competition_slug", "season_start_year"], sort=True):
        team_count = len(set(group["home_team_normalized"]).union(set(group["away_team_normalized"])))
        expected = team_count * (team_count - 1) if team_count else pd.NA
        examples.append(
            {
                "competition_slug": slug,
                "season_start_year": season,
                "season_label": group["season_label"].iloc[0],
                "first_canonical_match_id": int(group["canonical_match_id"].min()),
                "last_canonical_match_id": int(group["canonical_match_id"].max()),
                "row_count": len(group),
                "expected_row_count_if_inferable": expected,
                "sequence_min": int(group["match_sequence"].min()),
                "sequence_max": int(group["match_sequence"].max()),
            }
        )
    return pd.DataFrame(examples)


def write_reports(
    registry: pd.DataFrame,
    source_map: pd.DataFrame,
    validation: pd.DataFrame,
    examples: pd.DataFrame,
    decision: str,
) -> None:
    failed = validation[validation["status"] != "pass"]
    report = [
        "# Canonical Registry Prototype Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 linked audit table only. No external sources were joined. No models, value searches, or final super CSVs were built.",
        "",
        "## Outputs",
        "- data/processed/match_registry/competition_registry_v1_prototype.csv",
        "- data/processed/match_registry/canonical_match_registry_v1_prototype.csv",
        "- data/processed/match_registry/source_match_map_v1_prototype.csv",
        "",
        "## Counts",
        f"- Registry rows: {len(registry)}",
        f"- Source map rows: {len(source_map)}",
        f"- Leagues: {registry['competition_slug'].nunique()}",
        f"- League-seasons: {registry.groupby(['competition_slug', 'season_start_year']).ngroups}",
        "",
        "## Validation",
        f"- Passing checks: {int(validation['status'].eq('pass').sum())}",
        f"- Failing checks: {int(validation['status'].ne('pass').sum())}",
    ]
    if not failed.empty:
        report.extend(["", "## Failed Checks"])
        report.extend(f"- {r.check_name}: {r.details}" for r in failed.itertuples())
    report.extend(
        [
            "",
            "## Conservative Notes",
            "- This is a prototype registry, not the final canonical registry.",
            "- Footiqo is treated as the primary source only inside this prototype.",
            "- Odds timing remains outside this registry build and is not asserted here.",
            "- No confirmed edge is claimed.",
        ]
    )
    (REPORT_DIR / "canonical_registry_prototype_build_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    source_report = [
        "# Source Match Map Prototype Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Source: `footiqo`",
        "",
        f"- Rows: {len(source_map)}",
        f"- Unique source IDs: {source_map['source_match_id'].nunique()}",
        f"- Unique canonical IDs: {source_map['canonical_match_id'].nunique()}",
        f"- Manual review rows: {int(source_map['manual_review_required'].astype(bool).sum())}",
        "",
        "Mapping method: `source_primary_exact_id` with confidence `1.0`, after deterministic canonical ID assignment from the Footiqo linked audit table.",
    ]
    (REPORT_DIR / "source_match_map_prototype_report.md").write_text(
        "\n".join(source_report) + "\n", encoding="utf-8"
    )

    policy = """# Canonical ID Policy v1

Use int64 `canonical_match_id` with format `TLLLYYYYGGGG`.

- `T`: competition type. This prototype uses only `1` for domestic club leagues.
- `LLL`: three-digit competition code.
- `YYYY`: `season_start_year`.
- `GGGG`: deterministic match sequence within competition and season.

Prototype league codes:

- `1,001,england_premier_league`
- `1,002,spain_laliga`
- `1,003,germany_bundesliga`
- `1,004,italy_serie_a`
- `1,005,france_ligue_1`

Sequence ordering:

1. `match_datetime`
2. `home_team_normalized`
3. `away_team_normalized`
4. `source_match_id`

Type `0` is not used. IDs must be 12 digits when converted to string and int64-compatible.
"""
    (REPORT_DIR / "canonical_id_policy_v1.md").write_text(policy, encoding="utf-8")


def main() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    footiqo = pd.read_csv(INPUT_FOOTIQO)
    draft = pd.read_csv(INPUT_COMPETITION_DRAFT)
    pre_validation = validate_input(footiqo, draft)
    competition = build_locked_competition_registry(draft)
    registry = build_registry(footiqo, competition)
    source_map = build_source_map(footiqo, registry)
    validation, decision = validate_outputs(footiqo, registry, source_map, competition)
    validation = pd.concat([pd.DataFrame(pre_validation), validation], ignore_index=True)
    if validation["status"].ne("pass").any() and decision == "canonical_registry_prototype_ready_good":
        decision = "canonical_registry_prototype_failed"
    examples = build_examples(registry)

    competition.to_csv(REGISTRY_DIR / "competition_registry_v1_prototype.csv", index=False)
    registry.to_csv(REGISTRY_DIR / "canonical_match_registry_v1_prototype.csv", index=False)
    source_map.to_csv(REGISTRY_DIR / "source_match_map_v1_prototype.csv", index=False)
    validation.to_csv(REPORT_DIR / "canonical_registry_prototype_validation.csv", index=False)
    examples.to_csv(REPORT_DIR / "canonical_registry_prototype_examples.csv", index=False)
    write_reports(registry, source_map, validation, examples, decision)


if __name__ == "__main__":
    main()
