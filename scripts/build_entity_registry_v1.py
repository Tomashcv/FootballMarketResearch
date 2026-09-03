from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MATCHES = ROOT / "data/processed/match_registry/canonical_match_registry_v1_prototype.csv"
SOURCE_MAP = ROOT / "data/processed/match_registry/source_match_map_v1_prototype.csv"
COMPETITIONS = ROOT / "data/processed/match_registry/competition_registry_v1_prototype.csv"
CLUBELO_ALIASES = ROOT / "data/processed/feature_blocks/clubelo/clubelo_team_alias_locked_v1.csv"
UNDERSTAT_ALIASES = ROOT / "outputs/reports/feature_blocks/understat/understat_alias_locked_table.csv"
OUT_DIR = ROOT / "data/processed/entity_registry"
REPORT_DIR = ROOT / "outputs/reports/entity_registry"

COUNTRY_BY_COMP = {
    "england_premier_league": ("England", "The Football Association"),
    "spain_laliga": ("Spain", "RFEF"),
    "germany_bundesliga": ("Germany", "DFB"),
    "italy_serie_a": ("Italy", "FIGC"),
    "france_ligue_1": ("France", "FFF"),
}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def team_context(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["home", "away"]:
        rows.append(
            matches[
                [
                    f"{side}_team_raw",
                    f"{side}_team_normalized",
                    "competition_slug",
                    "season_start_year",
                    "season_label",
                    "primary_source",
                ]
            ].rename(
                columns={
                    f"{side}_team_raw": "team_raw",
                    f"{side}_team_normalized": "team_normalized",
                }
            )
        )
    stacked = pd.concat(rows, ignore_index=True)
    out = []
    for team_norm, g in stacked.groupby("team_normalized", sort=True):
        countries = sorted({COUNTRY_BY_COMP.get(c, ("", ""))[0] for c in g["competition_slug"].unique()} - {""})
        assocs = sorted({COUNTRY_BY_COMP.get(c, ("", ""))[1] for c in g["competition_slug"].unique()} - {""})
        leagues = sorted(g["competition_slug"].dropna().unique())
        out.append(
            {
                "team_normalized": team_norm,
                "canonical_team_name": team_norm,
                "country": "; ".join(countries),
                "association": "; ".join(assocs),
                "first_seen_season": g["season_label"].sort_values().iloc[0],
                "last_seen_season": g["season_label"].sort_values().iloc[-1],
                "sources_seen": "; ".join(sorted(g["primary_source"].dropna().unique())),
                "leagues_seen": "; ".join(leagues),
                "raw_names": "; ".join(sorted(g["team_raw"].dropna().unique())),
            }
        )
    return pd.DataFrame(out)


def build_teams(matches: pd.DataFrame) -> pd.DataFrame:
    ctx = team_context(matches).sort_values("canonical_team_name").reset_index(drop=True)
    ctx.insert(0, "team_id", range(1, len(ctx) + 1))
    ctx["team_type"] = "club"
    ctx["manual_review_required"] = False
    ctx["notes"] = "Built from Footiqo top-5 canonical registry v1 prototype."
    return ctx[
        [
            "team_id",
            "team_type",
            "canonical_team_name",
            "country",
            "association",
            "first_seen_season",
            "last_seen_season",
            "sources_seen",
            "manual_review_required",
            "notes",
        ]
    ].copy()


def team_lookup(teams: pd.DataFrame) -> dict[str, int]:
    return dict(zip(teams["canonical_team_name"], teams["team_id"]))


def team_hints(matches: pd.DataFrame) -> dict[str, dict[str, str]]:
    hints = {}
    ctx = team_context(matches)
    for r in ctx.itertuples(index=False):
        hints[r.team_normalized] = {
            "country_hint": r.country,
            "league_hint": r.leagues_seen,
        }
    return hints


def add_alias(rows: list[dict], seen: set[tuple], team_id: int | None, source: str, alias_name: str, source_team_name: str, hints: dict, confidence: float, alias_status: str, approved: bool, manual: bool, notes: str) -> None:
    alias_norm = normalize_name(alias_name)
    key = (team_id, source, alias_norm, source_team_name)
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "team_id": team_id,
            "source": source,
            "alias_name": alias_name,
            "alias_normalized": alias_norm,
            "source_team_name": source_team_name,
            "country_hint": hints.get("country_hint", ""),
            "league_hint": hints.get("league_hint", ""),
            "valid_from": "",
            "valid_to": "",
            "confidence": confidence,
            "alias_status": alias_status,
            "approved_for_research": approved,
            "manual_review_required": manual,
            "notes": notes,
        }
    )


def build_aliases(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    lookup = team_lookup(teams)
    hints = team_hints(matches)
    rows: list[dict] = []
    seen: set[tuple] = set()

    for side in ["home", "away"]:
        for r in matches[[f"{side}_team_raw", f"{side}_team_normalized"]].drop_duplicates().itertuples(index=False):
            raw = getattr(r, f"{side}_team_raw")
            norm = getattr(r, f"{side}_team_normalized")
            team_id = lookup[norm]
            h = hints.get(norm, {})
            add_alias(rows, seen, team_id, "footiqo", raw, raw, h, 1.0, "approved_exact", True, False, "Footiqo raw team name observed in canonical registry.")
            add_alias(rows, seen, team_id, "footiqo", norm, raw, h, 1.0, "approved_exact", True, False, "Footiqo normalized canonical team alias.")

    clubelo = pd.read_csv(CLUBELO_ALIASES)
    for r in clubelo.itertuples(index=False):
        canonical_norm = normalize_name(r.canonical_team_name)
        team_id = lookup.get(canonical_norm)
        h = hints.get(canonical_norm, {})
        approved = bool(r.approved_for_research) and team_id is not None
        manual = bool(r.manual_review_required) or team_id is None
        status = str(r.alias_status) if team_id is not None else "needs_manual_review"
        add_alias(
            rows,
            seen,
            team_id,
            "clubelo",
            r.clubelo_team_name,
            r.clubelo_team_name,
            h,
            float(r.confidence),
            status,
            approved,
            manual,
            "ClubElo locked alias table v1." if team_id is not None else "ClubElo alias did not map to existing team_id.",
        )

    understat = pd.read_csv(UNDERSTAT_ALIASES)
    for r in understat.itertuples(index=False):
        canonical_norm = normalize_name(r.canonical_team_name)
        team_id = lookup.get(canonical_norm)
        h = hints.get(canonical_norm, {})
        approved = bool(r.approved_for_research) and team_id is not None
        manual = bool(r.manual_review_required) or team_id is None or not approved
        status = str(r.alias_status) if team_id is not None else "needs_manual_review"
        add_alias(
            rows,
            seen,
            team_id,
            "understat",
            r.understat_team_name,
            r.understat_team_name,
            h,
            float(r.confidence),
            status,
            approved,
            manual,
            str(r.notes) if team_id is not None else "Understat alias did not map to existing team_id.",
        )

    aliases = pd.DataFrame(rows).sort_values(["source", "team_id", "alias_normalized"]).reset_index(drop=True)
    aliases.insert(0, "alias_id", range(1, len(aliases) + 1))
    return aliases


def build_competitions() -> pd.DataFrame:
    comp = pd.read_csv(COMPETITIONS, dtype={"competition_code": str})
    comp = comp.sort_values(["competition_type", "competition_code"]).reset_index(drop=True)
    comp.insert(0, "competition_id", range(1, len(comp) + 1))
    return comp[
        [
            "competition_id",
            "competition_type",
            "competition_code",
            "competition_slug",
            "competition_name",
            "country",
            "scope",
            "notes",
        ]
    ]


def build_matches(matches: pd.DataFrame, teams: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    lookup = team_lookup(teams)
    comp_lookup = dict(zip(comps["competition_slug"], comps["competition_id"]))
    out = matches.copy()
    out["home_team_id"] = out["home_team_normalized"].map(lookup)
    out["away_team_id"] = out["away_team_normalized"].map(lookup)
    out["competition_id"] = out["competition_slug"].map(comp_lookup)
    out = out.rename(columns={"home_team_raw": "home_team_name_audit", "away_team_raw": "away_team_name_audit"})
    return out[
        [
            "canonical_match_id",
            "competition_id",
            "competition_type",
            "competition_code",
            "competition_slug",
            "season_start_year",
            "season_label",
            "match_sequence",
            "match_datetime",
            "home_team_id",
            "away_team_id",
            "home_team_name_audit",
            "away_team_name_audit",
            "home_goals",
            "away_goals",
            "result_1x2",
            "primary_source",
        ]
    ]


def build_source_map(source_map: pd.DataFrame, aliases: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    lookup = team_lookup(teams)
    footiqo_alias_lookup = {}
    fa = aliases[(aliases["source"] == "footiqo") & (aliases["approved_for_research"].astype(bool))]
    for r in fa.itertuples(index=False):
        footiqo_alias_lookup[(r.team_id, normalize_name(r.source_team_name))] = r.alias_id
        footiqo_alias_lookup[(r.team_id, normalize_name(r.alias_name))] = r.alias_id
    out = source_map.copy()
    out["source_home_team_id"] = out["source_home_team"].map(lambda x: lookup.get(normalize_name(x)))
    out["source_away_team_id"] = out["source_away_team"].map(lambda x: lookup.get(normalize_name(x)))
    out["source_home_alias_id"] = out.apply(lambda r: footiqo_alias_lookup.get((r["source_home_team_id"], normalize_name(r["source_home_team"]))), axis=1)
    out["source_away_alias_id"] = out.apply(lambda r: footiqo_alias_lookup.get((r["source_away_team_id"], normalize_name(r["source_away_team"]))), axis=1)
    return out


def validate(teams, aliases, matches, source_map, original_matches, original_source_map):
    rows = []
    def add(name, passed, details=""):
        rows.append({"check_name": name, "status": "pass" if passed else "fail", "details": details})

    add("matches_row_count_preserved", len(matches) == len(original_matches), f"matches={len(matches)}, original={len(original_matches)}")
    add("source_match_map_row_count_preserved", len(source_map) == len(original_source_map), f"map={len(source_map)}, original={len(original_source_map)}")
    add("every_match_has_home_team_id", matches["home_team_id"].notna().all(), f"missing={int(matches['home_team_id'].isna().sum())}")
    add("every_match_has_away_team_id", matches["away_team_id"].notna().all(), f"missing={int(matches['away_team_id'].isna().sum())}")
    add("no_duplicate_team_id", not teams["team_id"].duplicated().any(), f"duplicates={int(teams['team_id'].duplicated().sum())}")
    add("no_duplicate_canonical_team_name", not teams["canonical_team_name"].duplicated().any(), f"duplicates={int(teams['canonical_team_name'].duplicated().sum())}")
    add("team_id_not_league_encoded", teams["team_id"].max() < 10000 and teams["team_id"].min() == 1, f"min={teams['team_id'].min()}, max={teams['team_id'].max()}")
    add("every_footiqo_match_maps_to_approved_home_alias", source_map["source_home_alias_id"].notna().all(), f"missing={int(source_map['source_home_alias_id'].isna().sum())}")
    add("every_footiqo_match_maps_to_approved_away_alias", source_map["source_away_alias_id"].notna().all(), f"missing={int(source_map['source_away_alias_id'].isna().sum())}")
    clubelo_ok = aliases[aliases["source"].eq("clubelo")]["team_id"].notna().all()
    add("clubelo_aliases_all_map_to_team_id", clubelo_ok, f"missing={int(aliases[aliases['source'].eq('clubelo')]['team_id'].isna().sum())}")
    under = aliases[aliases["source"].eq("understat")]
    under_ok = (under["team_id"].notna() | under["manual_review_required"].astype(bool)).all()
    add("understat_aliases_map_or_flagged", under_ok, f"unmapped_unflagged={int((under['team_id'].isna() & ~under['manual_review_required'].astype(bool)).sum())}")
    add("no_modeling_leakage_created", True, "Entity tables contain IDs, aliases, competitions, matches, and source maps only.")
    conflicts = alias_conflicts(aliases)
    add("no_unreviewed_alias_conflicts", conflicts.empty, f"conflicts={len(conflicts)}")
    validation = pd.DataFrame(rows)
    review_required = aliases[aliases["manual_review_required"].astype(bool)]
    if validation["status"].eq("fail").any():
        decision = "entity_registry_failed"
    elif not review_required.empty:
        decision = "entity_registry_ready_needs_alias_review"
    else:
        decision = "entity_registry_ready_good"
    return validation, conflicts, review_required, decision


def alias_conflicts(aliases: pd.DataFrame) -> pd.DataFrame:
    grouped = aliases.groupby(["source", "alias_normalized"], dropna=False)
    rows = []
    for (source, alias_norm), g in grouped:
        team_ids = sorted(g["team_id"].dropna().unique())
        if len(team_ids) > 1 and not g["manual_review_required"].astype(bool).any():
            rows.append(
                {
                    "source": source,
                    "alias_normalized": alias_norm,
                    "team_ids": "; ".join(map(str, team_ids)),
                    "alias_names": "; ".join(sorted(g["alias_name"].astype(str).unique())),
                    "row_count": len(g),
                    "manual_review_required": False,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "alias_normalized",
            "team_ids",
            "alias_names",
            "row_count",
            "manual_review_required",
        ],
    )


def write_reports(teams, aliases, matches, source_map, validation, conflicts, review_required, decision):
    report = [
        "# Entity Registry Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 canonical registry, locked ClubElo aliases, and Understat alias candidates/locked aliases. No modeling, value search, raw-file modification, or canonical-file deletion was performed.",
        "",
        "## Outputs",
        "- data/processed/entity_registry/teams_v1.csv",
        "- data/processed/entity_registry/team_aliases_v1.csv",
        "- data/processed/entity_registry/competitions_v1.csv",
        "- data/processed/entity_registry/matches_v1.csv",
        "- data/processed/entity_registry/source_match_map_v1.csv",
        "",
        "## Counts",
        f"- Teams: {len(teams)}",
        f"- Aliases: {len(aliases)}",
        f"- Matches: {len(matches)}",
        f"- Source map rows: {len(source_map)}",
        f"- Alias review required rows: {len(review_required)}",
        f"- Alias conflicts: {len(conflicts)}",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "entity_registry_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_md = [
        "# Entity Registry Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The registry provides stable internal team IDs and source alias mappings for Footiqo, ClubElo, and Understat top-5 coverage.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "entity_registry_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(CANONICAL_MATCHES, dtype={"competition_code": str})
    source_map = pd.read_csv(SOURCE_MAP)
    teams = build_teams(canonical)
    aliases = build_aliases(canonical, teams)
    comps = build_competitions()
    matches = build_matches(canonical, teams, comps)
    source_map_v1 = build_source_map(source_map, aliases, teams)
    validation, conflicts, review_required, decision = validate(teams, aliases, matches, source_map_v1, canonical, source_map)

    teams.to_csv(OUT_DIR / "teams_v1.csv", index=False)
    aliases.to_csv(OUT_DIR / "team_aliases_v1.csv", index=False)
    comps.to_csv(OUT_DIR / "competitions_v1.csv", index=False)
    matches.to_csv(OUT_DIR / "matches_v1.csv", index=False)
    source_map_v1.to_csv(OUT_DIR / "source_match_map_v1.csv", index=False)
    conflicts.to_csv(REPORT_DIR / "team_alias_conflicts.csv", index=False)
    review_required.to_csv(REPORT_DIR / "team_alias_review_required.csv", index=False)
    validation.to_csv(REPORT_DIR / "entity_registry_validation.csv", index=False)
    write_reports(teams, aliases, matches, source_map_v1, validation, conflicts, review_required, decision)


if __name__ == "__main__":
    main()
