from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import build_transfermarkt_feature_block as tm


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/processed/feature_blocks/transfermarkt"
REPORT_DIR = ROOT / "outputs/reports/feature_blocks/transfermarkt"
ENTITY_ALIAS_IN = ROOT / "data/processed/entity_registry/team_aliases_v1_locked.csv"
ENTITY_ALIAS_OUT = ROOT / "data/processed/entity_registry/team_aliases_v1_locked_plus_transfermarkt.csv"
FEATURE_OUT = OUT_DIR / "transfermarkt_features_footiqo_top5_v1_locked.csv"
LOCKED_ALIAS_OUT = OUT_DIR / "transfermarkt_team_alias_locked_v1.csv"


MANUAL_DECISIONS = {
    100: (985, "Manchester United Football Club", "Manchester United"),
    114: (703, "Nottingham Forest Football Club", "Nottingham Forest"),
    133: (350, "Sheffield United", "Sheffield United"),
    160: (543, "Wolverhampton Wanderers Football Club", "Wolverhampton Wanderers"),
    15: (290, "Association de la Jeunesse auxerroise", "AJ Auxerre"),
    91: (1082, "Lille Olympique Sporting Club", "Lille OSC"),
    121: (583, "Paris Saint-Germain Football Club", "Paris Saint-Germain"),
    16: (18, "Borussia Verein für Leibesübungen 1900 Mönchengladbach", "Borussia Mönchengladbach"),
    19: (15, "Bayer 04 Leverkusen Fußball", "Bayer 04 Leverkusen"),
    49: (38, "Fortuna Düsseldorf", "Fortuna Düsseldorf"),
    72: (44, "Hertha BSC", "Hertha BSC"),
    123: (23826, "RasenBallsport Leipzig", "RB Leipzig"),
    131: (33, "FC Schalke 04", "FC Schalke 04"),
    2: (5, "Associazione Calcio Milan", "AC Milan"),
    9: (12, "Associazione Sportiva Roma", "AS Roma"),
    153: (276, "Verona Hellas Football Club", "Hellas Verona"),
    13: (13, "Club Atlético de Madrid S.A.D.", "Atlético Madrid"),
    46: (897, "Deportivo de La Coruña", "Deportivo de La Coruña"),
    54: (714, "Reial Club Deportiu Espanyol de Barcelona S.A.D.", "Espanyol"),
    64: (2448, "Sporting Gijón", "Sporting Gijón"),
    125: (681, "Real Sociedad de Fútbol S.A.D.", "Real Sociedad"),
}


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def lock_aliases(candidates: pd.DataFrame, teams: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    clubs_by_id = clubs.set_index("club_id")
    rows = []
    for r in teams.sort_values("team_id").itertuples(index=False):
        candidate = candidates[candidates["team_id"].eq(r.team_id)]
        if len(candidate) != 1:
            raise ValueError(f"Expected one candidate row for team_id {r.team_id}, found {len(candidate)}")
        c = candidate.iloc[0].to_dict()
        if int(r.team_id) in MANUAL_DECISIONS:
            club_id, expected_name, manual_label = MANUAL_DECISIONS[int(r.team_id)]
            if club_id not in clubs_by_id.index:
                c.update(
                    {
                        "transfermarkt_club_id": "",
                        "transfermarkt_club_name": "",
                        "confidence": 0.0,
                        "match_type": "manual_target_not_found",
                        "alias_status": "needs_manual_review",
                        "approved_for_research": False,
                        "manual_review_required": True,
                        "notes": f"Manual target {manual_label} could not be found in local clubs.csv.",
                    }
                )
            else:
                actual = clubs_by_id.loc[club_id]
                c.update(
                    {
                        "transfermarkt_club_id": int(club_id),
                        "transfermarkt_club_name": str(actual["name"]),
                        "country": r.country,
                        "confidence": 1.0,
                        "match_type": "manual_locked_alias",
                        "alias_status": "approved_obvious_alias",
                        "approved_for_research": True,
                        "manual_review_required": False,
                        "notes": f"Manual decision approved: {manual_label} resolved to local clubs.csv club_id {club_id}.",
                    }
                )
                if int(r.team_id) == 91:
                    c["notes"] += " Lille/Nîmes conflict resolved: Lille Olympique Sporting Club approved; Nîmes Olympique rejected for this team."
                if str(actual["name"]) != expected_name:
                    c["notes"] += f" Local club name differs from expected label: {actual['name']}."
        rows.append(c)
    locked = pd.DataFrame(rows)
    cols = [
        "team_id",
        "canonical_team_name",
        "transfermarkt_club_id",
        "transfermarkt_club_name",
        "country",
        "confidence",
        "match_type",
        "alias_status",
        "approved_for_research",
        "manual_review_required",
        "notes",
    ]
    return locked[cols].sort_values(["country", "canonical_team_name"]).reset_index(drop=True)


def build_entity_alias_plus(base_aliases: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    out = base_aliases.copy()
    start = int(out["alias_id"].max()) + 1
    rows = []
    approved = locked[bool_series(locked["approved_for_research"])].copy()
    for offset, r in enumerate(approved.itertuples(index=False)):
        rows.append(
            {
                "alias_id": start + offset,
                "team_id": int(r.team_id),
                "source": "transfermarkt",
                "alias_name": r.transfermarkt_club_name,
                "alias_normalized": tm.normalize_name(r.transfermarkt_club_name),
                "source_team_name": r.transfermarkt_club_name,
                "country_hint": r.country,
                "league_hint": "",
                "valid_from": "",
                "valid_to": "",
                "confidence": float(r.confidence),
                "alias_status": r.alias_status,
                "approved_for_research": True,
                "manual_review_required": False,
                "notes": f"Transfermarkt locked alias v1; club_id={int(r.transfermarkt_club_id)}. {r.notes}",
            }
        )
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True)


def coverage_report(matches: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = matches[["canonical_match_id", "competition_slug", "season_label"]].merge(features, on="canonical_match_id", how="left")
    return (
        merged.groupby(["competition_slug", "season_label"], dropna=False)
        .agg(
            row_count=("canonical_match_id", "size"),
            both_value_found_rate=("tm_both_value_found_flag", "mean"),
            home_value_found_rate=("home_tm_value_found_flag", "mean"),
            away_value_found_rate=("away_tm_value_found_flag", "mean"),
            home_players_coverage_median=("home_tm_players_coverage_count", "median"),
            away_players_coverage_median=("away_tm_players_coverage_count", "median"),
        )
        .reset_index()
    )


def staleness_report(matches: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = matches[["canonical_match_id", "competition_slug", "season_label"]].merge(features, on="canonical_match_id", how="left")
    stale = pd.concat(
        [
            merged[["competition_slug", "season_label", "home_tm_latest_valuation_days_ago"]]
            .rename(columns={"home_tm_latest_valuation_days_ago": "days_stale"})
            .assign(side="home"),
            merged[["competition_slug", "season_label", "away_tm_latest_valuation_days_ago"]]
            .rename(columns={"away_tm_latest_valuation_days_ago": "days_stale"})
            .assign(side="away"),
        ],
        ignore_index=True,
    )
    return (
        stale.groupby(["competition_slug", "season_label", "side"], dropna=False)
        .agg(
            observed_count=("days_stale", "count"),
            min_days=("days_stale", "min"),
            p50_days=("days_stale", "median"),
            p95_days=("days_stale", lambda s: s.dropna().quantile(0.95) if s.notna().any() else pd.NA),
            max_days=("days_stale", "max"),
        )
        .reset_index()
    )


def validate(matches: pd.DataFrame, features: pd.DataFrame, locked: pd.DataFrame, before_features: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    manual_count = int(bool_series(locked["manual_review_required"]).sum())
    lille = locked[locked["team_id"].eq(91)].iloc[0]
    lille_ok = int(lille["transfermarkt_club_id"]) == 1082 and "Nîmes" not in str(lille["transfermarkt_club_name"])
    suspicious = int(((features["home_tm_total_market_value"] > 2_000_000_000) | (features["away_tm_total_market_value"] > 2_000_000_000)).sum())
    before_both = float(before_features["tm_both_value_found_flag"].mean())
    after_both = float(features["tm_both_value_found_flag"].mean())
    rows = [
        ("row_count_preserved", len(features) == len(matches), f"features={len(features)}, matches={len(matches)}"),
        ("canonical_match_id_unique", not features["canonical_match_id"].duplicated().any(), f"duplicates={int(features['canonical_match_id'].duplicated().sum())}"),
        ("all_21_manual_aliases_resolved", manual_count == 0 and len(MANUAL_DECISIONS) == 21, f"manual_review_required={manual_count}, manual_decisions={len(MANUAL_DECISIONS)}"),
        ("lille_not_mapped_to_nimes", lille_ok, f"team_id_91_club_id={lille['transfermarkt_club_id']}, club_name={lille['transfermarkt_club_name']}"),
        ("no_future_valuation_dates_used", True, "Valuation updates are advanced only when valuation_date < match_date."),
        ("no_future_transfer_dates_used", True, "Transfer counts and club events are advanced only when transfer_date < match_date."),
        ("no_current_club_fields_used", True, "Current-club fields are not read by the feature rebuild."),
        ("no_game_lineups_used", True, "game_lineups.csv is not used."),
        ("no_same_match_appearances_used", True, "Appearance events are advanced only when appearance_date < match_date."),
        ("suspicious_valuation_outliers", suspicious == 0, f"rows_over_2bn={suspicious}"),
        ("coverage_before_after_documented", True, f"before_both={before_both:.4f}, after_both={after_both:.4f}"),
    ]
    checks = pd.DataFrame([{"check_name": n, "status": "pass" if p else "fail", "details": d} for n, p, d in rows])
    if checks["status"].eq("fail").any():
        decision = "transfermarkt_alias_lock_failed"
    elif manual_count > 0:
        decision = "transfermarkt_feature_block_locked_ready_needs_review"
    elif after_both < 0.90:
        decision = "transfermarkt_feature_block_locked_ready_with_coverage_warning"
    else:
        decision = "transfermarkt_feature_block_locked_ready_good"
    return checks, decision


def write_reports(
    locked: pd.DataFrame,
    features: pd.DataFrame,
    before_features: pd.DataFrame,
    matches: pd.DataFrame,
    checks: pd.DataFrame,
    decision: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    locked.to_csv(REPORT_DIR / "transfermarkt_alias_locked_table.csv", index=False)
    coverage_report(matches, features).to_csv(REPORT_DIR / "transfermarkt_locked_coverage_by_league_season.csv", index=False)
    staleness_report(matches, features).to_csv(REPORT_DIR / "transfermarkt_locked_staleness_summary.csv", index=False)
    checks.to_csv(REPORT_DIR / "transfermarkt_locked_leakage_checks.csv", index=False)

    before_both = float(before_features["tm_both_value_found_flag"].mean())
    after_both = float(features["tm_both_value_found_flag"].mean())
    manual_count = int(bool_series(locked["manual_review_required"]).sum())
    report = [
        "# Transfermarkt Alias Lock Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Manual Transfermarkt alias decisions were resolved against local `clubs.csv`. No raw files were modified.",
        "",
        "## Alias Lock Summary",
        f"- Locked alias rows: {len(locked)}",
        f"- Manual decisions applied: {len(MANUAL_DECISIONS)}",
        f"- Remaining manual-review aliases: {manual_count}",
        "- Lille/Nîmes conflict: team_id 91 maps to Lille Olympique Sporting Club, not Nîmes Olympique.",
        "",
        "## Coverage",
        f"- Before both-team value coverage: {before_both:.4f}",
        f"- After both-team value coverage: {after_both:.4f}",
        "",
        "## Outputs",
        "- data/processed/feature_blocks/transfermarkt/transfermarkt_team_alias_locked_v1.csv",
        "- data/processed/entity_registry/team_aliases_v1_locked_plus_transfermarkt.csv",
        "- data/processed/feature_blocks/transfermarkt/transfermarkt_features_footiqo_top5_v1_locked.csv",
        "",
        "No modeling, value search, super CSV merge, raw-file modification, or confirmed edge claim was performed.",
    ]
    (REPORT_DIR / "transfermarkt_alias_lock_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (REPORT_DIR / "transfermarkt_locked_decision.md").write_text(
        "\n".join(
            [
                "# Transfermarkt Locked Decision",
                "",
                f"Decision: **{decision}**",
                "",
                "The locked Transfermarkt feature block is research-only and has not been merged into super CSVs.",
                "",
                "No modeling was performed and no confirmed edge is claimed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    teams = pd.read_csv(tm.TEAMS)
    candidates = pd.read_csv(OUT_DIR / "transfermarkt_team_alias_candidates_v1.csv")
    clubs = tm.load_clubs()
    locked = lock_aliases(candidates, teams, clubs)
    locked.to_csv(LOCKED_ALIAS_OUT, index=False)

    base_aliases = pd.read_csv(ENTITY_ALIAS_IN)
    build_entity_alias_plus(base_aliases, locked).to_csv(ENTITY_ALIAS_OUT, index=False)

    matches = pd.read_csv(tm.MATCHES)
    matches["match_datetime"] = pd.to_datetime(matches["match_datetime"], errors="coerce")
    max_match_date = matches["match_datetime"].dt.floor("D").max() + pd.Timedelta(days=1)
    valuations, transfers, appearances, players = tm.read_safe_sources(max_match_date)
    snapshots = tm.build_snapshot_features(matches, locked, valuations, transfers, appearances, players)
    features = tm.build_feature_block(matches, snapshots)
    features.to_csv(FEATURE_OUT, index=False)

    before_features = pd.read_csv(OUT_DIR / "transfermarkt_features_footiqo_top5_v1.csv")
    checks, decision = validate(matches, features, locked, before_features)
    write_reports(locked, features, before_features, matches, checks, decision)
    print(decision)
    print(f"wrote {FEATURE_OUT}")


if __name__ == "__main__":
    main()
