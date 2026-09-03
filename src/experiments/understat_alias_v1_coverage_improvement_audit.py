from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

import build_feature_matrix_v4_1_understat_partial as base


V3_MATRIX = Path("data/processed/features/football_feature_matrix_v3_clubelo_partial.csv")
UNDERSTAT_ZIP = Path("data/raw_external/understat_manual/understat_archive.zip")
MAPPING_DIR = Path("data/mappings")
REPORT_DIR = Path("outputs/reports")

ALIAS_MAPPING_CSV = MAPPING_DIR / "understat_football_data_aliases_v1.csv"
IMPACT_MD = REPORT_DIR / "understat_alias_v1_impact_audit.md"
ACCEPTED_CSV = REPORT_DIR / "understat_aliases_accepted_v1.csv"
MANUAL_CSV = REPORT_DIR / "understat_aliases_manual_review_required_v1.csv"
REJECTED_CSV = REPORT_DIR / "understat_aliases_rejected_or_ambiguous_v1.csv"
COVERAGE_AFTER_CSV = REPORT_DIR / "understat_mapping_coverage_after_alias_v1.csv"

OUT_MATRIX_V2 = Path("data/processed/features/football_feature_matrix_v4_1_understat_partial_v2.csv")
BUILD_REPORT_V2 = REPORT_DIR / "feature_matrix_v4_1_understat_v2_build_report.md"
COVERAGE_V2 = REPORT_DIR / "feature_matrix_v4_1_understat_v2_coverage_by_league_season.csv"
MISSINGNESS_V2 = REPORT_DIR / "feature_matrix_v4_1_understat_v2_missingness.csv"
LEAKAGE_V2 = REPORT_DIR / "feature_matrix_v4_1_understat_v2_leakage_checks.csv"
SCOPE_V2 = REPORT_DIR / "feature_matrix_v4_1_understat_v2_recommended_model_scope.md"

LOCKED_ROW_PREDICTIONS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_row_predictions.csv")
LOCKED_SELECTED_BETS = Path("outputs/reports/feature_matrix_v2_tm_1x2_locked_selected_bets.csv")


ACCEPTED_ALIASES = [
    ("E0", "Man City", "Manchester City", "abbreviation_vs_full_name"),
    ("E0", "Man United", "Manchester United", "abbreviation_vs_full_name"),
    ("E0", "Newcastle", "Newcastle United", "shortened_name_vs_full_name"),
    ("E0", "Wolves", "Wolverhampton Wanderers", "common_club_nickname_vs_full_name"),
    ("E0", "Nott'm Forest", "Nottingham Forest", "apostrophe_abbreviation_vs_full_name"),
    ("E0", "West Brom", "West Bromwich Albion", "shortened_name_vs_full_name"),
    ("E0", "QPR", "Queens Park Rangers", "initialism_vs_full_name"),
    ("D1", "Dortmund", "Borussia Dortmund", "shortened_name_vs_full_name"),
    ("D1", "Leverkusen", "Bayer Leverkusen", "shortened_name_vs_full_name"),
    ("D1", "M'gladbach", "Borussia M.Gladbach", "football_data_abbreviation_vs_understat_name"),
    ("D1", "Stuttgart", "VfB Stuttgart", "shortened_name_vs_full_name"),
    ("D1", "Ein Frankfurt", "Eintracht Frankfurt", "football_data_abbreviation_vs_full_name"),
    ("D1", "Mainz", "Mainz 05", "shortened_name_vs_full_name"),
    ("D1", "RB Leipzig", "RasenBallsport Leipzig", "brand_abbreviation_vs_understat_full_name"),
    ("D1", "FC Koln", "FC Cologne", "english_vs_german_city_spelling"),
    ("D1", "Hertha", "Hertha Berlin", "shortened_name_vs_full_name"),
    ("D1", "Bielefeld", "Arminia Bielefeld", "shortened_name_vs_full_name"),
    ("D1", "Hamburg", "Hamburger SV", "football_data_short_name_vs_understat_name"),
    ("D1", "Hannover", "Hannover 96", "shortened_name_vs_full_name"),
    ("SP1", "Ath Madrid", "Atletico Madrid", "football_data_abbreviation_vs_full_name"),
    ("SP1", "Ath Bilbao", "Athletic Club", "football_data_abbreviation_vs_understat_name"),
    ("SP1", "Celta", "Celta Vigo", "shortened_name_vs_full_name"),
    ("SP1", "Vallecano", "Rayo Vallecano", "shortened_name_vs_full_name"),
    ("SP1", "Espanol", "Espanyol", "alternate_spelling_without_y"),
    ("SP1", "Huesca", "SD Huesca", "shortened_name_vs_full_name"),
    ("SP1", "La Coruna", "Deportivo La Coruna", "shortened_name_vs_full_name"),
    ("SP1", "Sp Gijon", "Sporting Gijon", "football_data_abbreviation_vs_full_name"),
    ("I1", "Milan", "AC Milan", "shortened_name_vs_full_name"),
    ("I1", "Spal", "SPAL 2013", "shortened_name_vs_understat_registered_name"),
    ("F1", "Paris SG", "Paris Saint Germain", "football_data_abbreviation_vs_full_name"),
    ("F1", "St Etienne", "Saint-Etienne", "abbreviation_and_punctuation_variant"),
    ("F1", "Clermont", "Clermont Foot", "shortened_name_vs_full_name"),
]


MANUAL_REVIEW_ALIASES = [
    ("I1", "Pisa", "", "no matching Understat Serie A club in archive; do not map automatically"),
    ("SP1", "Oviedo", "", "no matching Understat La Liga club in archive; do not map automatically"),
    ("D1", "Aachen", "", "no matching Understat Bundesliga club in archive period"),
    ("I1", "Como", "Como", "already exact in Understat where present; leave to automatic exact mapping"),
]


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return path.stat().st_size, int(path.stat().st_mtime), h.hexdigest()


def locked_ids(path: Path) -> tuple[set[object], str]:
    if not path.exists():
        return set(), "unavailable"
    ids = set(pd.read_csv(path, usecols=["match_id"])["match_id"])
    return ids, str(path)


def accepted_alias_frame(game: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = game[["league", "club_name"]].drop_duplicates()
    valid_pairs = set(zip(valid["league"], valid["club_name"]))
    rows = []
    understat_league_by_fd = base.LEAGUE_TO_UNDERSTAT
    for league, fd_team, club, reason in ACCEPTED_ALIASES:
        under_league = understat_league_by_fd[league]
        ok = (under_league, club) in valid_pairs
        rows.append(
            {
                "league": league,
                "football_data_team": fd_team,
                "understat_club_name": club,
                "alias_status": "accepted_high_confidence_aliases_v1" if ok else "rejected_or_ambiguous",
                "confidence": "high" if ok else "none",
                "reason": reason if ok else f"target club not present in Understat {under_league}",
            }
        )
    for league, fd_team, club, reason in MANUAL_REVIEW_ALIASES:
        rows.append(
            {
                "league": league,
                "football_data_team": fd_team,
                "understat_club_name": club,
                "alias_status": "manual_review_required",
                "confidence": "manual_review",
                "reason": reason,
            }
        )
    alias_all = pd.DataFrame(rows).drop_duplicates(["league", "football_data_team", "alias_status"])
    accepted = alias_all[alias_all["alias_status"].eq("accepted_high_confidence_aliases_v1")].copy()
    manual = alias_all[alias_all["alias_status"].eq("manual_review_required")].copy()
    rejected = alias_all[alias_all["alias_status"].eq("rejected_or_ambiguous")].copy()
    return alias_all, accepted, manual, rejected


def combine_accepted_mapping(auto_accepted: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    alias_map = aliases.rename(
        columns={
            "football_data_team": "football_team",
            "alias_status": "mapping_status",
        }
    ).copy()
    alias_map["understat_league"] = alias_map["league"].map(base.LEAGUE_TO_UNDERSTAT)
    alias_map = alias_map[["league", "football_team", "understat_league", "understat_club_name", "mapping_status"]]
    combined = pd.concat([auto_accepted, alias_map], ignore_index=True)
    priority = combined["mapping_status"].map(
        {
            "accepted_exact_normalized": 0,
            "accepted_high_confidence_fuzzy": 1,
            "accepted_high_confidence_aliases_v1": 2,
        }
    ).fillna(9)
    combined = combined.assign(_priority=priority).sort_values(["league", "football_team", "_priority"])
    return combined.drop_duplicates(["league", "football_team"], keep="first").drop(columns="_priority")


def impact_table(v3: pd.DataFrame, mapping: pd.DataFrame, locked_row_ids: set[object], selected_ids: set[object]) -> pd.DataFrame:
    teams = mapping.drop_duplicates(["league", "football_team"])
    missing = teams[
        teams["league"].isin(base.TOP5)
        & ~teams["mapping_status"].isin(["accepted_exact_normalized", "accepted_high_confidence_fuzzy"])
    ].copy()
    rows = []
    for rec in missing.to_dict("records"):
        mask = v3["league"].eq(rec["league"]) & (
            v3["home_team"].eq(rec["football_team"]) | v3["away_team"].eq(rec["football_team"])
        )
        rows.append(
            {
                "league": rec["league"],
                "football_data_team": rec["football_team"],
                "mapping_status_before": rec["mapping_status"],
                "top_candidate": rec.get("candidate_club_name", ""),
                "top_candidate_score": rec.get("candidate_score", np.nan),
                "fixture_rows_impacted": int(mask.sum()),
                "test_2020_2025_rows_impacted": int((mask & v3["season_start_year"].between(2020, 2025)).sum()),
                "locked_prediction_rows_impacted": int((mask & v3["match_id"].isin(locked_row_ids)).sum()) if locked_row_ids else 0,
                "locked_selected_bets_impacted": int((mask & v3["match_id"].isin(selected_ids)).sum()) if selected_ids else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["test_2020_2025_rows_impacted", "fixture_rows_impacted"], ascending=[False, False]
    )


def make_coverage(v4: pd.DataFrame, selected_ids: set[object], locked_row_ids: set[object]) -> pd.DataFrame:
    rows = []
    rows.extend(base.segment_summary(v4, "all_rows"))
    rows.extend(base.segment_summary(v4[v4["league"].isin(base.TOP5)], "top5_only"))
    rows.extend(
        base.segment_summary(
            v4[v4["league"].isin(base.TOP5) & v4["season_start_year"].between(2020, 2025)],
            "top5_test_2020_2025",
        )
    )
    if locked_row_ids:
        rows.extend(base.segment_summary(v4[v4["match_id"].isin(locked_row_ids)], "locked_v3_prediction_row_universe"))
    if selected_ids:
        rows.extend(base.segment_summary(v4[v4["match_id"].isin(selected_ids)], "locked_v3_selected_bets"))
    rows.extend(base.segment_summary(v4, "by_league", "league"))
    rows.extend(base.segment_summary(v4, "by_season_start_year", "season_start_year"))
    return pd.DataFrame(rows)


def coverage_rate(coverage: pd.DataFrame, segment: str) -> float:
    row = coverage[coverage["segment"].eq(segment)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["both_available_rate"])


def write_outputs(
    decision: str,
    impact: pd.DataFrame,
    alias_all: pd.DataFrame,
    accepted_aliases: pd.DataFrame,
    manual_aliases: pd.DataFrame,
    rejected_aliases: pd.DataFrame,
    before_cov: pd.DataFrame,
    after_cov: pd.DataFrame,
    checks: pd.DataFrame,
    selected_source: str,
    locked_source: str,
    rebuilt: bool,
) -> None:
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    alias_all.to_csv(ALIAS_MAPPING_CSV, index=False)
    accepted_aliases.to_csv(ACCEPTED_CSV, index=False)
    manual_aliases.to_csv(MANUAL_CSV, index=False)
    rejected_aliases.to_csv(REJECTED_CSV, index=False)
    after_cov.to_csv(COVERAGE_AFTER_CSV, index=False)

    key_segments = ["all_rows", "top5_only", "top5_test_2020_2025", "locked_v3_prediction_row_universe", "locked_v3_selected_bets"]
    before_key = before_cov[before_cov["segment"].isin(key_segments)][["segment", "rows", "both_available", "both_available_rate"]].copy()
    after_key = after_cov[after_cov["segment"].isin(key_segments)][["segment", "rows", "both_available", "both_available_rate"]].copy()
    delta = after_key.merge(before_key, on="segment", suffixes=("_after", "_before"), how="left")
    delta["both_available_delta"] = delta["both_available_after"] - delta["both_available_before"]
    delta["both_available_rate_delta"] = delta["both_available_rate_after"] - delta["both_available_rate_before"]

    BUILD_REPORT_V2.write_text(
        "\n".join(
            [
                "# Feature Matrix V4.1 Understat V2 Alias Build Report",
                "",
                f"Decision: `{decision}`",
                "",
                "No predictive models, value searches, threshold optimization, FBref data, v3 candidate changes, or confirmed edge claims were made.",
                "",
                f"- Rebuilt matrix: `{OUT_MATRIX_V2}`" if rebuilt else "- Rebuilt matrix: not written because alias coverage did not materially improve.",
                f"- Accepted high-confidence aliases: {len(accepted_aliases)}",
                f"- Manual-review aliases: {len(manual_aliases)}",
                f"- Rejected/ambiguous aliases: {len(rejected_aliases)}",
                f"- Locked prediction universe source: `{locked_source}`",
                f"- Locked selected-bets source: `{selected_source}`",
                "",
                "## Key Coverage Delta",
                base.md_table(delta, 20),
                "",
                "## Leakage Checks",
                base.md_table(checks, 50),
                "",
            ]
        ),
        encoding="utf-8",
    )
    IMPACT_MD.write_text(
        "\n".join(
            [
                "# Understat Alias V1 Impact Audit",
                "",
                f"Decision: `{decision}`",
                "",
                "Aliases were limited to same-league, high-confidence football naming variants. No match results, current-match Understat stats, or future information were used.",
                "",
                "## Highest-Impact Unmatched Teams Before Alias V1",
                base.md_table(impact.head(40), 40),
                "",
                "## Accepted Alias Summary",
                base.md_table(accepted_aliases, 80),
                "",
                "## Coverage Delta",
                base.md_table(delta, 20),
                "",
            ]
        ),
        encoding="utf-8",
    )
    SCOPE_V2.write_text(
        "\n".join(
            [
                "# V4.1 Understat V2 Recommended Model Scope",
                "",
                f"Decision: `{decision}`",
                "",
                "Recommended safe scope remains top-five Understat leagues only: `E0`, `D1`, `SP1`, `I1`, `F1`.",
                "",
                "Use only rolling past-only Understat columns computed from rows with `understat_date < match_date`, plus availability/history/staleness controls.",
                "Do not use RFPL, current-match Understat xG/result/score fields, team-name strings, or same-day rows as pre-match features.",
                "Understat ends on 2024-09-29, so 2024/25 and 2025/26 need explicit stale/missing handling.",
                "No confirmed edge is claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    raw_before = file_fingerprint(UNDERSTAT_ZIP)
    v3 = pd.read_csv(V3_MATRIX, low_memory=False)
    v3["match_date"] = pd.to_datetime(v3["match_date"], errors="coerce")
    game = base.read_understat()

    mapping, auto_accepted = base.build_mapping(v3, game)
    locked_row_ids, locked_source = locked_ids(LOCKED_ROW_PREDICTIONS)
    selected_ids, selected_source = locked_ids(LOCKED_SELECTED_BETS)
    impact = impact_table(v3, mapping, locked_row_ids, selected_ids)

    alias_all, accepted_aliases, manual_aliases, rejected_aliases = accepted_alias_frame(game)
    combined_accepted = combine_accepted_mapping(auto_accepted, accepted_aliases)
    audit = base.attach_mapping(v3, combined_accepted)
    features, safety = base.build_features(v3, audit, game)
    v4 = pd.concat([v3.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    after_cov = make_coverage(v4, selected_ids, locked_row_ids)

    before_cov = pd.read_csv(base.COVERAGE_CSV)
    before_test = coverage_rate(before_cov, "top5_test_2020_2025")
    after_test = coverage_rate(after_cov, "top5_test_2020_2025")
    before_locked = coverage_rate(before_cov, "locked_v3_prediction_row_universe")
    after_locked = coverage_rate(after_cov, "locked_v3_prediction_row_universe")
    material = (after_test - before_test >= 0.05) and (np.isnan(before_locked) or after_locked - before_locked >= 0.05)

    raw_after = file_fingerprint(UNDERSTAT_ZIP)
    checks = base.leakage_checks(v3.reset_index(drop=True), v4, features, safety, raw_before, raw_after)
    checks.to_csv(LEAKAGE_V2, index=False)

    rebuilt = False
    if checks["status"].ne("pass").any():
        decision = "v4_1_understat_aliases_no_material_improvement"
    elif not material:
        decision = "v4_1_understat_aliases_no_material_improvement"
    else:
        rebuilt = True
        v4.to_csv(OUT_MATRIX_V2, index=False)
        after_cov.to_csv(COVERAGE_V2, index=False)
        missing = pd.DataFrame(
            [
                {
                    "column": col,
                    "missing": int(features[col].isna().sum()),
                    "missing_rate": float(features[col].isna().mean()),
                    "non_null": int(features[col].notna().sum()),
                }
                for col in features.columns
            ]
        )
        missing.to_csv(MISSINGNESS_V2, index=False)
        if coverage_rate(after_cov, "top5_only") >= 0.90 and coverage_rate(after_cov, "top5_test_2020_2025") >= 0.90:
            decision = "v4_1_understat_feature_build_ready_good"
        else:
            decision = "v4_1_understat_feature_build_ready_partial"

    if not rebuilt:
        pd.DataFrame().to_csv(COVERAGE_V2, index=False)
        pd.DataFrame().to_csv(MISSINGNESS_V2, index=False)

    write_outputs(
        decision=decision,
        impact=impact,
        alias_all=alias_all,
        accepted_aliases=accepted_aliases,
        manual_aliases=manual_aliases,
        rejected_aliases=rejected_aliases,
        before_cov=before_cov,
        after_cov=after_cov,
        checks=checks,
        selected_source=selected_source,
        locked_source=locked_source,
        rebuilt=rebuilt,
    )
    key = after_cov[after_cov["segment"].isin(["top5_only", "top5_test_2020_2025", "locked_v3_prediction_row_universe", "locked_v3_selected_bets"])]
    print(
        {
            "decision": decision,
            "rebuilt": rebuilt,
            "accepted_aliases": len(accepted_aliases),
            "coverage_after": {r.segment: round(float(r.both_available_rate), 6) for r in key.itertuples(index=False)},
            "failed_checks": int(checks["status"].ne("pass").sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
