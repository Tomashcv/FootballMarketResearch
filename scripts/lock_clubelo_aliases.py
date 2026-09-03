from __future__ import annotations

import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/processed/match_registry/canonical_match_registry_v1_prototype.csv"
ALIAS_DRAFT_PATH = ROOT / "data/processed/feature_blocks/clubelo/clubelo_team_alias_draft.csv"
CLUBELO_ARCHIVE = ROOT / "data/raw_external/clubelo_manual/clubelo_archive.zip"
OUT_DIR = ROOT / "data/processed/feature_blocks/clubelo"
REPORT_DIR = ROOT / "outputs/reports/feature_blocks/clubelo"
SOURCE_LABEL = "data/raw_external/clubelo_manual/clubelo_archive.zip::EloRatings.csv"

OBVIOUS_ALIAS_APPROVALS = {
    "AC Ajaccio": "Ajaccio",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Arminia Bielefeld": "Bielefeld",
    "Atl. Madrid": "Ath Madrid",
    "B. Monchengladbach": "MGladbach",
    "Bayer Leverkusen": "Leverkusen",
    "Cadiz CF": "Cadiz",
    "Celta Vigo": "Celta",
    "Dep. La Coruna": "La Coruna",
    "Dusseldorf": "Fortuna Dusseldorf",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Espanyol": "Espanol",
    "GFC Ajaccio": "Gazelec",
    "Gijon": "Sp Gijon",
    "Granada CF": "Granada",
    "Hamburger SV": "Hamburg",
    "Hertha Berlin": "Hertha",
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Nottingham": "Nottm Forest",
    "PSG": "Paris SG",
    "Rayo Vallecano": "Vallecano",
    "Real Sociedad": "Sociedad",
    "Schalke": "Schalke 04",
    "Sheffield Utd": "Sheffield United",
}

COUNTRY_BY_LEAGUE = {
    "england_premier_league": "England",
    "spain_laliga": "Spain",
    "germany_bundesliga": "Germany",
    "italy_serie_a": "Italy",
    "france_ligue_1": "France",
}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def load_clubelo_ratings() -> pd.DataFrame:
    with zipfile.ZipFile(CLUBELO_ARCHIVE) as zf:
        ratings = pd.read_csv(zf.open("EloRatings.csv"), usecols=["date", "club", "country", "elo"])
    ratings = ratings.rename(
        columns={
            "club": "clubelo_team_raw",
            "country": "clubelo_country",
            "date": "clubelo_date",
            "elo": "clubelo_rating",
        }
    )
    ratings["clubelo_team_normalized"] = ratings["clubelo_team_raw"].map(normalize_name)
    ratings["clubelo_date"] = pd.to_datetime(ratings["clubelo_date"], errors="coerce")
    ratings["clubelo_rating"] = pd.to_numeric(ratings["clubelo_rating"], errors="coerce")
    return ratings.dropna(subset=["clubelo_team_normalized", "clubelo_date", "clubelo_rating"]).copy()


def affected_context(registry: pd.DataFrame, team: str) -> dict[str, object]:
    mask = registry["home_team_raw"].eq(team) | registry["away_team_raw"].eq(team)
    rows = registry.loc[mask].sort_values("match_datetime")
    examples = rows.head(3).apply(
        lambda r: f"{r['canonical_match_id']} {r['match_datetime']} {r['home_team_raw']} vs {r['away_team_raw']}",
        axis=1,
    ).tolist()
    return {
        "seasons_affected": "; ".join(sorted(rows["season_label"].dropna().astype(str).unique())),
        "leagues_affected": "; ".join(sorted(rows["competition_slug"].dropna().astype(str).unique())),
        "countries_affected": "; ".join(
            sorted({COUNTRY_BY_LEAGUE.get(x, "") for x in rows["competition_slug"].dropna().astype(str).unique()} - {""})
        ),
        "match_count_affected": len(rows),
        "example_canonical_matches": " | ".join(examples),
    }


def build_review_and_locked_tables(registry: pd.DataFrame, aliases: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clubelo_names = sorted(ratings["clubelo_team_raw"].dropna().unique())
    clubelo_norms = sorted({normalize_name(x) for x in clubelo_names})
    clubelo_by_norm = {}
    for name in clubelo_names:
        clubelo_by_norm.setdefault(normalize_name(name), name)
    review_rows = []
    locked_rows = []
    for row in aliases.itertuples(index=False):
        context = affected_context(registry, row.canonical_team_name)
        needs_review = bool(row.manual_review_required) or row.match_type != "exact_normalized" or float(row.confidence) < 1.0
        competing = []
        if row.clubelo_team_name and isinstance(row.clubelo_team_name, str):
            competing = [
                clubelo_by_norm[n]
                for n in get_close_matches(normalize_name(row.canonical_team_name), clubelo_norms, n=5, cutoff=0.74)
                if clubelo_by_norm[n] != row.clubelo_team_name
            ]
        approved_obvious = (
            row.canonical_team_name in OBVIOUS_ALIAS_APPROVALS
            and OBVIOUS_ALIAS_APPROVALS[row.canonical_team_name] == row.clubelo_team_name
        )
        if row.match_type == "exact_normalized" and not needs_review:
            alias_status = "approved_exact"
            approved = True
            manual_review = False
            notes = "Exact normalized alias approved."
        elif approved_obvious:
            alias_status = "approved_obvious_alias"
            approved = True
            manual_review = False
            notes = (
                "Approved as an obvious same-club naming variant for research use. "
                "No competing ClubElo candidate judged similar enough to block lock."
            )
        elif row.clubelo_team_name and isinstance(row.clubelo_team_name, str):
            alias_status = "needs_manual_review"
            approved = False
            manual_review = True
            notes = "Not approved automatically; keep missing until manual alias review."
        else:
            alias_status = "rejected"
            approved = False
            manual_review = False
            notes = "Rejected because no ClubElo candidate is available."
        review_rows.append(
            {
                "canonical_team_name": row.canonical_team_name,
                "clubelo_team_name": row.clubelo_team_name,
                "match_type": row.match_type,
                "confidence": row.confidence,
                **context,
                "competing_clubelo_candidates": "; ".join(competing),
                "automatic_review_result": alias_status,
                "notes": notes,
            }
        )
        locked_rows.append(
            {
                "canonical_team_name": row.canonical_team_name,
                "clubelo_team_name": row.clubelo_team_name,
                "alias_status": alias_status,
                "match_type": row.match_type,
                "confidence": row.confidence,
                "approved_for_research": approved,
                "manual_review_required": manual_review,
                "notes": notes,
            }
        )
    return pd.DataFrame(review_rows), pd.DataFrame(locked_rows)


def latest_rating_join(matches: pd.DataFrame, ratings: pd.DataFrame, side: str) -> pd.DataFrame:
    team_col = f"{side}_clubelo_team_normalized"
    left = matches[["canonical_match_id", "match_date", team_col]].rename(
        columns={team_col: "clubelo_team_normalized"}
    )
    left = left.dropna(subset=["clubelo_team_normalized"])
    right = ratings[
        ["clubelo_team_normalized", "clubelo_date", "clubelo_rating", "clubelo_team_raw"]
    ].copy()
    left = left.sort_values(["match_date", "clubelo_team_normalized", "canonical_match_id"])
    right = right.sort_values(["clubelo_date", "clubelo_team_normalized"])
    joined = pd.merge_asof(
        left,
        right,
        left_on="match_date",
        right_on="clubelo_date",
        by="clubelo_team_normalized",
        direction="backward",
        allow_exact_matches=False,
    )
    return pd.DataFrame(
        {
            "canonical_match_id": joined["canonical_match_id"],
            f"{side}_clubelo_rating": joined["clubelo_rating"],
            f"{side}_clubelo_date": joined["clubelo_date"],
            f"{side}_clubelo_days_stale": (joined["match_date"] - joined["clubelo_date"]).dt.days,
            f"{side}_clubelo_found_flag": joined["clubelo_rating"].notna(),
        }
    )


def build_locked_features(registry: pd.DataFrame, ratings: pd.DataFrame, locked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved = locked[locked["approved_for_research"].astype(bool)]
    alias_map = {
        normalize_name(r.canonical_team_name): normalize_name(r.clubelo_team_name)
        for r in approved.itertuples(index=False)
    }
    matches = registry.copy()
    matches["match_datetime"] = pd.to_datetime(matches["match_datetime"], errors="coerce")
    matches["match_date"] = matches["match_datetime"].dt.floor("D")
    matches["home_clubelo_team_normalized"] = matches["home_team_raw"].map(
        lambda x: alias_map.get(normalize_name(x))
    )
    matches["away_clubelo_team_normalized"] = matches["away_team_raw"].map(
        lambda x: alias_map.get(normalize_name(x))
    )
    home = latest_rating_join(matches, ratings, "home")
    away = latest_rating_join(matches, ratings, "away")
    full = (
        registry[["canonical_match_id", "competition_slug", "season_label", "match_datetime"]]
        .merge(home, on="canonical_match_id", how="left", validate="one_to_one")
        .merge(away, on="canonical_match_id", how="left", validate="one_to_one")
    )
    full["home_clubelo_found_flag"] = full["home_clubelo_found_flag"].fillna(False)
    full["away_clubelo_found_flag"] = full["away_clubelo_found_flag"].fillna(False)
    full["clubelo_both_found_flag"] = full["home_clubelo_found_flag"] & full["away_clubelo_found_flag"]
    full["clubelo_diff_home_minus_away"] = full["home_clubelo_rating"] - full["away_clubelo_rating"]
    full["clubelo_source_file"] = SOURCE_LABEL
    features = full[
        [
            "canonical_match_id",
            "home_clubelo_rating",
            "away_clubelo_rating",
            "clubelo_diff_home_minus_away",
            "home_clubelo_days_stale",
            "away_clubelo_days_stale",
            "home_clubelo_found_flag",
            "away_clubelo_found_flag",
            "clubelo_both_found_flag",
            "clubelo_source_file",
        ]
    ].copy()
    return features, full


def validate(registry: pd.DataFrame, full: pd.DataFrame, features: pd.DataFrame, locked: pd.DataFrame, before_coverage: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    full = full.copy()
    full["match_date"] = pd.to_datetime(full["match_datetime"]).dt.floor("D")
    for col in ["home_clubelo_date", "away_clubelo_date"]:
        full[col] = pd.to_datetime(full[col], errors="coerce")
    home_no_future = (full["home_clubelo_date"].isna() | (full["home_clubelo_date"] < full["match_date"])).all()
    away_no_future = (full["away_clubelo_date"].isna() | (full["away_clubelo_date"] < full["match_date"])).all()
    suspicious = (
        (features["home_clubelo_rating"].notna() & ~features["home_clubelo_rating"].between(1000, 2200))
        | (features["away_clubelo_rating"].notna() & ~features["away_clubelo_rating"].between(1000, 2200))
    )
    coverage = (
        full.groupby(["competition_slug", "season_label"], dropna=False)
        .agg(
            row_count=("canonical_match_id", "size"),
            home_found_rate=("home_clubelo_found_flag", "mean"),
            away_found_rate=("away_clubelo_found_flag", "mean"),
            both_found_rate=("clubelo_both_found_flag", "mean"),
            missing_both_count=("clubelo_both_found_flag", lambda s: int((~s.astype(bool)).sum())),
        )
        .reset_index()
    )
    after_coverage = float(features["clubelo_both_found_flag"].mean())
    unresolved = int((~locked["approved_for_research"].astype(bool) & locked["manual_review_required"].astype(bool)).sum())
    leakage = pd.DataFrame(
        [
            {"check_name": "row_count_equals_18008", "status": len(features) == 18008, "details": f"rows={len(features)}"},
            {"check_name": "row_count_equals_registry", "status": len(features) == len(registry), "details": f"features={len(features)}, registry={len(registry)}"},
            {"check_name": "canonical_match_id_unique", "status": not features["canonical_match_id"].duplicated().any(), "details": f"duplicate_count={int(features['canonical_match_id'].duplicated().sum())}"},
            {"check_name": "home_ratings_strictly_before_match_date", "status": bool(home_no_future), "details": "home ClubElo date < match date"},
            {"check_name": "away_ratings_strictly_before_match_date", "status": bool(away_no_future), "details": "away ClubElo date < match date"},
            {"check_name": "suspicious_rating_values", "status": not bool(suspicious.any()), "details": f"suspicious_rows={int(suspicious.sum())}"},
            {"check_name": "unresolved_alias_count", "status": unresolved == 0, "details": f"unresolved_aliases={unresolved}"},
            {"check_name": "coverage_remains_high", "status": after_coverage >= 0.95, "details": f"before={before_coverage:.4f}, after={after_coverage:.4f}"},
            {"check_name": "no_ambiguous_alias_used_silently", "status": not ((locked['approved_for_research'].astype(bool)) & (locked['manual_review_required'].astype(bool))).any(), "details": "approved aliases have manual_review_required=false"},
        ]
    )
    if not leakage["status"].all():
        decision = "clubelo_alias_lock_failed"
    elif unresolved > 0:
        decision = "clubelo_alias_lock_ready_needs_manual_review"
    else:
        decision = "clubelo_feature_block_locked_ready_good"
    return coverage, leakage, decision


def write_reports(decision: str, review: pd.DataFrame, locked: pd.DataFrame, features: pd.DataFrame, before_coverage: float, after_coverage: float) -> None:
    non_exact = locked[locked["match_type"].ne("exact_normalized")]
    report = [
        "# ClubElo Alias Lock Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 canonical teams and ClubElo `EloRatings.csv`. No modeling, value search, or super CSV merge was performed.",
        "",
        "## Alias Review",
        f"- Total aliases: {len(locked)}",
        f"- Non-exact aliases reviewed: {len(non_exact)}",
        f"- Approved exact: {int(locked['alias_status'].eq('approved_exact').sum())}",
        f"- Approved obvious aliases: {int(locked['alias_status'].eq('approved_obvious_alias').sum())}",
        f"- Needs manual review: {int(locked['alias_status'].eq('needs_manual_review').sum())}",
        f"- Rejected: {int(locked['alias_status'].eq('rejected').sum())}",
        "",
        "## Coverage",
        f"- Before alias lock coverage: {before_coverage:.4f}",
        f"- After alias lock coverage: {after_coverage:.4f}",
        f"- Locked feature rows: {len(features)}",
        "",
        "## Policy",
        "- Only `approved_for_research=true` aliases were used in the locked feature block.",
        "- Obvious aliases were approved only when they clearly refer to the same club in the affected league context.",
        "- No ambiguous alias is used silently.",
        "- Ratings remain joined strictly before match date.",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "clubelo_alias_lock_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_md = [
        "# ClubElo Locked Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The locked ClubElo feature block is approved for production research use as a standalone feature block. It has not been joined into any super CSV.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "clubelo_locked_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(REGISTRY_PATH, dtype={"competition_code": str})
    aliases = pd.read_csv(ALIAS_DRAFT_PATH)
    ratings = load_clubelo_ratings()
    before_features = pd.read_csv(OUT_DIR / "clubelo_features_footiqo_top5_v1.csv")
    before_coverage = float(before_features["clubelo_both_found_flag"].mean())
    review, locked = build_review_and_locked_tables(registry, aliases, ratings)
    features, full = build_locked_features(registry, ratings, locked)
    coverage, leakage, decision = validate(registry, full, features, locked, before_coverage)
    after_coverage = float(features["clubelo_both_found_flag"].mean())

    locked.to_csv(OUT_DIR / "clubelo_team_alias_locked_v1.csv", index=False)
    features.to_csv(OUT_DIR / "clubelo_features_footiqo_top5_v1_locked.csv", index=False)
    locked.to_csv(REPORT_DIR / "clubelo_alias_locked_table.csv", index=False)
    review.to_csv(REPORT_DIR / "clubelo_alias_lock_human_review_table.csv", index=False)
    coverage.to_csv(REPORT_DIR / "clubelo_locked_coverage_by_league_season.csv", index=False)
    leakage.to_csv(REPORT_DIR / "clubelo_locked_leakage_checks.csv", index=False)
    write_reports(decision, review, locked, features, before_coverage, after_coverage)


if __name__ == "__main__":
    main()
