from __future__ import annotations

import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/processed/match_registry/canonical_match_registry_v1_prototype.csv"
CLUBELO_ARCHIVE = ROOT / "data/raw_external/clubelo_manual/clubelo_archive.zip"
OUT_DIR = ROOT / "data/processed/feature_blocks/clubelo"
REPORT_DIR = ROOT / "outputs/reports/feature_blocks/clubelo"
SOURCE_LABEL = "data/raw_external/clubelo_manual/clubelo_archive.zip::EloRatings.csv"

MANUAL_ALIAS_CANDIDATES = {
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
    "Gijon": "Sp Gijon",
    "GFC Ajaccio": "Gazelec",
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


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio(), 4)


def load_clubelo_ratings() -> pd.DataFrame:
    with zipfile.ZipFile(CLUBELO_ARCHIVE) as zf:
        ratings = pd.read_csv(zf.open("EloRatings.csv"), usecols=["date", "club", "elo"])
    ratings = ratings.rename(
        columns={"club": "clubelo_team_raw", "date": "clubelo_date", "elo": "clubelo_rating"}
    )
    ratings["clubelo_team_normalized"] = ratings["clubelo_team_raw"].map(normalize_name)
    ratings["clubelo_date"] = pd.to_datetime(ratings["clubelo_date"], errors="coerce")
    ratings["clubelo_rating"] = pd.to_numeric(ratings["clubelo_rating"], errors="coerce")
    ratings["source_file"] = SOURCE_LABEL
    return ratings.dropna(subset=["clubelo_team_normalized", "clubelo_date", "clubelo_rating"]).copy()


def build_alias_draft(registry: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    canonical_teams = sorted(set(registry["home_team_raw"]).union(set(registry["away_team_raw"])))
    clubelo_names = sorted(ratings["clubelo_team_raw"].dropna().unique())
    clubelo_by_norm = {}
    for name in clubelo_names:
        clubelo_by_norm.setdefault(normalize_name(name), name)
    clubelo_norms = sorted(clubelo_by_norm)
    rows = []
    for team in canonical_teams:
        canonical_norm = normalize_name(team)
        if canonical_norm in clubelo_by_norm:
            clubelo = clubelo_by_norm[canonical_norm]
            rows.append(
                {
                    "canonical_team_name": team,
                    "clubelo_team_name": clubelo,
                    "match_type": "exact_normalized",
                    "confidence": 1.0,
                    "manual_review_required": False,
                    "notes": "Exact match after deterministic normalization.",
                }
            )
            continue
        candidate = MANUAL_ALIAS_CANDIDATES.get(team)
        if candidate and normalize_name(candidate) in clubelo_by_norm:
            clubelo = clubelo_by_norm[normalize_name(candidate)]
            rows.append(
                {
                    "canonical_team_name": team,
                    "clubelo_team_name": clubelo,
                    "match_type": "conservative_alias_candidate",
                    "confidence": max(0.86, similarity(team, clubelo)),
                    "manual_review_required": True,
                    "notes": "High-confidence naming variant candidate; used in prototype feature block and requires manual alias review before final registry use.",
                }
            )
            continue
        close = get_close_matches(canonical_norm, clubelo_norms, n=1, cutoff=0.84)
        if close:
            clubelo = clubelo_by_norm[close[0]]
            rows.append(
                {
                    "canonical_team_name": team,
                    "clubelo_team_name": clubelo,
                    "match_type": "fuzzy_candidate",
                    "confidence": similarity(team, clubelo),
                    "manual_review_required": True,
                    "notes": "Fuzzy candidate only; requires manual review.",
                }
            )
        else:
            rows.append(
                {
                    "canonical_team_name": team,
                    "clubelo_team_name": "",
                    "match_type": "unmatched",
                    "confidence": 0.0,
                    "manual_review_required": True,
                    "notes": "No conservative ClubElo alias candidate found.",
                }
            )
    return pd.DataFrame(rows)


def latest_rating_join(matches: pd.DataFrame, ratings: pd.DataFrame, side: str) -> pd.DataFrame:
    team_col = f"{side}_clubelo_team_normalized"
    left = matches[
        ["canonical_match_id", "match_datetime", "match_date", team_col]
    ].copy()
    left = left.rename(columns={team_col: "clubelo_team_normalized"})
    left = left.dropna(subset=["clubelo_team_normalized"])
    right = ratings[
        [
            "clubelo_team_normalized",
            "clubelo_team_raw",
            "clubelo_date",
            "clubelo_rating",
            "source_file",
        ]
    ].copy()
    # Pandas requires a globally sorted asof key.
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
    joined[f"{side}_clubelo_rating"] = joined["clubelo_rating"]
    joined[f"{side}_clubelo_date"] = joined["clubelo_date"]
    joined[f"{side}_clubelo_team_raw"] = joined["clubelo_team_raw"]
    joined[f"{side}_clubelo_source_file"] = joined["source_file"]
    joined[f"{side}_clubelo_found_flag"] = joined[f"{side}_clubelo_rating"].notna()
    joined[f"{side}_clubelo_days_stale"] = (
        joined["match_date"] - joined[f"{side}_clubelo_date"]
    ).dt.days
    return joined[
        [
            "canonical_match_id",
            f"{side}_clubelo_rating",
            f"{side}_clubelo_date",
            f"{side}_clubelo_team_raw",
            f"{side}_clubelo_days_stale",
            f"{side}_clubelo_found_flag",
            f"{side}_clubelo_source_file",
        ]
    ]


def build_features(registry: pd.DataFrame, ratings: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        normalize_name(row.canonical_team_name): normalize_name(row.clubelo_team_name)
        for row in aliases.itertuples()
        if isinstance(row.clubelo_team_name, str) and row.clubelo_team_name
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
    features = (
        registry[["canonical_match_id", "competition_slug", "season_label", "match_datetime"]]
        .merge(home, on="canonical_match_id", how="left", validate="one_to_one")
        .merge(away, on="canonical_match_id", how="left", validate="one_to_one")
    )
    features["home_clubelo_found_flag"] = features["home_clubelo_found_flag"].fillna(False)
    features["away_clubelo_found_flag"] = features["away_clubelo_found_flag"].fillna(False)
    features["clubelo_both_found_flag"] = (
        features["home_clubelo_found_flag"] & features["away_clubelo_found_flag"]
    )
    features["clubelo_diff_home_minus_away"] = (
        features["home_clubelo_rating"] - features["away_clubelo_rating"]
    )
    features["clubelo_source_file"] = SOURCE_LABEL
    out = features[
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
            "home_clubelo_date",
            "away_clubelo_date",
            "home_clubelo_team_raw",
            "away_clubelo_team_raw",
        ]
    ].copy()
    out["home_clubelo_date"] = pd.to_datetime(out["home_clubelo_date"]).dt.strftime("%Y-%m-%d")
    out["away_clubelo_date"] = pd.to_datetime(out["away_clubelo_date"]).dt.strftime("%Y-%m-%d")
    return out


def validation_reports(registry: pd.DataFrame, features: pd.DataFrame, aliases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    merged = registry[
        ["canonical_match_id", "competition_slug", "season_label", "match_datetime"]
    ].merge(features, on="canonical_match_id", how="left", validate="one_to_one")
    merged["match_date"] = pd.to_datetime(merged["match_datetime"]).dt.floor("D")
    for col in ["home_clubelo_date", "away_clubelo_date"]:
        merged[col] = pd.to_datetime(merged[col], errors="coerce")
    home_no_future = (merged["home_clubelo_date"].isna() | (merged["home_clubelo_date"] < merged["match_date"])).all()
    away_no_future = (merged["away_clubelo_date"].isna() | (merged["away_clubelo_date"] < merged["match_date"])).all()
    suspicious_home = merged["home_clubelo_rating"].notna() & ~merged["home_clubelo_rating"].between(1000, 2200)
    suspicious_away = merged["away_clubelo_rating"].notna() & ~merged["away_clubelo_rating"].between(1000, 2200)
    coverage = (
        merged.groupby(["competition_slug", "season_label"], dropna=False)
        .agg(
            row_count=("canonical_match_id", "size"),
            home_found_rate=("home_clubelo_found_flag", "mean"),
            away_found_rate=("away_clubelo_found_flag", "mean"),
            both_found_rate=("clubelo_both_found_flag", "mean"),
            missing_both_count=("clubelo_both_found_flag", lambda s: int((~s.astype(bool)).sum())),
        )
        .reset_index()
    )
    stale_values = pd.concat(
        [
            merged[["competition_slug", "season_label", "home_clubelo_days_stale"]].rename(
                columns={"home_clubelo_days_stale": "days_stale"}
            ).assign(side="home"),
            merged[["competition_slug", "season_label", "away_clubelo_days_stale"]].rename(
                columns={"away_clubelo_days_stale": "days_stale"}
            ).assign(side="away"),
        ],
        ignore_index=True,
    ).dropna(subset=["days_stale"])
    staleness = (
        stale_values.groupby("side")
        .agg(
            count=("days_stale", "size"),
            min_days=("days_stale", "min"),
            p50_days=("days_stale", "median"),
            p95_days=("days_stale", lambda s: s.quantile(0.95)),
            max_days=("days_stale", "max"),
        )
        .reset_index()
    )
    leakage = pd.DataFrame(
        [
            {
                "check_name": "row_count_equals_registry",
                "status": len(features) == len(registry),
                "details": f"features={len(features)}, registry={len(registry)}",
            },
            {
                "check_name": "canonical_match_id_unique",
                "status": not features["canonical_match_id"].duplicated().any(),
                "details": f"duplicate_count={int(features['canonical_match_id'].duplicated().sum())}",
            },
            {
                "check_name": "home_ratings_strictly_before_match_date",
                "status": bool(home_no_future),
                "details": "latest ClubElo date must be < match date",
            },
            {
                "check_name": "away_ratings_strictly_before_match_date",
                "status": bool(away_no_future),
                "details": "latest ClubElo date must be < match date",
            },
            {
                "check_name": "suspicious_rating_values",
                "status": not bool(suspicious_home.any() or suspicious_away.any()),
                "details": f"home={int(suspicious_home.sum())}, away={int(suspicious_away.sum())}",
            },
            {
                "check_name": "alias_manual_review_count",
                "status": int(aliases["manual_review_required"].sum()) == 0,
                "details": f"manual_review_required={int(aliases['manual_review_required'].sum())}",
            },
        ]
    )
    high_coverage = bool(merged["clubelo_both_found_flag"].mean() >= 0.95)
    leakage_pass = bool(leakage.loc[~leakage["check_name"].eq("alias_manual_review_count"), "status"].all())
    if not leakage_pass or len(features) != len(registry):
        decision = "clubelo_feature_block_failed"
    elif int(aliases["manual_review_required"].sum()) > 0 or not high_coverage:
        decision = "clubelo_feature_block_ready_needs_alias_review"
    else:
        decision = "clubelo_feature_block_ready_good"
    return coverage, staleness, leakage, merged, decision


def write_markdown(decision: str, registry: pd.DataFrame, features: pd.DataFrame, aliases: pd.DataFrame, coverage: pd.DataFrame) -> None:
    coverage_rate = features["clubelo_both_found_flag"].mean()
    report = [
        "# ClubElo Feature Block Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Source used: `EloRatings.csv` inside the manual ClubElo archive. `Matches.csv` was not used as a predictive feature source.",
        "",
        "## Build Policy",
        "- Ratings are joined using the latest ClubElo date strictly before `match_datetime` date.",
        "- Same-day ratings are not used because the ratings source has dates only.",
        "- Unmatched rows are retained with missing ratings and found flags.",
        "- Non-exact aliases are draft candidates and require manual review before final use.",
        "",
        "## Counts",
        f"- Registry rows: {len(registry)}",
        f"- Feature rows: {len(features)}",
        f"- Both-team coverage: {coverage_rate:.4f}",
        f"- Alias rows needing manual review: {int(aliases['manual_review_required'].sum())}",
        "",
        "## Conservative Notes",
        "- This is a feature block, not a final merged super CSV.",
        "- No modeling, value search, or edge claim was performed.",
    ]
    (REPORT_DIR / "clubelo_feature_block_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_md = [
        "# ClubElo Feature Block Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The feature block is date-safe under the strict-before-match-date rule. Because non-exact ClubElo team aliases are used as draft candidates, this remains subject to alias review before final production use.",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "clubelo_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(REGISTRY_PATH, dtype={"competition_code": str})
    ratings = load_clubelo_ratings()
    aliases = build_alias_draft(registry, ratings)
    features_full = build_features(registry, ratings, aliases)
    coverage, staleness, leakage, merged, decision = validation_reports(registry, features_full, aliases)
    features = features_full[
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
    aliases.to_csv(OUT_DIR / "clubelo_team_alias_draft.csv", index=False)
    aliases.to_csv(REPORT_DIR / "clubelo_alias_review.csv", index=False)
    features.to_csv(OUT_DIR / "clubelo_features_footiqo_top5_v1.csv", index=False)
    coverage.to_csv(REPORT_DIR / "clubelo_coverage_by_league_season.csv", index=False)
    staleness.to_csv(REPORT_DIR / "clubelo_staleness_summary.csv", index=False)
    leakage.to_csv(REPORT_DIR / "clubelo_leakage_checks.csv", index=False)
    write_markdown(decision, registry, features_full, aliases, coverage)


if __name__ == "__main__":
    main()
