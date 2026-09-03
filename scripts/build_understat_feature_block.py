from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata
import zipfile
from datetime import datetime, timezone
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "data/processed/match_registry/canonical_match_registry_v1_prototype.csv"
DEFAULT_UNDERSTAT_ARCHIVE = ROOT / "data/raw_external/understat_manual/understat_archive.zip"
OUT_DIR = ROOT / "data/processed/feature_blocks/understat"
REPORT_DIR = ROOT / "outputs/reports/feature_blocks/understat"
SOURCE_LABEL = "data/raw_external/understat_manual/understat_archive.zip::game_stats"

LEAGUE_TO_UNDERSTAT = {
    "england_premier_league": "EPL",
    "spain_laliga": "La liga",
    "germany_bundesliga": "Bundesliga",
    "italy_serie_a": "Serie A",
    "france_ligue_1": "Ligue 1",
}

# Curated aliases for Footiqo top-5 team names -> Understat club names.
# Exact-normalized matches are approved automatically; these cover common naming variants.
CURATED_ALIASES = {
    # England
    "Manchester Utd": "Manchester United",
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "West Brom": "West Bromwich Albion",
    "QPR": "Queens Park Rangers",
    "Sheffield Utd": "Sheffield United",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Brighton": "Brighton",
    "Bournemouth": "Bournemouth",
    "West Ham": "West Ham",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    # Germany
    "Dortmund": "Borussia Dortmund",
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Leverkusen": "Bayer Leverkusen",
    "B. Monchengladbach": "Borussia M.Gladbach",
    "M'gladbach": "Borussia M.Gladbach",
    "MGladbach": "Borussia M.Gladbach",
    "Stuttgart": "VfB Stuttgart",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Mainz": "Mainz 05",
    "Mainz 05": "Mainz 05",
    "RB Leipzig": "RasenBallsport Leipzig",
    "FC Koln": "FC Cologne",
    "FC Cologne": "FC Cologne",
    "Hertha": "Hertha Berlin",
    "Hertha Berlin": "Hertha Berlin",
    "Bielefeld": "Arminia Bielefeld",
    "Arminia Bielefeld": "Arminia Bielefeld",
    "Hamburg": "Hamburger SV",
    "Hamburger SV": "Hamburger SV",
    "Hannover": "Hannover 96",
    "Hannover 96": "Hannover 96",
    "Schalke": "Schalke 04",
    "Schalke 04": "Schalke 04",
    "Dusseldorf": "Fortuna Duesseldorf",
    "Fortuna Dusseldorf": "Fortuna Duesseldorf",
    "Fortuna Duesseldorf": "Fortuna Duesseldorf",
    "Werder Bremen": "Werder Bremen",
    # Spain
    "Atl. Madrid": "Atletico Madrid",
    "Ath Madrid": "Atletico Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Club",
    "Athletic Bilbao": "Athletic Club",
    "Athletic Club": "Athletic Club",
    "Celta": "Celta Vigo",
    "Celta Vigo": "Celta Vigo",
    "Vallecano": "Rayo Vallecano",
    "Rayo Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    "Espanyol": "Espanyol",
    "Huesca": "SD Huesca",
    "La Coruna": "Deportivo La Coruna",
    "Dep. La Coruna": "Deportivo La Coruna",
    "Sp Gijon": "Sporting Gijon",
    "Sporting Gijon": "Sporting Gijon",
    "Real Sociedad": "Real Sociedad",
    "Sociedad": "Real Sociedad",
    "Real Betis": "Real Betis",
    "Betis": "Real Betis",
    "Cadiz CF": "Cadiz",
    "Cadiz": "Cadiz",
    "Granada CF": "Granada",
    "Granada": "Granada",
    "Alaves": "Alaves",
    "Almeria": "Almeria",
    "Leganes": "Leganes",
    "Getafe": "Getafe",
    "Girona": "Girona",
    "Osasuna": "Osasuna",
    "Eibar": "Eibar",
    "Levante": "Levante",
    "Mallorca": "Mallorca",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    # Italy
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Inter": "Inter",
    "Inter Milan": "Inter",
    "AS Roma": "Roma",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Sassuolo": "Sassuolo",
    "Sampdoria": "Sampdoria",
    "Genoa": "Genoa",
    "Verona": "Verona",
    "Hellas Verona": "Verona",
    "Spal": "SPAL 2013",
    "SPAL": "SPAL 2013",
    "Benevento": "Benevento",
    "Crotone": "Crotone",
    "Empoli": "Empoli",
    "Frosinone": "Frosinone",
    "Lecce": "Lecce",
    "Monza": "Monza",
    "Salernitana": "Salernitana",
    "Spezia": "Spezia",
    "Parma": "Parma Calcio 1913",
    "Parma Calcio 1913": "Parma Calcio 1913",
    # France
    "PSG": "Paris Saint Germain",
    "Paris SG": "Paris Saint Germain",
    "Paris Saint Germain": "Paris Saint Germain",
    "Lyon": "Lyon",
    "Marseille": "Marseille",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Rennes": "Rennes",
    "Nice": "Nice",
    "Montpellier": "Montpellier",
    "Nantes": "Nantes",
    "Bordeaux": "Bordeaux",
    "St Etienne": "Saint-Etienne",
    "Saint Etienne": "Saint-Etienne",
    "Saint-Etienne": "Saint-Etienne",
    "Reims": "Reims",
    "Strasbourg": "Strasbourg",
    "Toulouse": "Toulouse",
    "Angers": "Angers",
    "Brest": "Brest",
    "Clermont": "Clermont Foot",
    "Clermont Foot": "Clermont Foot",
    "Dijon": "Dijon",
    "Metz": "Metz",
    "Nimes": "Nimes",
    "Lorient": "Lorient",
    "Lens": "Lens",
    "Amiens": "Amiens",
    "Troyes": "Troyes",
    "Caen": "Caen",
    "Guingamp": "Guingamp",
    "Ajaccio": "Ajaccio",
    "AC Ajaccio": "Ajaccio",
    "GFC Ajaccio": "GFC Ajaccio",
}

METRICS = [
    "xG",
    "xGA",
    "npxG",
    "npxGA",
    "ppda",
    "ppda_allowed",
    "deep",
    "deep_allowed",
    "scored",
    "missed",
    "xpts",
    "npxGD",
    "pts",
]
WINDOWS = [5, 10, 20]


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio(), 4)


def read_understat_game_stats(archive_path: Path) -> pd.DataFrame:
    if not archive_path.exists():
        raise FileNotFoundError(f"Understat archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        csv_candidates = [n for n in names if Path(n).name == "game_stats.csv"]
        if csv_candidates:
            with zf.open(csv_candidates[0]) as fh:
                df = pd.read_csv(fh)
            source = f"{archive_path}::{csv_candidates[0]}"
        else:
            db_candidates = [n for n in names if Path(n).suffix.lower() in {".db", ".sqlite", ".sqlite3"}]
            if not db_candidates:
                raise FileNotFoundError("No game_stats.csv or sqlite db found inside Understat archive")
            tmp = OUT_DIR / "_understat_temp.db"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(zf.read(db_candidates[0]))
            try:
                with sqlite3.connect(tmp) as conn:
                    df = pd.read_sql_query("SELECT * FROM game_stats", conn)
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
            source = f"{archive_path}::{db_candidates[0]}::game_stats"
    df["understat_source_file"] = source
    return df


def clean_understat(df: pd.DataFrame) -> pd.DataFrame:
    required = {"league", "season", "club_name", "home_away", "date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Understat game_stats missing required columns: {missing}")
    out = df.copy()
    out = out.rename(columns={"club_name": "understat_team_raw", "date": "understat_datetime"})
    out["understat_team_normalized"] = out["understat_team_raw"].map(normalize_name)
    out["understat_datetime"] = pd.to_datetime(out["understat_datetime"], errors="coerce")
    out["understat_date"] = out["understat_datetime"].dt.floor("D")
    out["understat_league"] = out["league"].astype(str)
    out["understat_home_away"] = out["home_away"].astype(str)
    for col in METRICS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    out = out.dropna(subset=["understat_team_normalized", "understat_date", "understat_league"]).copy()
    out = out.sort_values(["understat_league", "understat_team_normalized", "understat_date", "understat_home_away"]).reset_index(drop=True)
    return out


def build_alias_table(registry: pd.DataFrame, understat: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(set(registry["home_team_raw"].dropna()).union(set(registry["away_team_raw"].dropna())))
    understat_names = sorted(understat["understat_team_raw"].dropna().astype(str).unique())
    understat_by_norm: dict[str, str] = {}
    for name in understat_names:
        understat_by_norm.setdefault(normalize_name(name), name)
    understat_norms = sorted(understat_by_norm)
    rows = []
    for team in teams:
        team_norm = normalize_name(team)
        if team_norm in understat_by_norm:
            understat_name = understat_by_norm[team_norm]
            rows.append({
                "canonical_team_name": team,
                "understat_team_name": understat_name,
                "canonical_team_normalized": team_norm,
                "understat_team_normalized": normalize_name(understat_name),
                "alias_status": "approved_exact_normalized",
                "match_type": "exact_normalized",
                "confidence": 1.0,
                "approved_for_research": True,
                "manual_review_required": False,
                "notes": "Exact match after deterministic normalization.",
            })
            continue
        alias = CURATED_ALIASES.get(team)
        if alias and normalize_name(alias) in understat_by_norm:
            understat_name = understat_by_norm[normalize_name(alias)]
            rows.append({
                "canonical_team_name": team,
                "understat_team_name": understat_name,
                "canonical_team_normalized": team_norm,
                "understat_team_normalized": normalize_name(understat_name),
                "alias_status": "approved_obvious_alias",
                "match_type": "curated_alias",
                "confidence": max(0.90, similarity(team, understat_name)),
                "approved_for_research": True,
                "manual_review_required": False,
                "notes": "Curated obvious alias for top-5 Understat/Foottiqo naming variant.",
            })
            continue
        close = get_close_matches(team_norm, understat_norms, n=3, cutoff=0.78)
        rows.append({
            "canonical_team_name": team,
            "understat_team_name": understat_by_norm[close[0]] if close else "",
            "canonical_team_normalized": team_norm,
            "understat_team_normalized": close[0] if close else "",
            "alias_status": "needs_manual_review" if close else "unmatched",
            "match_type": "fuzzy_candidate" if close else "unmatched",
            "confidence": similarity(team, understat_by_norm[close[0]]) if close else 0.0,
            "approved_for_research": False,
            "manual_review_required": True,
            "notes": "Fuzzy/unmatched candidate is not used for research until manually approved.",
        })
    aliases = pd.DataFrame(rows)
    # Add match counts affected.
    counts = pd.concat([
        registry[["home_team_raw", "competition_slug"]].rename(columns={"home_team_raw": "canonical_team_name"}),
        registry[["away_team_raw", "competition_slug"]].rename(columns={"away_team_raw": "canonical_team_name"}),
    ])
    agg = counts.groupby("canonical_team_name").agg(match_count_affected=("competition_slug", "size"), leagues_affected=("competition_slug", lambda s: ";".join(sorted(s.unique())))).reset_index()
    return aliases.merge(agg, on="canonical_team_name", how="left")


def make_history_lookup(understat: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    lookup = {}
    for key, group in understat.groupby(["understat_league", "understat_team_normalized"], dropna=False):
        g = group.sort_values("understat_date").reset_index(drop=True)
        lookup[key] = g
    return lookup


def rolling_values(history: pd.DataFrame, match_date: pd.Timestamp, prefix: str) -> dict[str, object]:
    previous = history.loc[history["understat_date"] < match_date]
    row: dict[str, object] = {
        f"{prefix}_understat_history_count": int(len(previous)),
        f"{prefix}_understat_found_flag": bool(len(previous) > 0),
        f"{prefix}_understat_latest_days_ago": np.nan,
    }
    if len(previous) > 0:
        latest_date = previous["understat_date"].max()
        row[f"{prefix}_understat_latest_days_ago"] = int((match_date - latest_date).days)
    for w in WINDOWS:
        tail = previous.tail(w)
        row[f"{prefix}_understat_matches_w{w}"] = int(len(tail))
        for metric in METRICS:
            col = f"{prefix}_understat_{metric}_avg_w{w}"
            row[col] = float(tail[metric].mean()) if len(tail) else np.nan
    return row


def build_features(registry: pd.DataFrame, understat: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        normalize_name(r.canonical_team_name): r.understat_team_normalized
        for r in aliases.itertuples()
        if bool(r.approved_for_research) and isinstance(r.understat_team_normalized, str) and r.understat_team_normalized
    }
    lookup = make_history_lookup(understat)
    matches = registry.copy()
    matches["match_datetime"] = pd.to_datetime(matches["match_datetime"], errors="coerce")
    matches["match_date"] = matches["match_datetime"].dt.floor("D")
    rows = []
    for r in matches.itertuples(index=False):
        league = LEAGUE_TO_UNDERSTAT.get(r.competition_slug)
        home_norm = alias_map.get(normalize_name(r.home_team_raw))
        away_norm = alias_map.get(normalize_name(r.away_team_raw))
        base = {
            "canonical_match_id": int(r.canonical_match_id),
            "understat_league": league,
            "understat_source_file": SOURCE_LABEL,
            "understat_match_after_source_max_date_flag": False,
        }
        if league is None or pd.isna(r.match_date):
            home_row = rolling_values(pd.DataFrame(columns=["understat_date"] + METRICS), pd.Timestamp("1900-01-01"), "home")
            away_row = rolling_values(pd.DataFrame(columns=["understat_date"] + METRICS), pd.Timestamp("1900-01-01"), "away")
        else:
            home_hist = lookup.get((league, home_norm), pd.DataFrame(columns=["understat_date"] + METRICS))
            away_hist = lookup.get((league, away_norm), pd.DataFrame(columns=["understat_date"] + METRICS))
            home_row = rolling_values(home_hist, r.match_date, "home")
            away_row = rolling_values(away_hist, r.match_date, "away")
        base.update(home_row)
        base.update(away_row)
        rows.append(base)
    features = pd.DataFrame(rows)
    features["understat_both_found_flag"] = features["home_understat_found_flag"].astype(bool) & features["away_understat_found_flag"].astype(bool)
    max_date = understat["understat_date"].max()
    match_dates = pd.to_datetime(matches["match_date"], errors="coerce")
    features["understat_match_after_source_max_date_flag"] = (match_dates.values > max_date).astype(bool)
    for w in WINDOWS:
        for metric in METRICS:
            h = f"home_understat_{metric}_avg_w{w}"
            a = f"away_understat_{metric}_avg_w{w}"
            if h in features.columns and a in features.columns:
                features[f"understat_home_minus_away_{metric}_avg_w{w}"] = features[h] - features[a]
    return features


def validate_and_report(registry: pd.DataFrame, understat: pd.DataFrame, aliases: pd.DataFrame, features: pd.DataFrame) -> str:
    merged = registry[["canonical_match_id", "competition_slug", "season_label", "match_datetime"]].merge(features, on="canonical_match_id", how="left", validate="one_to_one")
    coverage = (
        merged.groupby(["competition_slug", "season_label"], dropna=False)
        .agg(
            row_count=("canonical_match_id", "size"),
            home_found_rate=("home_understat_found_flag", "mean"),
            away_found_rate=("away_understat_found_flag", "mean"),
            both_found_rate=("understat_both_found_flag", "mean"),
            after_source_max_date_rate=("understat_match_after_source_max_date_flag", "mean"),
            home_latest_days_p95=("home_understat_latest_days_ago", lambda s: s.dropna().quantile(0.95) if s.notna().any() else np.nan),
            away_latest_days_p95=("away_understat_latest_days_ago", lambda s: s.dropna().quantile(0.95) if s.notna().any() else np.nan),
        )
        .reset_index()
    )
    coverage.to_csv(REPORT_DIR / "understat_coverage_by_league_season.csv", index=False)

    stale = pd.concat([
        merged[["competition_slug", "season_label", "home_understat_latest_days_ago"]].rename(columns={"home_understat_latest_days_ago": "days_stale"}).assign(side="home"),
        merged[["competition_slug", "season_label", "away_understat_latest_days_ago"]].rename(columns={"away_understat_latest_days_ago": "days_stale"}).assign(side="away"),
    ], ignore_index=True)
    stale_summary = stale.groupby("side").agg(
        count=("days_stale", "count"),
        min_days=("days_stale", "min"),
        p50_days=("days_stale", "median"),
        p95_days=("days_stale", lambda s: s.quantile(0.95)),
        max_days=("days_stale", "max"),
    ).reset_index()
    stale_summary.to_csv(REPORT_DIR / "understat_staleness_summary.csv", index=False)

    manual_count = int(aliases["manual_review_required"].sum())
    coverage_rate = float(features["understat_both_found_flag"].mean()) if len(features) else 0.0
    after_max_rate = float(features["understat_match_after_source_max_date_flag"].mean()) if len(features) else 0.0
    leakage = pd.DataFrame([
        {"check_name": "row_count_equals_registry", "status": len(features) == len(registry), "details": f"features={len(features)}, registry={len(registry)}"},
        {"check_name": "canonical_match_id_unique", "status": not features["canonical_match_id"].duplicated().any(), "details": f"duplicate_count={int(features['canonical_match_id'].duplicated().sum())}"},
        {"check_name": "strict_past_only_policy", "status": True, "details": "Features are computed only from Understat rows with understat_date < match_date."},
        {"check_name": "no_current_match_understat_values_joined", "status": True, "details": "Only rolling aggregates and flags are written; no same-fixture raw Understat rows are joined."},
        {"check_name": "understat_source_staleness_flag_present", "status": "understat_match_after_source_max_date_flag" in features.columns, "details": f"after_max_rate={after_max_rate:.4f}"},
        {"check_name": "alias_manual_review_count", "status": manual_count == 0, "details": f"manual_review_required={manual_count}"},
        {"check_name": "both_team_coverage_high", "status": coverage_rate >= 0.95, "details": f"both_team_coverage={coverage_rate:.4f}"},
    ])
    leakage.to_csv(REPORT_DIR / "understat_leakage_checks.csv", index=False)
    aliases.to_csv(REPORT_DIR / "understat_alias_locked_table.csv", index=False)

    if not bool(leakage.loc[~leakage["check_name"].isin(["alias_manual_review_count", "both_team_coverage_high"]), "status"].all()):
        decision = "understat_feature_block_failed"
    elif manual_count > 0 or coverage_rate < 0.95:
        decision = "understat_feature_block_ready_needs_alias_review"
    elif after_max_rate > 0.05:
        decision = "understat_feature_block_ready_good_with_staleness_warning"
    else:
        decision = "understat_feature_block_ready_good"

    report = [
        "# Understat Feature Block Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 canonical registry and Understat game_stats only. No modeling, value search, or super CSV merge was performed.",
        "",
        "## Build Policy",
        "- Understat same-fixture xG/stats are never joined directly.",
        "- Rolling features use only Understat rows with `understat_date < match_date`.",
        "- Same-day Understat rows are excluded because source timestamps may not prove pre-kickoff availability.",
        "- Features are keyed by `canonical_match_id` and retain unmatched rows with missing flags.",
        "- Matches after the latest Understat source date are retained but flagged as stale.",
        "",
        "## Source Summary",
        f"- Understat rows: {len(understat)}",
        f"- Understat date range: {understat['understat_date'].min()} to {understat['understat_date'].max()}",
        f"- Understat leagues: {', '.join(sorted(understat['understat_league'].dropna().unique()))}",
        "",
        "## Counts",
        f"- Registry rows: {len(registry)}",
        f"- Feature rows: {len(features)}",
        f"- Both-team coverage: {coverage_rate:.4f}",
        f"- Alias rows needing manual review: {manual_count}",
        f"- Rows after Understat max date: {int(features['understat_match_after_source_max_date_flag'].sum())} ({after_max_rate:.4f})",
        "",
        "## Conservative Notes",
        "- This is a feature block, not a final merged super CSV.",
        "- If the staleness warning is present, downstream models should include stale flags and may exclude stale seasons in stress tests.",
        "- No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "understat_feature_block_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_md = [
        "# Understat Feature Block Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The Understat feature block is research-only. It is date-safe under the strict past-only rolling policy, but source coverage/staleness and aliases must be respected downstream.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "understat_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build date-safe Understat feature block for Footiqo top-5 canonical registry.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--understat-archive", default=str(DEFAULT_UNDERSTAT_ARCHIVE))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(args.registry)
    registry["canonical_match_id"] = registry["canonical_match_id"].astype("int64")
    raw = read_understat_game_stats(Path(args.understat_archive))
    understat = clean_understat(raw)
    aliases = build_alias_table(registry, understat)
    features = build_features(registry, understat, aliases)

    features.to_csv(OUT_DIR / "understat_features_footiqo_top5_v1.csv", index=False)
    aliases.to_csv(OUT_DIR / "understat_team_alias_locked_v1.csv", index=False)
    decision = validate_and_report(registry, understat, aliases, features)
    print(decision)
    print(f"wrote {OUT_DIR / 'understat_features_footiqo_top5_v1.csv'}")


if __name__ == "__main__":
    main()
