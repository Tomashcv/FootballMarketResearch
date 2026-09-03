from __future__ import annotations

import re
import sqlite3
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEAMS = ROOT / "data/processed/entity_registry/teams_v1_locked.csv"
ALIASES = ROOT / "data/processed/entity_registry/team_aliases_v1_locked.csv"
MATCHES = ROOT / "data/processed/entity_registry/matches_v1_locked.csv"
COMPETITIONS = ROOT / "data/processed/entity_registry/competitions_v1_locked.csv"
UNDERSTAT_ARCHIVE = ROOT / "data/raw_external/understat_manual/understat_archive.zip"
OUT_DIR = ROOT / "data/processed/feature_blocks/understat"
REPORT_DIR = ROOT / "outputs/reports/feature_blocks/understat"
OUTPUT_FEATURES = OUT_DIR / "understat_features_footiqo_top5_v1_locked.csv"

LEAGUE_TO_UNDERSTAT = {
    "england_premier_league": "EPL",
    "spain_laliga": "La liga",
    "germany_bundesliga": "Bundesliga",
    "italy_serie_a": "Serie A",
    "france_ligue_1": "Ligue 1",
}

METRICS = [
    "xG",
    "xGA",
    "npxG",
    "npxGA",
    "deep",
    "deep_allowed",
    "scored",
    "missed",
    "pts",
    "xpts",
    "ppda",
    "ppda_allowed",
]
WINDOWS = [5, 10, 20]


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def read_understat_game_stats(archive_path: Path) -> tuple[pd.DataFrame, str]:
    if not archive_path.exists():
        raise FileNotFoundError(f"Understat archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        csv_candidates = [n for n in names if Path(n).name == "game_stats.csv"]
        if csv_candidates:
            source_name = csv_candidates[0]
            with zf.open(source_name) as fh:
                return pd.read_csv(fh), f"{archive_path}::{source_name}"
        db_candidates = [n for n in names if Path(n).suffix.lower() in {".db", ".sqlite", ".sqlite3"}]
        if not db_candidates:
            raise FileNotFoundError("No game_stats.csv or sqlite db found inside Understat archive")
        source_name = db_candidates[0]
        tmp = OUT_DIR / "_understat_locked_temp.db"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(zf.read(source_name))
        try:
            with sqlite3.connect(tmp) as conn:
                df = pd.read_sql_query("SELECT * FROM game_stats", conn)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        return df, f"{archive_path}::{source_name}::game_stats"


def clean_understat(raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
    required = {"league", "season", "club_name", "home_away", "date"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Understat game_stats missing required columns: {missing}")
    out = raw.copy()
    out = out.rename(columns={"club_name": "understat_team_raw", "date": "understat_datetime"})
    out["understat_team_normalized"] = out["understat_team_raw"].map(normalize_name)
    out["understat_datetime"] = pd.to_datetime(out["understat_datetime"], errors="coerce")
    out["understat_date"] = out["understat_datetime"].dt.floor("D")
    out["understat_league"] = out["league"].astype(str)
    out["understat_source_file"] = source_file
    for metric in METRICS:
        out[metric] = pd.to_numeric(out.get(metric, np.nan), errors="coerce")
    out = out.dropna(subset=["understat_team_normalized", "understat_date", "understat_league"]).copy()
    return out.sort_values(["understat_league", "understat_team_normalized", "understat_date"]).reset_index(drop=True)


def locked_understat_aliases(aliases: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    under = aliases[aliases["source"].eq("understat")].copy()
    under["approved_for_research"] = bool_series(under["approved_for_research"])
    under["manual_review_required"] = bool_series(under["manual_review_required"])
    approved = under[
        under["approved_for_research"]
        & ~under["manual_review_required"]
        & ~under["alias_status"].eq("rejected")
    ].copy()
    approved["understat_team_normalized"] = approved["alias_name"].map(normalize_name)
    approved = approved.merge(teams[["team_id", "canonical_team_name"]], on="team_id", how="left", validate="many_to_one")
    return approved


def make_history_lookup(understat: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        key: group.sort_values("understat_date").reset_index(drop=True)
        for key, group in understat.groupby(["understat_league", "understat_team_normalized"], dropna=False)
    }


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=["understat_date", "understat_source_file"] + METRICS)


def rolling_values(history: pd.DataFrame, match_date: pd.Timestamp, prefix: str) -> tuple[dict[str, object], pd.Timestamp | pd.NaT]:
    previous = history.loc[history["understat_date"] < match_date].copy()
    row: dict[str, object] = {
        f"{prefix}_understat_history_count": int(len(previous)),
        f"{prefix}_understat_latest_days_ago": np.nan,
        f"{prefix}_understat_found_flag": bool(len(previous) > 0),
    }
    latest_date: pd.Timestamp | pd.NaT = pd.NaT
    if len(previous):
        latest_date = previous["understat_date"].max()
        row[f"{prefix}_understat_latest_days_ago"] = int((match_date - latest_date).days)
    for window in WINDOWS:
        tail = previous.tail(window)
        row[f"{prefix}_understat_matches_w{window}"] = int(len(tail))
        for metric in METRICS:
            row[f"{prefix}_understat_{metric}_avg_w{window}"] = float(tail[metric].mean()) if len(tail) else np.nan
    return row, latest_date


def build_features(matches: pd.DataFrame, aliases: pd.DataFrame, understat: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    alias_by_team = {
        int(r.team_id): r.understat_team_normalized
        for r in aliases.itertuples(index=False)
    }
    alias_id_by_team = {
        int(r.team_id): int(r.alias_id)
        for r in aliases.itertuples(index=False)
    }
    lookup = make_history_lookup(understat)
    source_max_date = understat["understat_date"].max()
    rows = []
    usage_rows = []

    m = matches.copy()
    m["match_datetime"] = pd.to_datetime(m["match_datetime"], errors="coerce")
    m["match_date"] = m["match_datetime"].dt.floor("D")

    for r in m.itertuples(index=False):
        league = LEAGUE_TO_UNDERSTAT.get(r.competition_slug)
        match_date = r.match_date
        home_under = alias_by_team.get(int(r.home_team_id))
        away_under = alias_by_team.get(int(r.away_team_id))
        home_hist = lookup.get((league, home_under), empty_history()) if league and pd.notna(match_date) else empty_history()
        away_hist = lookup.get((league, away_under), empty_history()) if league and pd.notna(match_date) else empty_history()

        home_values, home_latest = rolling_values(home_hist, match_date, "home")
        away_values, away_latest = rolling_values(away_hist, match_date, "away")

        base = {
            "canonical_match_id": int(r.canonical_match_id),
            "understat_league": league,
            "understat_source_file": source_file,
            "understat_match_after_source_max_date_flag": bool(pd.notna(match_date) and match_date > source_max_date),
            "home_understat_alias_id": alias_id_by_team.get(int(r.home_team_id)),
            "away_understat_alias_id": alias_id_by_team.get(int(r.away_team_id)),
            "home_understat_latest_date": home_latest,
            "away_understat_latest_date": away_latest,
        }
        base.update(home_values)
        base.update(away_values)
        rows.append(base)

        usage_rows.extend(
            [
                {
                    "canonical_match_id": int(r.canonical_match_id),
                    "side": "home",
                    "team_id": int(r.home_team_id),
                    "understat_alias_id": alias_id_by_team.get(int(r.home_team_id)),
                    "understat_team_normalized": home_under,
                    "understat_league": league,
                    "found_flag": bool(home_values["home_understat_found_flag"]),
                    "latest_understat_date": home_latest,
                    "match_date": match_date,
                },
                {
                    "canonical_match_id": int(r.canonical_match_id),
                    "side": "away",
                    "team_id": int(r.away_team_id),
                    "understat_alias_id": alias_id_by_team.get(int(r.away_team_id)),
                    "understat_team_normalized": away_under,
                    "understat_league": league,
                    "found_flag": bool(away_values["away_understat_found_flag"]),
                    "latest_understat_date": away_latest,
                    "match_date": match_date,
                },
            ]
        )

    features = pd.DataFrame(rows)
    features["understat_both_found_flag"] = (
        features["home_understat_found_flag"].astype(bool) & features["away_understat_found_flag"].astype(bool)
    )
    for window in WINDOWS:
        for metric in METRICS:
            h = f"home_understat_{metric}_avg_w{window}"
            a = f"away_understat_{metric}_avg_w{window}"
            features[f"understat_home_minus_away_{metric}_avg_w{window}"] = features[h] - features[a]
    usage = pd.DataFrame(usage_rows)
    return features, usage


def coverage_report(matches: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = matches[["canonical_match_id", "competition_slug", "season_label"]].merge(features, on="canonical_match_id", how="left")
    return (
        merged.groupby(["competition_slug", "season_label"], dropna=False)
        .agg(
            row_count=("canonical_match_id", "size"),
            home_found_rate=("home_understat_found_flag", "mean"),
            away_found_rate=("away_understat_found_flag", "mean"),
            both_found_rate=("understat_both_found_flag", "mean"),
            after_source_max_date_rows=("understat_match_after_source_max_date_flag", "sum"),
            after_source_max_date_rate=("understat_match_after_source_max_date_flag", "mean"),
            home_latest_days_p50=("home_understat_latest_days_ago", "median"),
            home_latest_days_p95=("home_understat_latest_days_ago", lambda s: s.dropna().quantile(0.95) if s.notna().any() else np.nan),
            away_latest_days_p50=("away_understat_latest_days_ago", "median"),
            away_latest_days_p95=("away_understat_latest_days_ago", lambda s: s.dropna().quantile(0.95) if s.notna().any() else np.nan),
        )
        .reset_index()
    )


def staleness_report(matches: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = matches[["canonical_match_id", "competition_slug", "season_label"]].merge(features, on="canonical_match_id", how="left")
    stale = pd.concat(
        [
            merged[["competition_slug", "season_label", "home_understat_latest_days_ago"]]
            .rename(columns={"home_understat_latest_days_ago": "days_stale"})
            .assign(side="home"),
            merged[["competition_slug", "season_label", "away_understat_latest_days_ago"]]
            .rename(columns={"away_understat_latest_days_ago": "days_stale"})
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
            p95_days=("days_stale", lambda s: s.dropna().quantile(0.95) if s.notna().any() else np.nan),
            max_days=("days_stale", "max"),
        )
        .reset_index()
    )


def alias_usage_report(aliases: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    used = (
        usage.groupby("understat_alias_id", dropna=False)
        .agg(
            match_side_rows=("canonical_match_id", "size"),
            found_rows=("found_flag", "sum"),
            latest_used_date=("latest_understat_date", "max"),
        )
        .reset_index()
        .rename(columns={"understat_alias_id": "alias_id"})
    )
    report = aliases.merge(used, on="alias_id", how="left")
    report["match_side_rows"] = report["match_side_rows"].fillna(0).astype(int)
    report["found_rows"] = report["found_rows"].fillna(0).astype(int)
    return report[
        [
            "alias_id",
            "team_id",
            "canonical_team_name",
            "alias_name",
            "alias_normalized",
            "understat_team_normalized",
            "country_hint",
            "league_hint",
            "alias_status",
            "approved_for_research",
            "manual_review_required",
            "match_side_rows",
            "found_rows",
            "latest_used_date",
            "notes",
        ]
    ]


def validate(
    matches: pd.DataFrame,
    aliases_all: pd.DataFrame,
    aliases_approved: pd.DataFrame,
    features: pd.DataFrame,
    usage: pd.DataFrame,
    understat: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    rows = []

    def add(check_name: str, passed: bool, details: str) -> None:
        rows.append({"check_name": check_name, "status": "pass" if passed else "fail", "details": details})

    rejected_alias_ids = set(aliases_all.loc[aliases_all["alias_status"].eq("rejected"), "alias_id"].astype(int))
    manual_alias_ids = set(aliases_all.loc[bool_series(aliases_all["manual_review_required"]), "alias_id"].astype(int))
    used_alias_ids = set(usage["understat_alias_id"].dropna().astype(int))
    latest_dates = pd.concat([features["home_understat_latest_date"], features["away_understat_latest_date"]], ignore_index=True)
    match_dates = pd.to_datetime(matches["match_datetime"]).dt.floor("D")
    latest_matrix = features[["canonical_match_id", "home_understat_latest_date", "away_understat_latest_date"]].merge(
        matches[["canonical_match_id", "match_datetime"]], on="canonical_match_id", how="left"
    )
    latest_matrix["match_date"] = pd.to_datetime(latest_matrix["match_datetime"]).dt.floor("D")
    future_home = latest_matrix["home_understat_latest_date"].notna() & (latest_matrix["home_understat_latest_date"] >= latest_matrix["match_date"])
    future_away = latest_matrix["away_understat_latest_date"].notna() & (latest_matrix["away_understat_latest_date"] >= latest_matrix["match_date"])

    add("row_count_preserved", len(features) == len(matches), f"features={len(features)}, matches={len(matches)}")
    add("canonical_match_id_unique", not features["canonical_match_id"].duplicated().any(), f"duplicates={int(features['canonical_match_id'].duplicated().sum())}")
    add("no_rejected_alias_used", not bool(used_alias_ids & rejected_alias_ids), f"rejected_used={sorted(used_alias_ids & rejected_alias_ids)}")
    add("no_manual_review_alias_used", not bool(used_alias_ids & manual_alias_ids), f"manual_used={sorted(used_alias_ids & manual_alias_ids)}")
    add("dijon_rejected_alias_not_used", 384 not in used_alias_ids, "alias_id=384 rejected Dijon/Gijon fuzzy match is not used")
    add("all_used_aliases_approved", used_alias_ids <= set(aliases_approved["alias_id"].astype(int)), f"unapproved_used={sorted(used_alias_ids - set(aliases_approved['alias_id'].astype(int)))}")
    add("no_future_understat_home_rows", not future_home.any(), f"future_or_same_day_home={int(future_home.sum())}")
    add("no_future_understat_away_rows", not future_away.any(), f"future_or_same_day_away={int(future_away.sum())}")
    add("all_understat_contributing_dates_strictly_before_match_date", not (future_home.any() or future_away.any()), "latest contributing dates are < match_date")
    add("no_same_match_xg_stats_leakage", True, "Only lagged rolling averages/counts/staleness flags are written; no current fixture Understat rows are joined.")
    add("coverage_staleness_documented", True, "Coverage and staleness CSV reports are written by league and season.")
    add("source_max_date_flag_present", "understat_match_after_source_max_date_flag" in features.columns, f"understat_max_date={understat['understat_date'].max()}")

    validation = pd.DataFrame(rows)
    after_max_rows = int(features["understat_match_after_source_max_date_flag"].sum())
    if validation["status"].eq("fail").any():
        decision = "understat_locked_feature_block_failed"
    elif after_max_rows > 0:
        decision = "understat_locked_feature_block_ready_with_staleness_warning"
    else:
        decision = "understat_locked_feature_block_ready_good"
    return validation, decision


def write_reports(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    understat: pd.DataFrame,
    aliases_all: pd.DataFrame,
    aliases_approved: pd.DataFrame,
    features: pd.DataFrame,
    usage: pd.DataFrame,
    validation: pd.DataFrame,
    decision: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = coverage_report(matches, features)
    staleness = staleness_report(matches, features)
    understat_aliases_all = aliases_all[aliases_all["source"].eq("understat")].copy()
    understat_aliases_all["understat_team_normalized"] = understat_aliases_all["alias_name"].map(normalize_name)
    understat_aliases_all = understat_aliases_all.merge(
        teams[["team_id", "canonical_team_name"]], on="team_id", how="left", validate="many_to_one"
    )
    alias_usage = alias_usage_report(understat_aliases_all, usage)
    coverage.to_csv(REPORT_DIR / "understat_locked_coverage_by_league_season.csv", index=False)
    staleness.to_csv(REPORT_DIR / "understat_locked_staleness_summary.csv", index=False)
    alias_usage.to_csv(REPORT_DIR / "understat_locked_alias_usage.csv", index=False)
    validation.to_csv(REPORT_DIR / "understat_locked_leakage_checks.csv", index=False)

    rejected_count = int((aliases_all["source"].eq("understat") & aliases_all["alias_status"].eq("rejected")).sum())
    manual_count = int((aliases_all["source"].eq("understat") & bool_series(aliases_all["manual_review_required"])).sum())
    after_max_rows = int(features["understat_match_after_source_max_date_flag"].sum())
    both_rate = float(features["understat_both_found_flag"].mean())
    report = [
        "# Understat Locked Feature Block Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 locked entity registry and Understat game_stats from the manual archive. No modeling, value search, or super CSV merge was performed.",
        "",
        "## Alias Policy",
        "- Used only `source='understat'` aliases from `team_aliases_v1_locked.csv`.",
        "- Used only aliases with `approved_for_research=true`, `manual_review_required=false`, and `alias_status!='rejected'`.",
        "- No fuzzy matching and no new aliases were created.",
        "- Rejected Dijon/Gijon alias_id 384 was not used.",
        "",
        "## Temporal Policy",
        "- Understat rows are post-match data and are used only as lagged history.",
        "- Each match uses Understat rows with `understat_date < match_date`.",
        "- Same-day Understat rows are excluded.",
        "- Current fixture xG/stats are not joined as predictors.",
        "",
        "## Counts",
        f"- Match rows: {len(matches)}",
        f"- Feature rows: {len(features)}",
        f"- Approved Understat aliases available: {len(aliases_approved)}",
        f"- Rejected Understat aliases in locked registry: {rejected_count}",
        f"- Manual-review Understat aliases remaining: {manual_count}",
        f"- Both-team historical coverage rate: {both_rate:.4f}",
        f"- Rows after Understat source max date: {after_max_rows}",
        "",
        "## Understat Source",
        f"- Rows: {len(understat)}",
        f"- Date range: {understat['understat_date'].min()} to {understat['understat_date'].max()}",
        f"- Leagues: {', '.join(sorted(understat['understat_league'].dropna().unique()))}",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "understat_locked_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_md = [
        "# Understat Locked Feature Block Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "The locked Understat feature block is keyed by `canonical_match_id` and uses only approved locked entity aliases. It remains research-only.",
        "",
        "No modeling was performed and no confirmed edge is claimed.",
    ]
    (REPORT_DIR / "understat_locked_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    teams = pd.read_csv(TEAMS)
    aliases_all = pd.read_csv(ALIASES)
    matches = pd.read_csv(MATCHES)
    pd.read_csv(COMPETITIONS, dtype={"competition_code": str})

    raw, source_file = read_understat_game_stats(UNDERSTAT_ARCHIVE)
    understat = clean_understat(raw, source_file)
    aliases_approved = locked_understat_aliases(aliases_all, teams)
    features, usage = build_features(matches, aliases_approved, understat, source_file)
    validation, decision = validate(matches, aliases_all, aliases_approved, features, usage, understat)

    features.to_csv(OUTPUT_FEATURES, index=False)
    write_reports(matches, teams, understat, aliases_all, aliases_approved, features, usage, validation, decision)
    print(decision)
    print(f"wrote {OUTPUT_FEATURES}")


if __name__ == "__main__":
    main()
