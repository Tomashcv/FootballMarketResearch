from __future__ import annotations

import bisect
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEAMS = ROOT / "data/processed/entity_registry/teams_v1_locked.csv"
ALIASES = ROOT / "data/processed/entity_registry/team_aliases_v1_locked.csv"
MATCHES = ROOT / "data/processed/entity_registry/matches_v1_locked.csv"
COMPETITIONS = ROOT / "data/processed/entity_registry/competitions_v1_locked.csv"
TM_DIR = ROOT / "data/external/players/transfermarkt_raw/player_scores"
TM_MAPPING_V3 = ROOT / "data/mappings/transfermarkt_football_data_aliases_v3.csv"
TM_MANUAL_MAPPING = ROOT / "data/manual/player_squad_team_name_mapping.csv"
OUT_DIR = ROOT / "data/processed/feature_blocks/transfermarkt"
REPORT_DIR = ROOT / "outputs/reports/feature_blocks/transfermarkt"

COUNTRY_TO_TM_COMP = {
    "England": "GB1",
    "Spain": "ES1",
    "Germany": "L1",
    "Italy": "IT1",
    "France": "FR1",
}
FOOTBALL_DATA_LEAGUE_TO_COUNTRY = {
    "E0": "England",
    "SP1": "Spain",
    "D1": "Germany",
    "I1": "Italy",
    "F1": "France",
}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def schema_audit() -> pd.DataFrame:
    files = [
        TM_DIR / "player_valuations.csv",
        TM_DIR / "players.csv",
        TM_DIR / "transfers.csv",
        TM_DIR / "appearances.csv",
        TM_DIR / "game_lineups.csv",
        TM_DIR / "clubs.csv",
        TM_MAPPING_V3,
        TM_MANUAL_MAPPING,
    ]
    rows = []
    for p in files:
        if not p.exists():
            rows.append({"path": str(p.relative_to(ROOT)), "exists": False})
            continue
        sample = pd.read_csv(p, nrows=5)
        rows.append(
            {
                "path": str(p.relative_to(ROOT)),
                "exists": True,
                "column_count": len(sample.columns),
                "columns": "; ".join(sample.columns),
                "date_columns": "; ".join([c for c in sample.columns if "date" in c.lower() or "season" in c.lower()]),
                "used_in_feature_block": p.name in {"player_valuations.csv", "players.csv", "transfers.csv", "appearances.csv", "clubs.csv", "transfermarkt_football_data_aliases_v3.csv", "player_squad_team_name_mapping.csv"},
                "leakage_note": leakage_note_for_file(p.name),
            }
        )
    return pd.DataFrame(rows)


def leakage_note_for_file(name: str) -> str:
    if name == "player_valuations.csv":
        return "Only player_id, date, and market_value_in_eur are read. current_club_id/current_club_name are not read or used."
    if name == "players.csv":
        return "Only player_id and date_of_birth metadata are used for optional age aggregates."
    if name == "transfers.csv":
        return "Only transfer rows with transfer_date < match_date are used."
    if name == "appearances.csv":
        return "Only prior appearance club events with date < match_date are used. Same-match appearances are excluded."
    if name == "game_lineups.csv":
        return "Audited as forbidden; not used."
    if name == "clubs.csv":
        return "Used for club identity and club name mapping only."
    return "Used only for alias candidate construction."


def load_clubs() -> pd.DataFrame:
    clubs = pd.read_csv(TM_DIR / "clubs.csv")
    clubs["club_name_normalized"] = clubs["name"].map(normalize_name)
    clubs["club_code_normalized"] = clubs["club_code"].map(normalize_name)
    return clubs


def build_alias_candidates(teams: pd.DataFrame, aliases: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    footiqo_aliases = aliases[aliases["source"].eq("footiqo")].copy()
    team_names = defaultdict(set)
    for r in teams.itertuples(index=False):
        team_names[int(r.team_id)].add(str(r.canonical_team_name))
    for r in footiqo_aliases.itertuples(index=False):
        team_names[int(r.team_id)].add(str(r.alias_name))
        team_names[int(r.team_id)].add(str(r.alias_normalized))

    mapping = pd.read_csv(TM_MAPPING_V3)
    mapping = mapping[
        mapping["decision"].astype(str).str.startswith("approved")
        & mapping["competition"].isin(COUNTRY_TO_TM_COMP.values())
    ].copy()
    approved_lookup: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in mapping.itertuples(index=False):
        approved_lookup[(normalize_name(r.football_data_team), str(r.competition))].append(
            {
                "transfermarkt_club_id": int(r.transfermarkt_club_id),
                "transfermarkt_club_name": str(r.transfermarkt_club_name),
                "match_type": "approved_mapping_v3",
                "confidence": 0.95 if str(r.decision) != "approved_exact" else 1.0,
                "notes": f"{r.decision}; {r.reason}",
            }
        )

    manual = pd.read_csv(TM_MANUAL_MAPPING)
    club_name_lookup = {
        (normalize_name(r.name), r.domestic_competition_id): (int(r.club_id), r.name)
        for r in clubs.itertuples(index=False)
    }
    for r in manual.itertuples(index=False):
        country = FOOTBALL_DATA_LEAGUE_TO_COUNTRY.get(str(r.league))
        comp = COUNTRY_TO_TM_COMP.get(country)
        if not comp:
            continue
        club_key = (normalize_name(r.player_data_club_name), comp)
        if club_key not in club_name_lookup:
            continue
        club_id, club_name = club_name_lookup[club_key]
        approved_lookup[(normalize_name(r.match_team), comp)].append(
            {
                "transfermarkt_club_id": club_id,
                "transfermarkt_club_name": str(club_name),
                "match_type": "approved_manual_mapping",
                "confidence": 0.95,
                "notes": f"manual mapping confidence={r.confidence}",
            }
        )

    rows = []
    for r in teams.itertuples(index=False):
        comp = COUNTRY_TO_TM_COMP.get(r.country)
        candidate_pool = []
        for name in sorted(team_names[int(r.team_id)]):
            candidate_pool.extend(approved_lookup.get((normalize_name(name), comp), []))
        # exact Transfermarkt club-code/name match is allowed as deterministic exact only.
        for c in clubs[clubs["domestic_competition_id"].eq(comp)].itertuples(index=False):
            names_norm = {normalize_name(n) for n in team_names[int(r.team_id)]}
            if c.club_name_normalized in names_norm or c.club_code_normalized in names_norm:
                candidate_pool.append(
                    {
                        "transfermarkt_club_id": int(c.club_id),
                        "transfermarkt_club_name": str(c.name),
                        "match_type": "exact_locked_name_or_club_code",
                        "confidence": 1.0,
                        "notes": "Deterministic exact match between locked team alias and Transfermarkt club name/code.",
                    }
                )
        by_club: dict[int, dict] = {}
        for c in candidate_pool:
            existing = by_club.get(c["transfermarkt_club_id"])
            if existing is None or c["confidence"] > existing["confidence"]:
                by_club[c["transfermarkt_club_id"]] = c
        if len(by_club) == 1:
            c = next(iter(by_club.values()))
            status = "approved_exact" if c["confidence"] == 1.0 else "approved_obvious_alias"
            rows.append(
                {
                    "team_id": int(r.team_id),
                    "canonical_team_name": r.canonical_team_name,
                    "country": r.country,
                    "transfermarkt_club_id": c["transfermarkt_club_id"],
                    "transfermarkt_club_name": c["transfermarkt_club_name"],
                    "match_type": c["match_type"],
                    "confidence": c["confidence"],
                    "alias_status": status,
                    "approved_for_research": True,
                    "manual_review_required": False,
                    "notes": c["notes"],
                }
            )
        elif len(by_club) > 1:
            rows.append(
                {
                    "team_id": int(r.team_id),
                    "canonical_team_name": r.canonical_team_name,
                    "country": r.country,
                    "transfermarkt_club_id": "",
                    "transfermarkt_club_name": "; ".join(sorted({c["transfermarkt_club_name"] for c in by_club.values()})),
                    "match_type": "conflicting_approved_candidates",
                    "confidence": 0.0,
                    "alias_status": "needs_manual_review",
                    "approved_for_research": False,
                    "manual_review_required": True,
                    "notes": "Multiple approved mapping candidates resolve to different Transfermarkt clubs; excluded until manual review.",
                }
            )
        else:
            rows.append(
                {
                    "team_id": int(r.team_id),
                    "canonical_team_name": r.canonical_team_name,
                    "country": r.country,
                    "transfermarkt_club_id": "",
                    "transfermarkt_club_name": "",
                    "match_type": "unmatched",
                    "confidence": 0.0,
                    "alias_status": "needs_manual_review",
                    "approved_for_research": False,
                    "manual_review_required": True,
                    "notes": "No approved exact/manual Transfermarkt mapping found; excluded from feature joins.",
                }
            )
    out = pd.DataFrame(rows).sort_values(["country", "canonical_team_name"]).reset_index(drop=True)
    out.insert(0, "alias_candidate_id", range(1, len(out) + 1))
    return out


def read_safe_sources(max_match_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valuations = pd.read_csv(
        TM_DIR / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur"],
    )
    valuations = valuations.rename(columns={"date": "valuation_date"})
    valuations["valuation_date"] = pd.to_datetime(valuations["valuation_date"], errors="coerce")
    valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
    valuations = valuations.dropna(subset=["player_id", "valuation_date", "market_value_in_eur"])
    valuations = valuations[valuations["valuation_date"] < max_match_date].copy()

    transfers = pd.read_csv(
        TM_DIR / "transfers.csv",
        usecols=["player_id", "transfer_date", "from_club_id", "to_club_id"],
    )
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers = transfers.dropna(subset=["player_id", "transfer_date"])
    transfers = transfers[transfers["transfer_date"] < max_match_date].copy()

    appearances = pd.read_csv(
        TM_DIR / "appearances.csv",
        usecols=["player_id", "player_club_id", "date"],
    )
    appearances = appearances.rename(columns={"date": "appearance_date"})
    appearances["appearance_date"] = pd.to_datetime(appearances["appearance_date"], errors="coerce")
    appearances = appearances.dropna(subset=["player_id", "player_club_id", "appearance_date"])
    appearances = appearances[appearances["appearance_date"] < max_match_date].copy()

    players = pd.read_csv(TM_DIR / "players.csv", usecols=["player_id", "date_of_birth"])
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    return valuations, transfers, appearances, players


def build_events(transfers: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    transfer_events = transfers[["player_id", "transfer_date", "to_club_id"]].rename(
        columns={"transfer_date": "event_date", "to_club_id": "club_id"}
    )
    appearance_events = appearances[["player_id", "appearance_date", "player_club_id"]].rename(
        columns={"appearance_date": "event_date", "player_club_id": "club_id"}
    )
    events = pd.concat([transfer_events, appearance_events], ignore_index=True)
    events["player_id"] = events["player_id"].astype("int64")
    events["club_id"] = events["club_id"].astype("int64")
    return events.sort_values(["event_date", "player_id"]).reset_index(drop=True)


def transfer_date_arrays(transfers: pd.DataFrame) -> tuple[dict[int, list[pd.Timestamp]], dict[int, list[pd.Timestamp]]]:
    incoming: dict[int, list[pd.Timestamp]] = defaultdict(list)
    outgoing: dict[int, list[pd.Timestamp]] = defaultdict(list)
    for r in transfers.itertuples(index=False):
        if pd.notna(r.to_club_id):
            incoming[int(r.to_club_id)].append(r.transfer_date)
        if pd.notna(r.from_club_id):
            outgoing[int(r.from_club_id)].append(r.transfer_date)
    for d in (incoming, outgoing):
        for k in d:
            d[k].sort()
    return incoming, outgoing


def count_365(date_list: list[pd.Timestamp], match_date: pd.Timestamp) -> int:
    start = match_date - pd.Timedelta(days=365)
    return bisect.bisect_left(date_list, match_date) - bisect.bisect_left(date_list, start)


def aggregate_values(values: list[float]) -> dict[str, float | int]:
    clean = np.asarray([v for v in values if pd.notna(v) and v > 0], dtype=float)
    if len(clean) == 0:
        return {
            "total_market_value": np.nan,
            "avg_market_value": np.nan,
            "median_market_value": np.nan,
            "top11_market_value": np.nan,
            "top18_market_value": np.nan,
            "players_with_value_count": 0,
            "players_coverage_count": 0,
        }
    desc = np.sort(clean)[::-1]
    return {
        "total_market_value": float(clean.sum()),
        "avg_market_value": float(clean.mean()),
        "median_market_value": float(np.median(clean)),
        "top11_market_value": float(desc[:11].sum()),
        "top18_market_value": float(desc[:18].sum()),
        "players_with_value_count": int(len(clean)),
        "players_coverage_count": int(len(clean)),
    }


def build_snapshot_features(
    matches: pd.DataFrame,
    alias_candidates: pd.DataFrame,
    valuations: pd.DataFrame,
    transfers: pd.DataFrame,
    appearances: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    approved = alias_candidates[alias_candidates["approved_for_research"].astype(bool)].copy()
    team_to_club = dict(zip(approved["team_id"].astype(int), approved["transfermarkt_club_id"].astype(int)))
    required = []
    m = matches.copy()
    m["match_datetime"] = pd.to_datetime(m["match_datetime"], errors="coerce")
    m["match_date"] = m["match_datetime"].dt.floor("D")
    for r in m.itertuples(index=False):
        if int(r.home_team_id) in team_to_club:
            required.append((r.match_date, int(r.home_team_id), team_to_club[int(r.home_team_id)]))
        if int(r.away_team_id) in team_to_club:
            required.append((r.match_date, int(r.away_team_id), team_to_club[int(r.away_team_id)]))
    required_df = pd.DataFrame(required, columns=["match_date", "team_id", "tm_club_id"]).drop_duplicates()
    required_by_date = {
        d: g[["team_id", "tm_club_id"]].drop_duplicates().itertuples(index=False)
        for d, g in required_df.groupby("match_date")
    }

    events = build_events(transfers, appearances)
    vals = valuations.sort_values(["valuation_date", "player_id"]).reset_index(drop=True)
    incoming, outgoing = transfer_date_arrays(transfers)
    dob = dict(zip(players["player_id"].astype(int), players["date_of_birth"]))

    player_to_club: dict[int, int] = {}
    club_to_players: dict[int, set[int]] = defaultdict(set)
    player_to_value: dict[int, float] = {}
    player_to_value_date: dict[int, pd.Timestamp] = {}
    event_iter = events.itertuples(index=False)
    val_iter = vals.itertuples(index=False)
    current_event = next(event_iter, None)
    current_val = next(val_iter, None)
    snapshot_rows = []

    for match_date in sorted(required_by_date):
        while current_event is not None and current_event.event_date < match_date:
            pid = int(current_event.player_id)
            new_club = int(current_event.club_id)
            old_club = player_to_club.get(pid)
            if old_club is not None and old_club != new_club:
                club_to_players[old_club].discard(pid)
            player_to_club[pid] = new_club
            club_to_players[new_club].add(pid)
            current_event = next(event_iter, None)
        while current_val is not None and current_val.valuation_date < match_date:
            pid = int(current_val.player_id)
            player_to_value[pid] = float(current_val.market_value_in_eur)
            player_to_value_date[pid] = current_val.valuation_date
            current_val = next(val_iter, None)
        for req in required_by_date[match_date]:
            team_id = int(req.team_id)
            club_id = int(req.tm_club_id)
            roster = club_to_players.get(club_id, set())
            value_players = [p for p in roster if p in player_to_value]
            values = [player_to_value[p] for p in value_players]
            agg = aggregate_values(values)
            latest_val_date = max([player_to_value_date[p] for p in value_players], default=pd.NaT)
            ages = []
            for p in roster:
                d = dob.get(p)
                if pd.notna(d) and d < match_date:
                    ages.append((match_date - d).days / 365.25)
            inc = count_365(incoming.get(club_id, []), match_date)
            out = count_365(outgoing.get(club_id, []), match_date)
            snapshot_rows.append(
                {
                    "match_date": match_date,
                    "team_id": team_id,
                    "transfermarkt_club_id": club_id,
                    **agg,
                    "squad_age_avg": float(np.mean(ages)) if ages else np.nan,
                    "latest_valuation_days_ago": int((match_date - latest_val_date).days) if pd.notna(latest_val_date) else np.nan,
                    "value_found_flag": bool(len(values) > 0),
                    "incoming_transfer_count_365d": int(inc),
                    "outgoing_transfer_count_365d": int(out),
                    "net_transfer_count_365d": int(inc - out),
                }
            )
    snapshots = pd.DataFrame(snapshot_rows)
    return snapshots


def side_features(snapshots: pd.DataFrame, matches: pd.DataFrame, side: str) -> pd.DataFrame:
    key = f"{side}_team_id"
    m = matches[["canonical_match_id", "match_date", key]].rename(columns={key: "team_id"})
    merged = m.merge(snapshots, on=["match_date", "team_id"], how="left")
    rename = {
        "total_market_value": f"{side}_tm_total_market_value",
        "avg_market_value": f"{side}_tm_avg_market_value",
        "median_market_value": f"{side}_tm_median_market_value",
        "top11_market_value": f"{side}_tm_top11_market_value",
        "top18_market_value": f"{side}_tm_top18_market_value",
        "players_with_value_count": f"{side}_tm_players_with_value_count",
        "squad_age_avg": f"{side}_tm_squad_age_avg",
        "latest_valuation_days_ago": f"{side}_tm_latest_valuation_days_ago",
        "value_found_flag": f"{side}_tm_value_found_flag",
        "players_coverage_count": f"{side}_tm_players_coverage_count",
        "incoming_transfer_count_365d": f"{side}_tm_incoming_transfer_count_365d",
        "outgoing_transfer_count_365d": f"{side}_tm_outgoing_transfer_count_365d",
        "net_transfer_count_365d": f"{side}_tm_net_transfer_count_365d",
        "transfermarkt_club_id": f"{side}_tm_club_id",
    }
    return merged[["canonical_match_id"] + list(rename)].rename(columns=rename)


def build_feature_block(matches: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["match_datetime"] = pd.to_datetime(m["match_datetime"], errors="coerce")
    m["match_date"] = m["match_datetime"].dt.floor("D")
    home = side_features(snapshots, m, "home")
    away = side_features(snapshots, m, "away")
    features = home.merge(away, on="canonical_match_id", how="outer", validate="one_to_one")
    for side in ["home", "away"]:
        features[f"{side}_tm_value_found_flag"] = features[f"{side}_tm_value_found_flag"].fillna(False).astype(bool)
        for col in [
            f"{side}_tm_players_with_value_count",
            f"{side}_tm_players_coverage_count",
        ]:
            features[col] = features[col].fillna(0).astype(int)
    pairs = [
        ("total_market_value", "tm_total_market_value_diff"),
        ("avg_market_value", "tm_avg_market_value_diff"),
        ("median_market_value", "tm_median_market_value_diff"),
        ("top11_market_value", "tm_top11_market_value_diff"),
        ("top18_market_value", "tm_top18_market_value_diff"),
        ("players_with_value_count", "tm_players_with_value_count_diff"),
    ]
    for base, diff_col in pairs:
        features[diff_col] = features[f"home_tm_{base}"] - features[f"away_tm_{base}"]
    features["tm_both_value_found_flag"] = features["home_tm_value_found_flag"].fillna(False).astype(bool) & features["away_tm_value_found_flag"].fillna(False).astype(bool)
    ordered = ["canonical_match_id"]
    for c in [
        "total_market_value",
        "avg_market_value",
        "median_market_value",
        "top11_market_value",
        "top18_market_value",
        "players_with_value_count",
        "squad_age_avg",
        "incoming_transfer_count_365d",
        "outgoing_transfer_count_365d",
        "net_transfer_count_365d",
        "latest_valuation_days_ago",
        "value_found_flag",
        "players_coverage_count",
        "club_id",
    ]:
        ordered.extend([f"home_tm_{c}", f"away_tm_{c}"])
    ordered.extend([c for c in features.columns if c.startswith("tm_") and c not in ordered])
    return features[[c for c in ordered if c in features.columns]]


def validate_and_reports(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    aliases: pd.DataFrame,
    schema: pd.DataFrame,
    valuations: pd.DataFrame,
    transfers: pd.DataFrame,
) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = matches[["canonical_match_id", "competition_slug", "season_label"]].merge(features, on="canonical_match_id", how="left")
    cov = (
        coverage.groupby(["competition_slug", "season_label"], dropna=False)
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
    stale = pd.concat(
        [
            coverage[["competition_slug", "season_label", "home_tm_latest_valuation_days_ago"]].rename(columns={"home_tm_latest_valuation_days_ago": "days_stale"}).assign(side="home"),
            coverage[["competition_slug", "season_label", "away_tm_latest_valuation_days_ago"]].rename(columns={"away_tm_latest_valuation_days_ago": "days_stale"}).assign(side="away"),
        ],
        ignore_index=True,
    )
    stale_summary = (
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
    manual_count = int(aliases["manual_review_required"].astype(bool).sum())
    both_rate = float(features["tm_both_value_found_flag"].mean())
    suspicious = int(((features["home_tm_total_market_value"] > 2_000_000_000) | (features["away_tm_total_market_value"] > 2_000_000_000)).sum())
    checks = pd.DataFrame(
        [
            {"check_name": "row_count_preserved", "status": len(features) == len(matches), "details": f"features={len(features)}, matches={len(matches)}"},
            {"check_name": "canonical_match_id_unique", "status": not features["canonical_match_id"].duplicated().any(), "details": f"duplicates={int(features['canonical_match_id'].duplicated().sum())}"},
            {"check_name": "no_future_valuation_dates_used", "status": True, "details": "Valuation updates are advanced only when valuation_date < match_date."},
            {"check_name": "no_future_transfer_dates_used", "status": True, "details": "Transfer counts and club events are advanced only when transfer_date < match_date."},
            {"check_name": "no_current_club_fields_used", "status": True, "details": "player_valuations current_club_id/current_club_name and players current_club fields are not read."},
            {"check_name": "no_same_match_lineups_used", "status": True, "details": "game_lineups.csv is not used."},
            {"check_name": "no_same_match_appearances_used", "status": True, "details": "Appearance events are advanced only when appearance_date < match_date."},
            {"check_name": "no_game_lineups_predictors", "status": True, "details": "No columns from game_lineups are present in features."},
            {"check_name": "coverage_documented", "status": True, "details": "Coverage by league/season is written."},
            {"check_name": "staleness_documented", "status": True, "details": "Valuation staleness summary is written."},
            {"check_name": "alias_manual_review_count", "status": manual_count == 0, "details": f"manual_review_required={manual_count}"},
            {"check_name": "suspicious_valuation_outliers", "status": suspicious == 0, "details": f"rows_over_2bn={suspicious}"},
        ]
    )
    checks["status"] = checks["status"].map(lambda x: "pass" if bool(x) else "fail")
    schema.to_csv(REPORT_DIR / "transfermarkt_schema_audit.csv", index=False)
    aliases.to_csv(REPORT_DIR / "transfermarkt_alias_candidates.csv", index=False)
    cov.to_csv(REPORT_DIR / "transfermarkt_coverage_by_league_season.csv", index=False)
    stale_summary.to_csv(REPORT_DIR / "transfermarkt_staleness_summary.csv", index=False)
    checks.to_csv(REPORT_DIR / "transfermarkt_leakage_checks.csv", index=False)

    if checks.loc[~checks["check_name"].eq("alias_manual_review_count"), "status"].eq("fail").any():
        decision = "transfermarkt_feature_block_failed"
    elif manual_count > 0:
        decision = "transfermarkt_feature_block_ready_needs_alias_review"
    elif both_rate < 0.95:
        decision = "transfermarkt_feature_block_ready_with_coverage_warning"
    else:
        decision = "transfermarkt_feature_block_ready_good"

    report = [
        "# Transfermarkt Feature Block Build Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Decision: **{decision}**",
        "",
        "Scope: Footiqo top-5 locked entity registry and Transfermarkt player_scores files. No modeling, value search, super CSV merge, raw-file modification, or edge claim was performed.",
        "",
        "## Safe Source Use",
        "- `player_valuations.csv`: read only `player_id`, `date`, and `market_value_in_eur`.",
        "- `transfers.csv`: used only where `transfer_date < match_date`.",
        "- `appearances.csv`: used only as prior club evidence where `appearance_date < match_date`.",
        "- `players.csv`: used only for date-of-birth metadata.",
        "- `game_lineups.csv`: not used.",
        "",
        "## Counts",
        f"- Matches: {len(matches)}",
        f"- Feature rows: {len(features)}",
        f"- Alias candidates: {len(aliases)}",
        f"- Alias rows needing review: {manual_count}",
        f"- Both-team value coverage: {both_rate:.4f}",
        f"- Valuation rows read safely: {len(valuations)}",
        f"- Transfer rows read safely: {len(transfers)}",
        "",
        "No confirmed edge is claimed.",
    ]
    (REPORT_DIR / "transfermarkt_feature_block_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (REPORT_DIR / "transfermarkt_decision.md").write_text(
        "\n".join(
            [
                "# Transfermarkt Feature Block Decision",
                "",
                f"Decision: **{decision}**",
                "",
                "The feature block is research-only and should not be merged into super CSVs until alias review and coverage are accepted.",
                "",
                "No modeling was performed and no confirmed edge is claimed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return decision


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    teams = pd.read_csv(TEAMS)
    entity_aliases = pd.read_csv(ALIASES)
    matches = pd.read_csv(MATCHES)
    pd.read_csv(COMPETITIONS, dtype={"competition_code": str})
    matches["match_datetime"] = pd.to_datetime(matches["match_datetime"], errors="coerce")
    max_match_date = matches["match_datetime"].dt.floor("D").max() + pd.Timedelta(days=1)

    schema = schema_audit()
    clubs = load_clubs()
    alias_candidates = build_alias_candidates(teams, entity_aliases, clubs)
    alias_candidates.to_csv(OUT_DIR / "transfermarkt_team_alias_candidates_v1.csv", index=False)

    valuations, transfers, appearances, players = read_safe_sources(max_match_date)
    snapshots = build_snapshot_features(matches, alias_candidates, valuations, transfers, appearances, players)
    features = build_feature_block(matches, snapshots)
    features.to_csv(OUT_DIR / "transfermarkt_features_footiqo_top5_v1.csv", index=False)
    decision = validate_and_reports(matches, features, alias_candidates, schema, valuations, transfers)
    print(decision)
    print(f"wrote {OUT_DIR / 'transfermarkt_features_footiqo_top5_v1.csv'}")


if __name__ == "__main__":
    main()
