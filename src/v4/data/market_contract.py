"""Conservative market-column classification for the V4 Phase 1 audit.

The rules in this module classify schema, not tradability.  In particular,
football-data's non-``C`` prices are *not* called opening prices: the local
provider notes describe a scheduled collection snapshot, while only ``C``
variants are explicitly documented as the last odds before kickoff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


TIMING_CLASSES = {
    "verified_opening",
    "verified_closing",
    "current_snapshot_unknown_time",
    "timing_unknown",
    "not_applicable",
}

MARKETS = {"1X2", "Asian Handicap", "Over/Under", "unknown market"}

FOOTBALL_DATA_EVIDENCE = {
    "verified_closing": "fd_archive_notes_closing_definition",
    "current_snapshot_unknown_time": "fd_archive_notes_scheduled_collection_not_opening",
    "not_applicable": "fd_archive_notes_line_or_bookmaker_count_definition",
}


@dataclass(frozen=True)
class ColumnContract:
    market: str
    role: str
    selection: str
    bookmaker: str
    line: float | None
    timing_classification: str
    timing_evidence_id: str
    feature_policy: str


_FD_1X2_PREFIXES = {
    "B365": "Bet365",
    "BS": "Blue Square",
    "BW": "Bet&Win/Bwin",
    "BFD": "Betfair Sportsbook",
    "BFE": "Betfair Exchange",
    "BF": "Betfair",
    "BMGM": "BetMGM",
    "BV": "BetVictor",
    "CL": "Coral",
    "GB": "Gamebookers",
    "IW": "Interwetten",
    "LB": "Ladbrokes",
    "P": "Pinnacle",
    "PS": "Pinnacle",
    "SO": "Sporting Odds",
    "SB": "Sportingbet",
    "SJ": "Stan James",
    "SY": "Stanleybet",
    "VC": "VC Bet",
    "WH": "William Hill",
    "1XB": "1xBet",
    "Avg": "Market average",
    "Max": "Market maximum",
    "BbAv": "BetBrain average",
    "BbMx": "BetBrain maximum",
}


def _policy(timing: str) -> str:
    if timing == "verified_opening":
        return "allowed_opening_time_feature"
    if timing == "verified_closing":
        return "closing_label_or_diagnostic_only"
    if timing in {"current_snapshot_unknown_time", "timing_unknown"}:
        return "prohibited_until_timing_resolved"
    return "not_a_price_feature"


def _fd_timing(column: str, role: str) -> tuple[str, str]:
    if role == "bookmaker_count":
        timing = "not_applicable"
    elif is_football_data_closing(column):
        timing = "verified_closing"
    else:
        timing = "current_snapshot_unknown_time"
    return timing, FOOTBALL_DATA_EVIDENCE[timing]


def is_football_data_closing(column: str) -> bool:
    """Return whether a known football-data price column is a C variant.

    This is only a schema recognizer.  The verified timing classification is
    justified by the provider notes recorded in the audit evidence table.
    """

    c = column.strip().lstrip("\ufeff").replace("ï»¿", "")
    if c == "AHCh":
        return True
    if re.match(r"^(?:B365|BFD|BMGM|BV|BW|CL|LB|PS|WH|VC|IW|BF|1XB)C[HDA]$", c):
        return True
    if re.match(r"^(?:Max|Avg)C[HDA]$", c):
        return True
    if re.match(r"^(?:B365|P|Max|Avg|BFE)C[<>]\d+(?:\.\d+)?$", c):
        return True
    if re.match(r"^(?:B365|P|Max|Avg|BFE)CAH[HA]$", c):
        return True
    return False


def _fd_bookmaker(prefix: str) -> str:
    # ``VC`` is the bookmaker abbreviation itself; only ``VCC`` represents
    # that abbreviation plus the closing marker.
    if prefix == "VC":
        return _FD_1X2_PREFIXES[prefix]
    prefix = prefix.removesuffix("C")
    return _FD_1X2_PREFIXES.get(prefix, prefix or "unknown")


def _classify_football_data(column: str) -> ColumnContract | None:
    c = column.strip().lstrip("\ufeff").replace("ï»¿", "")
    if not c:
        return None

    if c in {"Bb1X2", "BbOU", "BbAH"}:
        market = {"Bb1X2": "1X2", "BbOU": "Over/Under", "BbAH": "Asian Handicap"}[c]
        timing, evidence = _fd_timing(c, "bookmaker_count")
        return ColumnContract(market, "bookmaker_count", "count", "BetBrain", None, timing, evidence, _policy(timing))

    if c in {"AHh", "AHCh", "BbAHh", "B365AH", "GBAH", "LBAH"}:
        timing, evidence = _fd_timing(c, "line")
        return ColumnContract("Asian Handicap", "line", "home_handicap", "Market" if c in {"AHh", "AHCh"} else _fd_bookmaker(re.sub(r"AHh?$", "", c)), None, timing, evidence, _policy(timing))

    m = re.match(r"^(?P<prefix>BbMx|BbAv|B365|BFE|P|Max|Avg|GB)(?P<close>C?)(?P<side>[<>])(?P<line>\d+(?:\.\d+)?)$", c)
    if m:
        timing, evidence = _fd_timing(c, "odds")
        selection = "over" if m.group("side") == ">" else "under"
        return ColumnContract("Over/Under", "odds", selection, _fd_bookmaker(m.group("prefix") + m.group("close")), float(m.group("line")), timing, evidence, _policy(timing))

    # Modern and legacy Asian handicap odds.  The line lives in a paired line column.
    m = re.match(r"^(?P<prefix>BbMx|BbAv|B365|P|Max|Avg|GB|LB)(?P<close>C?)AH(?P<side>H|A)$", c)
    if m:
        timing, evidence = _fd_timing(c, "odds")
        return ColumnContract("Asian Handicap", "odds", "home" if m.group("side") == "H" else "away", _fd_bookmaker(m.group("prefix") + m.group("close")), None, timing, evidence, _policy(timing))
    m = re.match(r"^(?P<prefix>B365|P|Max|Avg|BFE)C?AH(?P<side>H|A)$", c)
    if m:
        timing, evidence = _fd_timing(c, "odds")
        return ColumnContract("Asian Handicap", "odds", "home" if m.group("side") == "H" else "away", _fd_bookmaker(m.group("prefix")), None, timing, evidence, _policy(timing))

    # 1X2 prices.  Restrict prefixes to documented/provider-observed families so
    # result/stat columns such as FTHG, HST, and FTAG cannot be misclassified.
    m = re.match(r"^(?P<prefix>BbMx|BbAv|B365|BFD|BMGM|BFE|BF|BV|BW|BS|CL|GB|IW|LB|PS|P|SO|SB|SJ|SY|VC|WH|1XB|Max|Avg)(?P<close>C?)(?P<side>H|D|A)$", c)
    if m:
        timing, evidence = _fd_timing(c, "odds")
        selection = {"H": "home", "D": "draw", "A": "away"}[m.group("side")]
        return ColumnContract("1X2", "odds", selection, _fd_bookmaker(m.group("prefix") + m.group("close")), None, timing, evidence, _policy(timing))
    return None


def _classify_footiqo(column: str) -> ColumnContract | None:
    c = column.strip()
    if c in {"H", "D", "A"}:
        selection = {"H": "home", "D": "draw", "A": "away"}[c]
        timing = "timing_unknown"
        return ColumnContract("1X2", "odds", selection, "unknown", None, timing, "footiqo_project_docs_timing_unknown", _policy(timing))
    m = re.match(r"^(O|U)(\d)(\d)$", c)
    if m:
        line = float(f"{m.group(2)}.{m.group(3)}")
        timing = "timing_unknown"
        return ColumnContract("Over/Under", "odds", "over" if m.group(1) == "O" else "under", "unknown", line, timing, "footiqo_project_docs_timing_unknown", _policy(timing))
    if c in {"BTTSY", "BTTSN"}:
        timing = "timing_unknown"
        return ColumnContract("unknown market", "odds", "yes" if c.endswith("Y") else "no", "unknown", None, timing, "footiqo_project_docs_timing_unknown", _policy(timing))
    return None


def _classify_beat_the_bookie(column: str, member: str) -> ColumnContract | None:
    c = column.strip()
    m = re.match(r"^(home|draw|away)_b(\d+)_(\d+)$", c)
    if m:
        timing = "timing_unknown"
        return ColumnContract("1X2", "odds", m.group(1), f"encoded_b{m.group(2)}", None, timing, "btb_local_metadata_no_series_timestamp_semantics", _policy(timing))
    if c in {
        "avg_odds_home_win", "avg_odds_draw", "avg_odds_away_win",
        "max_odds_home_win", "max_odds_draw", "max_odds_away_win",
    }:
        selection = "home" if "home" in c else "draw" if "draw" in c else "away"
        bookmaker = "market average" if c.startswith("avg") else "market maximum"
        timing = "timing_unknown"
        return ColumnContract("1X2", "odds", selection, bookmaker, None, timing, "btb_filename_not_sufficient_timing_evidence", _policy(timing))
    if c.startswith("n_odds_"):
        timing = "not_applicable"
        return ColumnContract("1X2", "bookmaker_count", c.removeprefix("n_odds_"), "market", None, timing, "btb_column_is_count", _policy(timing))
    return None


def classify_column(source: str, column: str, member: str = "") -> ColumnContract | None:
    """Classify a raw market column, returning ``None`` for non-market fields."""

    if source == "football_data":
        return _classify_football_data(column)
    if source == "footiqo":
        return _classify_footiqo(column)
    if source == "beat_the_bookie":
        return _classify_beat_the_bookie(column, member)
    raise ValueError(f"Unsupported source: {source}")
