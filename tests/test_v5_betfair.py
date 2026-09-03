import bz2
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.v5_betfair.core import (
    apply_ltp_updates, discover_raw_files, extract_market, first_complete_definition,
    iter_json_lines, no_vig_ltp_proxy, normalize_team, parse_market_definition,
    runner_orientation, temporal_partitions,
)
from src.v5_betfair.research import FEATURES, build_research_rows


def write_stream(path: Path, messages):
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wt", encoding="utf-8") as f:
        for message in messages:
            f.write(json.dumps(message, sort_keys=True) + "\n")


def pt(value):
    return int(pd.Timestamp(value).timestamp() * 1000)


def definition(in_play=False):
    return {"eventId":"e1","eventName":"Arsenal v Chelsea","marketType":"MATCH_ODDS",
            "marketTime":"2025-01-02T12:00:00Z","openDate":"2024-12-30T12:00:00Z",
            "timezone":"GMT","countryCode":"GB","complete":True,"inPlay":in_play,"status":"OPEN",
            "runners":[{"id":1,"name":"Arsenal","sortPriority":1,"status":"ACTIVE"},
                       {"id":2,"name":"Chelsea","sortPriority":2,"status":"ACTIVE"},
                       {"id":3,"name":"The Draw","sortPriority":3,"status":"ACTIVE"}]}


@pytest.fixture
def stream(tmp_path):
    path=tmp_path/"nested"/"BASIC"/"2025"/"January"/"2"/"e1"/"1.1.bz2"
    messages=[
      {"op":"mcm","pt":pt("2024-12-30T12:00:00Z"),"mc":[{"id":"1.1","marketDefinition":definition()}]},
      {"op":"mcm","pt":pt("2025-01-01T11:00:00Z"),"mc":[{"id":"1.1","rc":[{"id":1,"ltp":2.0},{"id":2,"ltp":4.0},{"id":3,"ltp":3.5}]}]},
      {"op":"mcm","pt":pt("2025-01-02T11:30:00Z"),"mc":[{"id":"1.1","rc":[{"id":1,"ltp":1.9}]}]},
      {"op":"mcm","pt":pt("2025-01-02T12:00:00Z"),"mc":[{"id":"1.1","rc":[{"id":2,"ltp":9.0}]}]},
      {"op":"mcm","pt":pt("2025-01-02T12:01:00Z"),"mc":[{"id":"1.1","marketDefinition":definition(True),"rc":[{"id":1,"ltp":8.0}]}]},
    ]
    write_stream(path,messages); return path


def mapping(path):
    return {"market_id":"1.1","event_id":"e1","canonical_fixture_id":"f1","season":2024,
            "fixture_date":"2025-01-02","market_time_utc":"2025-01-02T12:00:00Z",
            "source_file":str(path),"home_runner_id":1,"away_runner_id":2,"draw_runner_id":3}


def test_recursive_discovery_and_bz2_streaming(stream):
    assert discover_raw_files(stream.parents[6]) == [stream.resolve()]
    rows=list(iter_json_lines(stream)); assert len(rows)==5 and rows[0][1]["op"]=="mcm"


def test_json_line_error_is_explicit(tmp_path):
    path=tmp_path/"bad.bz2"
    with bz2.open(path,"wt") as f: f.write("not-json\n")
    with pytest.raises(json.JSONDecodeError): list(iter_json_lines(path))


def test_market_definition_parsing_and_catalog(stream):
    row=first_complete_definition(stream)
    assert row["market_id"]=="1.1" and row["event_id"]=="e1"
    assert row["market_type"]=="MATCH_ODDS" and row["number_of_runners"]==3
    assert row["runner_names"]==["Arsenal","Chelsea","The Draw"]


def test_runner_name_mapping_not_sort_only():
    d=definition(); d["runners"]=[d["runners"][2],d["runners"][1],d["runners"][0]]
    assert runner_orientation(d,"Arsenal","Chelsea")=={"draw":3,"away":2,"home":1}
    assert normalize_team("The Draw")=="draw"


def test_ltp_updates_and_proxy():
    state={}; ts=datetime(2025,1,1,tzinfo=timezone.utc)
    assert apply_ltp_updates(state,[{"id":1,"ltp":2.0},{"id":2,"ltp":1.0}],ts)==1
    proxy,over=no_vig_ltp_proxy({"home":2,"draw":4,"away":4})
    assert over==1 and proxy=={"home":.5,"draw":.25,"away":.25}
    assert no_vig_ltp_proxy({"home":2,"draw":None,"away":4})==(None,None)


def test_asof_no_future_fill_inplay_and_start_exclusion(stream):
    result=extract_market(stream,mapping(stream))
    states={x["cutoff"]:x for x in result.cutoffs}
    assert states["t24h"]["home_ltp"]==2.0
    assert states["t15m"]["home_ltp"]==1.9
    assert states["t15m"]["away_ltp"]==4.0
    assert all(x["away_ltp"]!=9.0 for x in result.cutoffs)
    assert all(x["home_ltp"]!=8.0 for x in result.cutoffs)
    assert result.metadata["inplay_observation_count"]==1


def test_staleness_is_runner_specific(stream):
    states={x["cutoff"]:x for x in extract_market(stream,mapping(stream)).cutoffs}
    assert states["t15m"]["home_staleness_seconds"]==15*60
    assert states["t15m"]["away_staleness_seconds"]==24*3600+45*60


def test_temporal_purge_and_determinism():
    assert temporal_partitions([2018,2019,2020,2021],2021,1)=={"train":[2018],"calibration":[2019],"test":[2021]}
    assert temporal_partitions([2020,2018,2021,2019],2021,1)==temporal_partitions([2018,2019,2020,2021],2021,1)


def test_target_feature_isolation(stream):
    extracted=extract_market(stream,mapping(stream))
    frame=build_research_rows(pd.DataFrame(extracted.cutoffs))
    assert not frame.empty
    assert not ({"future_probability_proxy","probability_shift","positive_movement","ltp_change"}&set(FEATURES))
    assert "result_1x2" not in FEATURES


def test_catalog_checkpoint_resume(tmp_path, monkeypatch, stream):
    pytest.importorskip("pyarrow")
    import src.v5_betfair.pipeline as pipeline
    monkeypatch.setattr(pipeline,"CHECKPOINTS",tmp_path/"checkpoints")
    monkeypatch.setattr(pipeline,"CATALOG",tmp_path/"catalog.parquet")
    pipeline.CHECKPOINTS.mkdir()
    first=pipeline.phase1([stream],log_every=999)
    second=pipeline.phase1([stream],log_every=999)
    assert len(first)==len(second)==1
    db=__import__("sqlite3").connect(tmp_path/"checkpoints/catalog.sqlite")
    assert db.execute("select count(*) from catalog").fetchone()[0]==1


def test_locked_e0_mapping_orientation(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import src.v5_betfair.pipeline as pipeline
    monkeypatch.setattr(pipeline,"MARKET_MAP",tmp_path/"map.parquet")
    monkeypatch.setattr(pipeline,"REPORTS",tmp_path)
    catalog=pd.DataFrame([{"market_id":"x","event_id":"x","source_file":"x","event_name":"Manchester Utd v Tottenham",
      "market_time_utc":"2015-08-08T13:45:00Z","country_code":"GB","market_type":"MATCH_ODDS",
      "runner_names":["Manchester Utd","Tottenham","The Draw"],"runner_ids":[100,200,300]}])
    mapped=pipeline.phase2(catalog)
    row=mapped.iloc[0]
    assert bool(row.approved_unique) and row.home_runner_id==100 and row.away_runner_id==200 and row.draw_runner_id==300


def test_duplicate_market_catalog_flag(tmp_path, monkeypatch):
    # Physical copies are deterministically reduced by market_id after cataloging.
    frame=pd.DataFrame({"market_id":["1","1","2"],"source_file":["a","b","c"]}).sort_values(["market_id","source_file"],kind="stable")
    frame["duplicate"]=frame.duplicated("market_id",keep="first")
    assert frame.duplicate.tolist()==[False,True,False]


def test_mixed_iso_timestamp_forms_are_supported():
    values=pd.Series(["2025-01-01T12:00:00Z","2025-01-01T12:00:00.123000Z"])
    parsed=pd.to_datetime(values,utc=True,format="mixed")
    assert parsed.notna().all() and parsed.iloc[0] < parsed.iloc[1]


def test_event_name_pair_supports_reordered_common_formats():
    from src.v5_betfair.mapping_coverage_audit import _event_pair_support
    fixture=SimpleNamespace(home_team_id=1,away_team_id=2)
    aliases={1:{"arsenal"},2:{"chelsea"}}
    market=SimpleNamespace(event_name="Chelsea vs Arsenal")
    assert _event_pair_support(market,fixture,aliases)==(True,"vs")
    assert _event_pair_support(SimpleNamespace(event_name="Chelsea @ Arsenal"),fixture,aliases)==(True,"@")
    assert _event_pair_support(SimpleNamespace(event_name="Arsenal v Everton"),fixture,aliases)[0] is False
