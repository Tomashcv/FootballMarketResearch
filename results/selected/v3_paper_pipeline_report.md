# V3 Paper Trading Pipeline

Decision: `v3_paper_pipeline_ready_for_sportsedge_integration_research_only`

Labels: `v3_paper_pipeline_built_research_only`, `v3_paper_pipeline_ready_for_sportsedge_integration_research_only` when current data and leakage checks pass.

## What Was Built
- Frozen V3 config at `configs/v3_frozen_candidate.yaml`.
- Local prediction, ledger update, settlement, HTML report, and master pipeline scripts.
- Idempotent paper ledger with deterministic `paper_bet_id`.
- Raw input snapshot manifests with file sizes, SHA256 hashes, row counts, dates, leagues, and warnings.

## How To Run
`python scripts/run_v3_paper_pipeline.py`

## Files Created
- `outputs/paper_trading/v3/v3_latest_row_predictions.csv`
- `outputs/paper_trading/v3/v3_latest_candidate_picks.csv`
- `outputs/paper_trading/v3/v3_paper_ledger.csv`
- `outputs/paper_trading/v3/html/v3_paper_latest.html`
- `outputs/paper_trading/v3/snapshots/<run_id>_manifest.json`

## Data Assumptions
- Current season rows come only from local football-data raw season CSVs under `data/raw/*/seasons/`.
- E1/E2/E3 remain excluded.
- Historical exact V3 rows are used only as pre-current-season training history.
- ClubElo, Transfermarkt, and internal Elo checks must remain point-in-time safe.

## Limitations
- No API keys or odds feeds are used.
- Missing odds create no picks.
- Paper results must be settled from later local raw files once `FTR` is available.
- Current-season outcomes are never used for model fitting or optimization.

## Why Paper Only
This is a research-only candidate with no confirmed edge claim. It uses flat 1u paper stakes, never places bets, and must not be optimized on paper results.

## SportsEdge Next Step
Expose the ledger and latest HTML report as read-only SportsEdge artifacts, keeping order placement disabled and preserving the `research_only` label.

## Warnings
- none
