# Rebuild Guide

Use the one-command entrypoint for ordinary rebuilds:

```bash
python scripts/rebuild_research_pipeline.py --stage all --skip-audits
```

Use `--dry-run` first to inspect commands without executing them.

## Stage Order

1. Data inventory

   Command:

   ```bash
   python scripts/build_data_inventory.py
   ```

   Expected outputs include:

   - `outputs/reports/data_inventory/full_file_inventory.csv`
   - `outputs/reports/data_inventory/source_schema_inventory.csv`
   - `outputs/reports/data_inventory/data_inventory_summary.md`

2. Footiqo canonical registry prototype

   ```bash
   python scripts/build_footiqo_canonical_registry_prototype.py
   ```

   Expected outputs include:

   - `data/processed/match_registry/competition_registry_v1_prototype.csv`
   - `data/processed/match_registry/canonical_match_registry_v1_prototype.csv`
   - `data/processed/match_registry/source_match_map_v1_prototype.csv`

3. Footiqo super CSV prototypes

   ```bash
   python scripts/build_footiqo_super_csv_prototypes.py
   ```

   Expected outputs include:

   - `data/processed/super_csvs/prototype/super_btts_footiqo_top5_v1.csv`
   - `data/processed/super_csvs/prototype/super_ou25_footiqo_top5_v1.csv`

4. Review/promote research-ready super CSVs

   ```bash
   python scripts/review_footiqo_super_csv_prototypes.py
   ```

   Expected outputs include:

   - `data/processed/super_csvs/research_ready/super_btts_footiqo_top5_research_v1.csv`
   - `data/processed/super_csvs/research_ready/super_ou25_footiqo_top5_research_v1.csv`

5. ClubElo feature block

   ```bash
   python scripts/build_clubelo_feature_block.py
   ```

   Expected outputs include:

   - `data/processed/feature_blocks/clubelo/clubelo_features_footiqo_top5_v1.csv`
   - `data/processed/feature_blocks/clubelo/clubelo_team_alias_draft.csv`

6. Lock ClubElo aliases

   ```bash
   python scripts/lock_clubelo_aliases.py
   ```

   Expected outputs include:

   - `data/processed/feature_blocks/clubelo/clubelo_team_alias_locked_v1.csv`
   - `data/processed/feature_blocks/clubelo/clubelo_features_footiqo_top5_v1_locked.csv`

7. Merge ClubElo into research CSVs

   ```bash
   python scripts/merge_clubelo_into_research_super_csvs.py
   ```

   Expected outputs include:

   - `data/processed/super_csvs/research_ready_plus/clubelo/super_btts_footiqo_top5_clubelo_research_v1.csv`
   - `data/processed/super_csvs/research_ready_plus/clubelo/super_ou25_footiqo_top5_clubelo_research_v1.csv`

8. Entity registry

   ```bash
   python scripts/build_entity_registry_v1.py
   ```

   Expected outputs include:

   - `data/processed/entity_registry/teams_v1.csv`
   - `data/processed/entity_registry/team_aliases_v1.csv`
   - `data/processed/entity_registry/matches_v1.csv`

9. Lock entity registry

   ```bash
   python scripts/lock_entity_registry_aliases_v1.py
   ```

   Expected outputs include:

   - `data/processed/entity_registry/teams_v1_locked.csv`
   - `data/processed/entity_registry/team_aliases_v1_locked.csv`
   - `data/processed/entity_registry/matches_v1_locked.csv`

10. Understat locked feature block

   ```bash
   python scripts/rebuild_understat_feature_block_locked.py
   ```

   Expected outputs include:

   - `data/processed/feature_blocks/understat/understat_features_footiqo_top5_v1_locked.csv`
   - `outputs/reports/feature_blocks/understat/understat_locked_leakage_checks.csv`

11. Merge Understat

   ```bash
   python scripts/merge_understat_into_clubelo_super_csvs.py
   ```

   Expected outputs include:

   - `data/processed/super_csvs/research_ready_plus/clubelo_understat/super_btts_footiqo_top5_clubelo_understat_research_v1.csv`
   - `data/processed/super_csvs/research_ready_plus/clubelo_understat/super_ou25_footiqo_top5_clubelo_understat_research_v1.csv`

12. Predictive audits

   ```bash
   python scripts/run_clubelo_multimarket_predictive_audit.py
   python scripts/run_understat_controlled_predictive_audit.py
   ```

   Expected outputs include:

   - `outputs/reports/clubelo_predictive/clubelo_predictive_decision.md`
   - `outputs/reports/understat_predictive/understat_predictive_decision.md`

Predictive audits are intentionally excluded from default `--skip-audits` rebuilds because they are slower and do not alter canonical data tables.
