from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    description: str

    @property
    def path(self) -> Path:
        return ROOT / self.script

    def command(self) -> list[str]:
        return [sys.executable, self.script]


STEPS = {
    "inventory": [
        Step("data_inventory", "scripts/build_data_inventory.py", "Inventory local football data files and schemas."),
    ],
    "registry": [
        Step("data_inventory", "scripts/build_data_inventory.py", "Inventory local football data files and schemas."),
        Step("footiqo_canonical_registry", "scripts/build_footiqo_canonical_registry_prototype.py", "Build Footiqo top-5 canonical match registry prototype."),
    ],
    "super_csvs": [
        Step("footiqo_super_csv_prototypes", "scripts/build_footiqo_super_csv_prototypes.py", "Build Footiqo market super CSV prototypes."),
        Step("review_footiqo_super_csvs", "scripts/review_footiqo_super_csv_prototypes.py", "Review prototypes and write research-ready Footiqo market CSVs."),
    ],
    "clubelo": [
        Step("clubelo_feature_block", "scripts/build_clubelo_feature_block.py", "Build ClubElo feature block."),
        Step("lock_clubelo_aliases", "scripts/lock_clubelo_aliases.py", "Apply locked ClubElo aliases and rebuild locked ClubElo block."),
    ],
    "entity_registry": [
        Step("entity_registry_v1", "scripts/build_entity_registry_v1.py", "Build entity registry v1 from canonical registry and aliases."),
        Step("lock_entity_registry_aliases", "scripts/lock_entity_registry_aliases_v1.py", "Apply confirmed alias decisions and write locked entity registry."),
    ],
    "understat": [
        Step("understat_locked_feature_block", "scripts/rebuild_understat_feature_block_locked.py", "Build locked Understat lagged feature block using locked entity aliases."),
    ],
    "merges": [
        Step("merge_clubelo", "scripts/merge_clubelo_into_research_super_csvs.py", "Merge locked ClubElo block into research-ready super CSVs."),
        Step("merge_understat", "scripts/merge_understat_into_clubelo_super_csvs.py", "Merge locked Understat block into ClubElo-enhanced research CSVs."),
    ],
    "audits": [
        Step("clubelo_predictive_audit", "scripts/run_clubelo_multimarket_predictive_audit.py", "Run ClubElo predictive audit."),
        Step("understat_predictive_audit", "scripts/run_understat_controlled_predictive_audit.py", "Run controlled Understat predictive audit."),
    ],
}

ALL_ORDER = [
    "registry",
    "super_csvs",
    "clubelo",
    "entity_registry",
    "understat",
    "merges",
]


def unique_steps(stage: str, skip_audits: bool) -> list[Step]:
    if stage == "all":
        stages = list(ALL_ORDER)
        if not skip_audits:
            print("Audits are not included in --stage all by default. Run --stage audits explicitly for predictive audits.")
    else:
        stages = [stage]
    out: list[Step] = []
    seen: set[str] = set()
    for s in stages:
        if s == "audits" and skip_audits:
            continue
        for step in STEPS[s]:
            if step.script not in seen:
                seen.add(step.script)
                out.append(step)
    return out


def validate_steps(steps: list[Step], dry_run: bool) -> bool:
    missing = [step for step in steps if not step.path.exists()]
    if not missing:
        return True
    print("Missing required pipeline scripts:")
    for step in missing:
        print(f"- {step.script} ({step.name})")
    if dry_run:
        print("Dry run only: missing scripts reported but execution was not attempted.")
        return True
    return False


def run_steps(steps: list[Step], dry_run: bool) -> int:
    if not validate_steps(steps, dry_run):
        return 2
    print("Pipeline plan:")
    for i, step in enumerate(steps, start=1):
        cmd = " ".join(step.command())
        print(f"{i}. {step.name}: {cmd}")
        print(f"   {step.description}")
    if dry_run:
        print("Dry run complete. No scripts were executed.")
        return 0
    for step in steps:
        print(f"\n==> Running {step.name}: {' '.join(step.command())}", flush=True)
        result = subprocess.run(step.command(), cwd=ROOT)
        if result.returncode != 0:
            print(f"Step failed: {step.name} exited with code {result.returncode}", file=sys.stderr)
            return result.returncode
    print("\nPipeline completed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely rebuild approved research-only football data pipeline stages."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "registry", "super_csvs", "clubelo", "entity_registry", "understat", "merges", "audits"],
        default="all",
        help="Pipeline stage to run. --stage all excludes predictive audits; use --stage audits explicitly.",
    )
    parser.add_argument("--skip-audits", action="store_true", help="Skip audit steps if a selected stage would include them.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and validate script paths without execution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = unique_steps(args.stage, args.skip_audits)
    if not steps:
        print("No steps selected.")
        return 0
    return run_steps(steps, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
