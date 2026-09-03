from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.research.v3_next import DEFAULT_CONFIG, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nested, market-anchored V3 challenger research pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quick", action="store_true", help="Run only the first three model candidates for a smoke test.")
    args = parser.parse_args()
    decision = run(args.config, quick=args.quick)
    print(decision)


if __name__ == "__main__":
    main()
