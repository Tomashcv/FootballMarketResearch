#!/usr/bin/env python3
"""Run the read-only V4 Phase 1 market-data contract audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4.data.phase1_audit import run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
