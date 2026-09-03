#!/usr/bin/env python3
"""Run the final V5 E0 mapping-coverage audit without changing V1 V5 outputs."""
from src.v5_betfair.mapping_coverage_audit import run

if __name__ == "__main__":
    print(run())
