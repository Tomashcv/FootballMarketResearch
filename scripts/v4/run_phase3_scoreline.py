#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v4.models.dynamic_scoreline import run_phase3
if __name__=="__main__": print(json.dumps(run_phase3(),indent=2))
