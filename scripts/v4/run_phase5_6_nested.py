#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.v4.validation.nested import run_nested
if __name__=="__main__": print(json.dumps(run_nested(),indent=2))
