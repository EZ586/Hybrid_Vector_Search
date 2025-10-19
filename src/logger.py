# src/logger.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any


def append_jsonl(row: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
