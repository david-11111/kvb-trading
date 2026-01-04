"""
轻量 JSONL 日志工具：用于记录交易决策/下单尝试/结果，便于排查为何没有实际下单。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _coerce_payload(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    return payload


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: Dict[str, Any]):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def data_dir_for(file_path: str) -> Path:
    return Path(file_path).resolve().parent / "trade_data"


def dated_jsonl_path(dir_path: Path, prefix: str, when: Optional[datetime] = None) -> Path:
    when = when or datetime.now()
    return dir_path / f"{prefix}_{when.strftime('%Y%m%d')}.jsonl"


def now_event(kind: str, **fields: Any) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "ts": datetime.now().timestamp(),
        "datetime": datetime.now().isoformat(),
        "kind": kind,
    }
    for k, v in fields.items():
        event[k] = _coerce_payload(v)
    return event

