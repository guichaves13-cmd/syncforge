"""Append-only audit log — JSONL, one event per line.

Records every action that affects state: signup, login, job creation,
subscription change, etc. Critical for security review and customer support.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from threading import Lock


_LOCK = Lock()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, event: str, actor: str = "system",
               metadata: dict | None = None,
               ip: str | None = None) -> None:
        line = {
            "ts": time.time(),
            "event": event,
            "actor": actor,
            "ip": ip,
            "metadata": metadata or {},
        }
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, separators=(",", ":")) + "\n")

    def tail(self, n: int = 100, *, actor: str | None = None,
             event: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict] = []
        for raw in reversed(lines):
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if actor and row.get("actor") != actor:
                continue
            if event and row.get("event") != event:
                continue
            out.append(row)
            if len(out) >= n:
                break
        return out
