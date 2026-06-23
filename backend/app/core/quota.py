"""Per-user quota tracking — videos this month, storage GB used.

File-backed JSON, monthly reset based on YYYY-MM key. Swap for Redis/Postgres
when scaling.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


_LOCK = Lock()


@dataclass
class QuotaCheck:
    allowed: bool
    reason: str = ""
    used_videos: int = 0
    limit_videos: int = 0


class QuotaStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    @staticmethod
    def _month_key(t: float | None = None) -> str:
        d = datetime.fromtimestamp(t or time.time(), tz=timezone.utc)
        return d.strftime("%Y-%m")

    def check(self, user_id: str, limit_videos: int) -> QuotaCheck:
        """Returns whether this user can create another video this month."""
        data = self._load()
        key = self._month_key()
        used = data.get(user_id, {}).get(key, {}).get("videos", 0)
        if used >= limit_videos:
            return QuotaCheck(
                allowed=False,
                reason=f"monthly quota exhausted ({used}/{limit_videos})",
                used_videos=used, limit_videos=limit_videos,
            )
        return QuotaCheck(allowed=True, used_videos=used, limit_videos=limit_videos)

    def record_video(self, user_id: str, bytes_used: int = 0) -> None:
        with _LOCK:
            data = self._load()
            key = self._month_key()
            user = data.setdefault(user_id, {})
            month = user.setdefault(key, {"videos": 0, "bytes": 0})
            month["videos"] = int(month.get("videos", 0)) + 1
            month["bytes"] = int(month.get("bytes", 0)) + max(0, int(bytes_used))
            self._save(data)

    def stats(self, user_id: str) -> dict:
        data = self._load().get(user_id, {})
        key = self._month_key()
        return data.get(key, {"videos": 0, "bytes": 0})

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
