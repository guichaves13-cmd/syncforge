"""Auto-update checker — polls a `latest.json` manifest hosted on GitHub Releases
(or anywhere accessible by HTTPS GET) and reports the latest available version.

Does NOT auto-apply updates — it only reports. The frontend shows a banner;
the user clicks to download. This avoids accidental restarts mid-render.

Manifest schema (`latest.json`):
{
    "version":      "0.5.0",
    "released_at":  "2026-06-23",
    "min_required": "0.1.0",
    "channels": {
        "stable":  { "url": "...", "sha256": "...", "size": 12345 },
        "preview": { "url": "...", "sha256": "...", "size": 12345 }
    },
    "notes_url":    "https://github.com/.../releases/tag/v0.5.0"
}
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests


CURRENT_VERSION = "0.5.0"
DEFAULT_MANIFEST_URL = (
    "https://raw.githubusercontent.com/syncforge/syncforge/main/latest.json"
)
DEFAULT_CHANNEL = "stable"


@dataclass
class UpdateInfo:
    current: str
    latest: str
    update_available: bool
    is_required: bool
    channel: str
    download_url: str = ""
    sha256: str = ""
    size: int = 0
    notes_url: str = ""
    released_at: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current, "latest": self.latest,
            "update_available": self.update_available,
            "is_required": self.is_required,
            "channel": self.channel, "download_url": self.download_url,
            "sha256": self.sha256, "size": self.size,
            "notes_url": self.notes_url, "released_at": self.released_at,
            "error": self.error,
        }


_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")


def parse_version(s: str) -> tuple[int, int, int]:
    """Lenient semver parser. Returns (major, minor, patch). Non-numeric → (0,0,0)."""
    m = _SEMVER.match((s or "").strip())
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def is_newer(latest: str, current: str) -> bool:
    return parse_version(latest) > parse_version(current)


def check_for_update(
    *,
    current_version: str = CURRENT_VERSION,
    channel: str = DEFAULT_CHANNEL,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    timeout: float = 5.0,
    fetcher: Callable[[str, float], dict[str, Any]] | None = None,
) -> UpdateInfo:
    """Fetch manifest and compare to current_version.

    `fetcher` lets tests inject a mock that returns the manifest dict directly.
    """
    try:
        manifest = fetcher(manifest_url, timeout) if fetcher else _http_fetch(manifest_url, timeout)
    except Exception as e:
        return UpdateInfo(current=current_version, latest=current_version,
                           update_available=False, is_required=False,
                           channel=channel, error=str(e))

    latest = str(manifest.get("version", "")).strip()
    min_required = str(manifest.get("min_required", "")).strip()
    ch_info = (manifest.get("channels") or {}).get(channel, {})

    info = UpdateInfo(
        current=current_version,
        latest=latest or current_version,
        update_available=bool(latest) and is_newer(latest, current_version),
        is_required=bool(min_required) and is_newer(min_required, current_version),
        channel=channel,
        download_url=str(ch_info.get("url", "")),
        sha256=str(ch_info.get("sha256", "")),
        size=int(ch_info.get("size", 0) or 0),
        notes_url=str(manifest.get("notes_url", "")),
        released_at=str(manifest.get("released_at", "")),
    )
    return info


def _http_fetch(url: str, timeout: float) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout, headers={"Cache-Control": "no-cache"})
    r.raise_for_status()
    return r.json()
