"""Phase 5 — distribution: updater + entry point + bundled paths."""
from __future__ import annotations
import importlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient


# ─── module-level imports work ──────────────────────────────────────────

@pytest.mark.parametrize("mod", [
    "app.updater.check",
    "app.__main__",
])
def test_phase5_modules_import(mod):
    importlib.import_module(mod)


# ─── semver parsing ─────────────────────────────────────────────────────

def test_parse_version_basic():
    from app.updater.check import parse_version
    assert parse_version("0.5.0") == (0, 5, 0)
    assert parse_version("12.34.56") == (12, 34, 56)
    assert parse_version("1.2.3-beta.4") == (1, 2, 3)
    assert parse_version("1.2.3+build5") == (1, 2, 3)


def test_parse_version_invalid_returns_zero():
    from app.updater.check import parse_version
    assert parse_version("") == (0, 0, 0)
    assert parse_version("garbage") == (0, 0, 0)
    assert parse_version("1.2") == (0, 0, 0)
    assert parse_version("v1.2.3") == (0, 0, 0)  # leading 'v' not allowed
    assert parse_version(None) == (0, 0, 0)


def test_is_newer():
    from app.updater.check import is_newer
    assert is_newer("0.5.1", "0.5.0") is True
    assert is_newer("0.6.0", "0.5.99") is True
    assert is_newer("1.0.0", "0.99.99") is True
    assert is_newer("0.5.0", "0.5.0") is False
    assert is_newer("0.4.9", "0.5.0") is False


# ─── check_for_update with mocked fetcher ───────────────────────────────

def _manifest(latest="0.6.0", min_required="0.1.0",
              channels=None, notes_url="https://example.com/notes"):
    return {
        "version": latest,
        "min_required": min_required,
        "channels": channels or {
            "stable": {"url": "https://x/stable.zip",
                       "sha256": "abc123", "size": 12345}
        },
        "notes_url": notes_url,
        "released_at": "2026-06-23",
    }


def test_update_available_when_newer():
    from app.updater.check import check_for_update
    info = check_for_update(
        current_version="0.5.0", channel="stable",
        fetcher=lambda url, timeout: _manifest(latest="0.6.0"),
    )
    assert info.update_available is True
    assert info.latest == "0.6.0"
    assert info.is_required is False
    assert info.download_url == "https://x/stable.zip"
    assert info.sha256 == "abc123"
    assert info.size == 12345
    assert info.notes_url == "https://example.com/notes"


def test_no_update_when_equal():
    from app.updater.check import check_for_update
    info = check_for_update(
        current_version="0.6.0",
        fetcher=lambda url, t: _manifest(latest="0.6.0"),
    )
    assert info.update_available is False


def test_is_required_when_min_required_newer():
    from app.updater.check import check_for_update
    info = check_for_update(
        current_version="0.5.0",
        fetcher=lambda url, t: _manifest(latest="1.0.0", min_required="0.9.0"),
    )
    assert info.update_available is True
    assert info.is_required is True


def test_fetcher_exception_returns_error_field():
    from app.updater.check import check_for_update
    def boom(url, timeout): raise RuntimeError("network down")
    info = check_for_update(
        current_version="0.5.0",
        fetcher=boom,
    )
    assert info.update_available is False
    assert "network down" in info.error
    # Returns gracefully with current_version as latest
    assert info.latest == "0.5.0"


def test_missing_channel_returns_empty_download_url():
    from app.updater.check import check_for_update
    info = check_for_update(
        current_version="0.5.0", channel="preview",
        fetcher=lambda url, t: _manifest(channels={"stable": {"url": "x"}}),
    )
    assert info.update_available is True   # version is still newer
    assert info.download_url == ""         # but no URL for preview channel


def test_to_dict_serializable():
    from app.updater.check import check_for_update
    info = check_for_update(
        current_version="0.5.0",
        fetcher=lambda url, t: _manifest(),
    )
    d = info.to_dict()
    json.dumps(d)  # must round-trip JSON
    assert d["update_available"] is True


# ─── /api/updates/check endpoint ────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCFORGE_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("SYNCFORGE_JWT_SECRET", "test-jwt-secret")
    monkeypatch.setenv("LICENSE_SECRET", "test-license-secret")
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, main


def test_health_reports_current_version(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    # Version should be a semver-like string
    from app.updater.check import parse_version, CURRENT_VERSION
    assert body["version"] == CURRENT_VERSION
    assert parse_version(body["version"]) != (0, 0, 0)


def test_updates_endpoint_returns_dict(client, monkeypatch):
    # Patch the fetcher to avoid hitting the real network
    monkeypatch.setattr(
        "app.updater.check._http_fetch",
        lambda url, timeout: _manifest(latest="999.0.0"),
    )
    c, _ = client
    r = c.get("/api/updates/check")
    assert r.status_code == 200
    body = r.json()
    assert body["update_available"] is True
    assert body["latest"] == "999.0.0"
    assert body["channel"] == "stable"


def test_updates_endpoint_network_failure_is_graceful(client, monkeypatch):
    def boom(url, timeout): raise RuntimeError("offline")
    monkeypatch.setattr("app.updater.check._http_fetch", boom)
    c, _ = client
    r = c.get("/api/updates/check")
    assert r.status_code == 200
    body = r.json()
    assert body["update_available"] is False
    assert "offline" in body["error"]


# ─── distribution artifacts exist ──────────────────────────────────────

def test_distribution_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "LICENSE").exists()
    assert (root / "LICENSE").read_text(encoding="utf-8").startswith("SyncForge")
    assert (root / "dist" / "syncforge.spec").exists()
    assert (root / "dist" / "build.bat").exists()
    assert (root / "dist" / "install.bat").exists()
    assert (root / ".github" / "workflows" / "ci.yml").exists()
    assert (root / "docs" / "getting-started.md").exists()


def test_main_entrypoint_exists():
    """`python -m app` must resolve to the FastAPI launcher."""
    root = Path(__file__).resolve().parents[1]
    main_py = root / "backend" / "app" / "__main__.py"
    assert main_py.exists()
    src = main_py.read_text(encoding="utf-8")
    assert "uvicorn.run" in src
    assert "app.main:app" in src


def test_ci_workflow_has_matrix_and_pytest():
    root = Path(__file__).resolve().parents[1]
    yml = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest" in yml
    assert "ubuntu-latest" in yml
    assert "windows-latest" in yml
    assert "ffmpeg" in yml
    assert "npm run build" in yml
