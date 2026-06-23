"""Phase 4 ADVANCED — auth, license, quota, audit, billing, end-to-end via FastAPI.

Hard cases:
  • password hashing roundtrip + tampering rejection
  • JWT signing/verification + expired token + bad sig + tampered payload
  • license tampering (every field) → rejected
  • license lifetime (expires=0) → never expires
  • quota monthly rollover + concurrent records
  • audit log filtering + ordering
  • Stripe webhook signature verification
  • Full REST flow: register → login → me → create job → quota exhausted
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import importlib
import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# Reset env so secrets are deterministic across tests
os.environ.setdefault("SYNCFORGE_JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("LICENSE_SECRET", "test-license-secret")
os.environ.setdefault("STRIPE_WEBHOOK_SECRET", "")  # disable webhook by default


# ─── module-level imports work ──────────────────────────────────────────

@pytest.mark.parametrize("mod", [
    "app.auth.users",
    "app.billing.stripe_handler",
    "app.core.audit",
    "app.core.license",
    "app.core.quota",
])
def test_phase4_modules_import(mod):
    importlib.import_module(mod)


# ─────────────────────────────────────────────────────────────────────
# 1. Password hashing — slow but verifiable
# ─────────────────────────────────────────────────────────────────────

def test_user_password_roundtrip(tmp_path):
    from app.auth.users import UserStore
    store = UserStore(tmp_path / "users.json")
    u = store.create("alice@example.com", "correct-horse-battery-staple")
    assert store.verify("alice@example.com", "correct-horse-battery-staple") is not None
    assert store.verify("alice@example.com", "wrong-password") is None
    assert store.verify("alice@example.com", "") is None
    # case-insensitive email lookup
    assert store.verify("ALICE@example.com", "correct-horse-battery-staple") is not None


def test_user_duplicate_email_rejected(tmp_path):
    from app.auth.users import UserStore
    store = UserStore(tmp_path / "users.json")
    store.create("dup@x.com", "password1234")
    with pytest.raises(ValueError, match="already registered"):
        store.create("dup@x.com", "anotherpass")
    # mixed-case duplicate also rejected
    with pytest.raises(ValueError):
        store.create("DUP@x.com", "yetanother")


def test_user_public_strips_secrets(tmp_path):
    from app.auth.users import UserStore
    u = UserStore(tmp_path / "u.json").create("x@y.z", "abcdefgh")
    pub = u.public()
    assert "password_hash" not in pub
    assert "password_salt" not in pub
    assert pub["email"] == "x@y.z"


def test_user_upgrade_changes_plan_and_quota(tmp_path):
    from app.auth.users import UserStore, PLANS
    store = UserStore(tmp_path / "u.json")
    u = store.create("p@x.com", "passw0rd!")
    assert u.plan == "free"
    assert u.quota_videos_month == PLANS["free"]["videos_month"]
    upgraded = store.upgrade(u.id, "pro", stripe_customer_id="cus_123")
    assert upgraded.plan == "pro"
    assert upgraded.quota_videos_month == PLANS["pro"]["videos_month"]
    assert upgraded.stripe_customer_id == "cus_123"


# ─────────────────────────────────────────────────────────────────────
# 2. JWT-like tokens — tamper resistance
# ─────────────────────────────────────────────────────────────────────

def _make_user(tmp_path):
    from app.auth.users import UserStore
    return UserStore(tmp_path / "u.json").create("j@w.t", "letmeinplz")


def test_jwt_roundtrip(tmp_path):
    from app.auth.users import issue_token, verify_token
    u = _make_user(tmp_path)
    tok = issue_token(u, "secret-A")
    payload = verify_token(tok, "secret-A")
    assert payload is not None
    assert payload["sub"] == u.id
    assert payload["plan"] == "free"
    assert payload["exp"] > int(time.time())


def test_jwt_rejects_wrong_secret(tmp_path):
    from app.auth.users import issue_token, verify_token
    tok = issue_token(_make_user(tmp_path), "secret-A")
    assert verify_token(tok, "secret-B") is None


def test_jwt_rejects_tampered_payload(tmp_path):
    from app.auth.users import issue_token, verify_token
    tok = issue_token(_make_user(tmp_path), "secret")
    h, p, s = tok.split(".")
    # Swap payload for a forged one claiming pro tier
    forged_payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "attacker", "plan": "team",
                    "exp": int(time.time()) + 999999}).encode()
    ).rstrip(b"=").decode()
    forged = f"{h}.{forged_payload}.{s}"
    assert verify_token(forged, "secret") is None


def test_jwt_rejects_expired(tmp_path):
    """Forge a token with already-past exp; signature is valid but it must reject."""
    from app.auth.users import _b64, verify_token
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(json.dumps({"sub": "x", "plan": "free",
                                "exp": int(time.time()) - 60}).encode())
    sig = _b64(hmac.new(b"k", f"{header}.{payload}".encode(),
                         hashlib.sha256).digest())
    tok = f"{header}.{payload}.{sig}"
    assert verify_token(tok, "k") is None


def test_jwt_rejects_malformed():
    from app.auth.users import verify_token
    assert verify_token("not.a.real.token.too-many-parts", "x") is None
    assert verify_token("only.two", "x") is None
    assert verify_token("", "x") is None


# ─────────────────────────────────────────────────────────────────────
# 3. License tampering matrix — every field individually mutated
# ─────────────────────────────────────────────────────────────────────

def test_license_roundtrip(monkeypatch):
    monkeypatch.setenv("LICENSE_SECRET", "S3CR3T")
    from app.core.license import issue_license, verify_license
    importlib.reload(__import__("app.core.license", fromlist=["x"]))
    from app.core.license import issue_license, verify_license

    expires = int(time.time()) + 86400
    key = issue_license("pro", "user12345678", expires)
    lic = verify_license(key)
    assert lic.valid is True
    assert lic.plan == "pro"
    assert lic.expires_at == expires


def test_license_lifetime_never_expires():
    from app.core.license import issue_license, verify_license
    key = issue_license("team", "userlife01", expires_at=0)
    lic = verify_license(key)
    assert lic.valid is True
    assert lic.expires_at == 0


def test_license_expired_rejected():
    from app.core.license import issue_license, verify_license
    past = int(time.time()) - 86400
    key = issue_license("pro", "userpastxx", expires_at=past)
    lic = verify_license(key)
    assert lic.valid is False
    assert "expired" in lic.reason


def test_license_tampered_signature_rejected():
    from app.core.license import issue_license, verify_license
    key = issue_license("pro", "tampered01", expires_at=int(time.time()) + 999)
    # mutate the signature: flip last char
    bad = key[:-1] + ("a" if key[-1] != "a" else "b")
    lic = verify_license(bad)
    assert lic.valid is False
    assert "signature mismatch" in lic.reason


def test_license_tampered_plan_rejected():
    from app.core.license import issue_license, verify_license
    key = issue_license("free", "upgrademe", expires_at=0)
    # try to promote 'free' → 'pro' in the key body
    bad = key.replace("-free-", "-pro-", 1)
    assert verify_license(bad).valid is False


def test_license_tampered_expiry_rejected():
    from app.core.license import issue_license, verify_license
    key = issue_license("pro", "extendme1", expires_at=int(time.time()) + 60)
    # try to push expiry far into the future without resigning
    parts = key.split("-", 4)
    parts[3] = str(int(parts[3]) + 10_000_000)
    bad = "-".join(parts)
    assert verify_license(bad).valid is False


def test_license_malformed_returns_invalid():
    from app.core.license import verify_license
    for bad in ("", "junk", "SF-too-few", "AB-pro-user-0-sig", "SF-evil-x-0-sig"):
        lic = verify_license(bad)
        assert lic.valid is False


def test_license_unknown_plan_rejected():
    from app.core.license import issue_license
    with pytest.raises(ValueError, match="invalid plan"):
        issue_license("super-duper", "user", 0)


# ─────────────────────────────────────────────────────────────────────
# 4. Quota — monthly rollover + multiple records
# ─────────────────────────────────────────────────────────────────────

def test_quota_allows_under_limit(tmp_path):
    from app.core.quota import QuotaStore
    qs = QuotaStore(tmp_path / "q.json")
    chk = qs.check("u1", limit_videos=5)
    assert chk.allowed is True
    assert chk.used_videos == 0


def test_quota_blocks_at_and_above_limit(tmp_path):
    from app.core.quota import QuotaStore
    qs = QuotaStore(tmp_path / "q.json")
    for _ in range(5):
        qs.record_video("u1")
    chk = qs.check("u1", limit_videos=5)
    assert chk.allowed is False
    assert chk.used_videos == 5
    assert "exhausted" in chk.reason


def test_quota_isolates_users(tmp_path):
    from app.core.quota import QuotaStore
    qs = QuotaStore(tmp_path / "q.json")
    qs.record_video("alice")
    qs.record_video("alice")
    assert qs.stats("alice")["videos"] == 2
    assert qs.stats("bob")["videos"] == 0


def test_quota_monthly_rollover(tmp_path):
    """Mocks _month_key to simulate month change → fresh counter."""
    from app.core.quota import QuotaStore
    qs = QuotaStore(tmp_path / "q.json")
    with mock.patch.object(QuotaStore, "_month_key", staticmethod(lambda t=None: "2024-01")):
        qs.record_video("u1")
        qs.record_video("u1")
        assert qs.check("u1", 5).used_videos == 2
    with mock.patch.object(QuotaStore, "_month_key", staticmethod(lambda t=None: "2024-02")):
        # New month → counter resets
        assert qs.check("u1", 5).used_videos == 0


def test_quota_persistence_survives_reopen(tmp_path):
    from app.core.quota import QuotaStore
    path = tmp_path / "q.json"
    QuotaStore(path).record_video("u1")
    QuotaStore(path).record_video("u1")
    qs2 = QuotaStore(path)
    assert qs2.stats("u1")["videos"] == 2


# ─────────────────────────────────────────────────────────────────────
# 5. Audit log — filtering + ordering + persistence
# ─────────────────────────────────────────────────────────────────────

def test_audit_records_in_reverse_chronological_order(tmp_path):
    from app.core.audit import AuditLog
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(event="job.created", actor="alice")
    log.record(event="user.login", actor="bob")
    log.record(event="job.failed", actor="alice")
    tail = log.tail(n=10)
    assert len(tail) == 3
    # tail() returns newest first
    assert tail[0]["event"] == "job.failed"
    assert tail[2]["event"] == "job.created"


def test_audit_filter_by_actor(tmp_path):
    from app.core.audit import AuditLog
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(event="x", actor="alice")
    log.record(event="x", actor="bob")
    log.record(event="x", actor="alice")
    only_alice = log.tail(actor="alice")
    assert len(only_alice) == 2
    assert all(r["actor"] == "alice" for r in only_alice)


def test_audit_filter_by_event(tmp_path):
    from app.core.audit import AuditLog
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(event="job.created", actor="x")
    log.record(event="user.login", actor="x")
    only_login = log.tail(event="user.login")
    assert len(only_login) == 1
    assert only_login[0]["event"] == "user.login"


def test_audit_tail_respects_n(tmp_path):
    from app.core.audit import AuditLog
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(20):
        log.record(event="e", actor=str(i))
    assert len(log.tail(n=5)) == 5


def test_audit_handles_corrupt_lines(tmp_path):
    """A garbage line in the middle must not break the read."""
    from app.core.audit import AuditLog
    p = tmp_path / "a.jsonl"
    log = AuditLog(p)
    log.record(event="a", actor="x")
    with open(p, "a", encoding="utf-8") as f:
        f.write("this is not json\n")
    log.record(event="b", actor="x")
    tail = log.tail()
    assert {r["event"] for r in tail} == {"a", "b"}


# ─────────────────────────────────────────────────────────────────────
# 6. Stripe webhook — signature verification
# ─────────────────────────────────────────────────────────────────────

def test_stripe_webhook_rejects_bad_signature():
    from app.billing.stripe_handler import verify_webhook
    body = b'{"id":"evt_1","type":"checkout.session.completed"}'
    assert verify_webhook(body, "sig-not-valid", "secret-x") is None


def test_stripe_webhook_disabled_when_no_secret():
    """No SECRET configured → reject everything (security default)."""
    from app.billing.stripe_handler import verify_webhook
    body = b'{"id":"evt"}'
    assert verify_webhook(body, "any", "") is None


def test_stripe_checkout_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("STRIPE_API_KEY", raising=False)
    from app.billing.stripe_handler import create_checkout_session
    sess = create_checkout_session(
        user_id="u1", user_email="x@y.z", plan="pro",
        success_url="http://x", cancel_url="http://x",
    )
    assert sess is None


# ─────────────────────────────────────────────────────────────────────
# 7. End-to-end via FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    """A clean app instance with isolated storage."""
    monkeypatch.setenv("SYNCFORGE_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("SYNCFORGE_JWT_SECRET", "test-jwt-secret")
    monkeypatch.setenv("LICENSE_SECRET", "test-license-secret")
    # Re-import main so it picks up the patched env
    import importlib, app.main as main
    importlib.reload(main)
    main.JOBS.clear()
    # Stub the heavy runner so create_job doesn't actually generate a video
    async def fake_run(jid, req):
        from app.main import publish, JOBS
        JOBS[jid].status = "done"
        await publish(jid, {"event": "done", "result": {"ok": True}})
    monkeypatch.setattr(main, "_run_job", fake_run)
    with TestClient(main.app) as c:
        yield c, main


def test_e2e_register_login_me(client):
    c, _ = client
    r = c.post("/api/auth/register",
               json={"email": "e2e@x.com", "password": "longenough1"})
    assert r.status_code == 200
    data = r.json()
    token = data["token"]
    assert data["user"]["email"] == "e2e@x.com"
    assert "password_hash" not in data["user"]

    # /me with valid token
    me = c.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    body = me.json()
    assert body["user"]["email"] == "e2e@x.com"
    assert body["limits"]["videos_month"] == 5

    # /me without token → 401
    assert c.get("/api/auth/me").status_code == 401


def test_e2e_login_rejects_wrong_password(client):
    c, _ = client
    c.post("/api/auth/register",
            json={"email": "u@x.com", "password": "longpass1"})
    bad = c.post("/api/auth/login",
                  json={"email": "u@x.com", "password": "WRONG-PASS"})
    assert bad.status_code == 401


def test_e2e_register_rejects_short_password(client):
    c, _ = client
    r = c.post("/api/auth/register",
               json={"email": "x@y.z", "password": "short"})
    assert r.status_code == 400


def test_e2e_duplicate_register_returns_409(client):
    c, _ = client
    c.post("/api/auth/register", json={"email": "d@x.com", "password": "longpass1"})
    r = c.post("/api/auth/register", json={"email": "d@x.com", "password": "anotherpw"})
    assert r.status_code == 409


def test_e2e_job_creation_enforces_quota(client, monkeypatch):
    c, main = client
    reg = c.post("/api/auth/register",
                  json={"email": "q@x.com", "password": "longpass1"}).json()
    token = reg["token"]
    hdr = {"Authorization": f"Bearer {token}"}
    # free tier = 5 videos/month; create 5
    for _ in range(5):
        r = c.post("/api/jobs", json={"title": "T"}, headers=hdr)
        assert r.status_code == 200
    # 6th must be rate-limited
    over = c.post("/api/jobs", json={"title": "T"}, headers=hdr)
    assert over.status_code == 429
    assert "quota" in over.json()["detail"].lower()


def test_e2e_license_issue_and_verify(client):
    c, _ = client
    reg = c.post("/api/auth/register",
                  json={"email": "L@x.com", "password": "longpass1"}).json()
    token = reg["token"]
    user_id = reg["user"]["id"]
    hdr = {"Authorization": f"Bearer {token}"}
    # Issue license for self
    iss = c.get(f"/api/license/issue?plan=pro&user_id={user_id}&days=30",
                 headers=hdr)
    assert iss.status_code == 200
    key = iss.json()["key"]
    # Verify (no auth needed)
    ver = c.post(f"/api/license/verify?key={key}")
    assert ver.status_code == 200
    j = ver.json()
    assert j["valid"] is True
    assert j["plan"] == "pro"


def test_e2e_license_issue_forbidden_for_others(client):
    c, _ = client
    a = c.post("/api/auth/register",
                json={"email": "a@x.com", "password": "longpass1"}).json()
    b = c.post("/api/auth/register",
                json={"email": "b@x.com", "password": "longpass1"}).json()
    # User A tries to issue a license for User B
    r = c.get(f"/api/license/issue?plan=pro&user_id={b['user']['id']}&days=30",
               headers={"Authorization": f"Bearer {a['token']}"})
    assert r.status_code == 403


def test_e2e_audit_records_user_actions(client):
    c, main = client
    c.post("/api/auth/register",
            json={"email": "audit@x.com", "password": "longpass1"})
    tail = main.AUDIT.tail()
    assert any(r["event"] == "user.register" for r in tail)


def test_e2e_anonymous_can_create_jobs_when_auth_optional(client):
    """If SYNCFORGE_REQUIRE_AUTH is not set, jobs may be created anonymously."""
    c, _ = client
    r = c.post("/api/jobs", json={"title": "anon job"})
    assert r.status_code == 200


def test_e2e_auth_required_when_env_set(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("SYNCFORGE_REQUIRE_AUTH", "1")
    r = c.post("/api/jobs", json={"title": "x"})
    assert r.status_code == 401
