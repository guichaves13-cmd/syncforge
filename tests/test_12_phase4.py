"""Phase 4 — productization: auth + quota + audit + license + billing webhook."""
from __future__ import annotations
import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth.users import (
    PLANS, UserStore, issue_token, verify_token, _hash_password,
)
from app.billing.stripe_handler import handle_event, verify_webhook
from app.core.audit import AuditLog
from app.core.license import issue_license, verify_license
from app.core.quota import QuotaStore


# ─────────────────────────────────────────────────────────────────────
# UserStore
# ─────────────────────────────────────────────────────────────────────

def test_user_create_and_verify(tmp_path):
    store = UserStore(tmp_path / "users.json")
    u = store.create("alice@example.com", "supersecret123")
    assert u.email == "alice@example.com"
    assert u.plan == "free"
    assert u.quota_videos_month == PLANS["free"]["videos_month"]
    # Right password → user; wrong password → None
    assert store.verify("alice@example.com", "supersecret123").id == u.id
    assert store.verify("alice@example.com", "wrongwrong") is None


def test_user_duplicate_email_raises(tmp_path):
    store = UserStore(tmp_path / "users.json")
    store.create("bob@ex.com", "password1")
    with pytest.raises(ValueError, match="already registered"):
        store.create("bob@ex.com", "differentpw")


def test_user_password_hashing_is_salted(tmp_path):
    """Same password, different users → different hashes."""
    store = UserStore(tmp_path / "users.json")
    u1 = store.create("a@a.com", "samepass1234")
    u2 = store.create("b@b.com", "samepass1234")
    assert u1.password_hash != u2.password_hash  # different salts
    assert u1.password_salt != u2.password_salt


def test_user_upgrade_updates_quota(tmp_path):
    store = UserStore(tmp_path / "users.json")
    u = store.create("c@c.com", "password1")
    assert u.plan == "free"
    upd = store.upgrade(u.id, "pro", stripe_customer_id="cus_123")
    assert upd.plan == "pro"
    assert upd.quota_videos_month == PLANS["pro"]["videos_month"]
    assert upd.stripe_customer_id == "cus_123"


def test_user_persists_across_instances(tmp_path):
    p = tmp_path / "users.json"
    UserStore(p).create("d@d.com", "password1")
    # New instance reads same file
    fresh = UserStore(p)
    assert fresh.get_by_email("d@d.com") is not None


def test_user_public_omits_secrets(tmp_path):
    store = UserStore(tmp_path / "users.json")
    u = store.create("e@e.com", "password1")
    pub = u.public()
    assert "password_hash" not in pub
    assert "password_salt" not in pub
    assert pub["email"] == "e@e.com"


# ─────────────────────────────────────────────────────────────────────
# JWT-like tokens
# ─────────────────────────────────────────────────────────────────────

def test_token_roundtrip(tmp_path):
    store = UserStore(tmp_path / "u.json")
    u = store.create("x@x.com", "password1")
    tok = issue_token(u, "secret")
    payload = verify_token(tok, "secret")
    assert payload["sub"] == u.id
    assert payload["email"] == "x@x.com"


def test_token_rejects_wrong_secret(tmp_path):
    store = UserStore(tmp_path / "u.json")
    u = store.create("x@x.com", "password1")
    tok = issue_token(u, "secret-a")
    assert verify_token(tok, "secret-b") is None


def test_token_rejects_tampered_payload(tmp_path):
    store = UserStore(tmp_path / "u.json")
    u = store.create("x@x.com", "password1")
    tok = issue_token(u, "secret")
    # Tamper: flip a character in the payload section
    header, payload, sig = tok.split(".")
    bad = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    assert verify_token(f"{header}.{bad}.{sig}", "secret") is None


def test_token_rejects_expired(tmp_path):
    store = UserStore(tmp_path / "u.json")
    u = store.create("x@x.com", "password1")
    tok = issue_token(u, "secret", ttl_seconds=-1)
    assert verify_token(tok, "secret") is None


def test_token_rejects_malformed():
    assert verify_token("not.a.jwt", "secret") is None
    assert verify_token("", "secret") is None
    assert verify_token("only-two.parts", "secret") is None


# ─────────────────────────────────────────────────────────────────────
# Quota
# ─────────────────────────────────────────────────────────────────────

def test_quota_starts_at_zero(tmp_path):
    q = QuotaStore(tmp_path / "q.json")
    chk = q.check("user1", limit_videos=5)
    assert chk.allowed is True
    assert chk.used_videos == 0


def test_quota_enforces_limit(tmp_path):
    q = QuotaStore(tmp_path / "q.json")
    for _ in range(3):
        q.record_video("user1")
    chk = q.check("user1", limit_videos=3)
    assert chk.allowed is False
    assert "exhausted" in chk.reason
    assert chk.used_videos == 3


def test_quota_records_bytes(tmp_path):
    q = QuotaStore(tmp_path / "q.json")
    q.record_video("user1", bytes_used=1024 * 1024)
    q.record_video("user1", bytes_used=2048 * 1024)
    stats = q.stats("user1")
    assert stats["videos"] == 2
    assert stats["bytes"] == 3 * 1024 * 1024


def test_quota_isolates_users(tmp_path):
    q = QuotaStore(tmp_path / "q.json")
    for _ in range(5):
        q.record_video("alice")
    chk = q.check("bob", limit_videos=5)
    assert chk.allowed is True   # bob hasn't spent anything


# ─────────────────────────────────────────────────────────────────────
# Audit log
# ─────────────────────────────────────────────────────────────────────

def test_audit_appends_one_line_per_event(tmp_path):
    a = AuditLog(tmp_path / "audit.jsonl")
    a.record(event="signup", actor="u1", metadata={"plan": "free"})
    a.record(event="login", actor="u1")
    a.record(event="login", actor="u2")
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    rows = [json.loads(l) for l in lines]
    assert rows[0]["event"] == "signup"


def test_audit_tail_filters_by_actor_and_event(tmp_path):
    a = AuditLog(tmp_path / "audit.jsonl")
    a.record(event="signup", actor="u1")
    a.record(event="login", actor="u1")
    a.record(event="login", actor="u2")
    a.record(event="signup", actor="u3")
    out_actor = a.tail(actor="u1")
    assert {r["event"] for r in out_actor} == {"signup", "login"}
    out_event = a.tail(event="signup")
    assert {r["actor"] for r in out_event} == {"u1", "u3"}


def test_audit_tail_returns_most_recent_first(tmp_path):
    a = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        a.record(event=f"e{i}", actor="x")
    out = a.tail(n=3)
    assert [r["event"] for r in out] == ["e4", "e3", "e2"]


# ─────────────────────────────────────────────────────────────────────
# License
# ─────────────────────────────────────────────────────────────────────

def test_license_issue_and_verify_roundtrip():
    key = issue_license("pro", "user1234", expires_at=int(time.time()) + 86400,
                         secret="test-secret")
    lic = verify_license(key, secret="test-secret")
    assert lic.valid is True
    assert lic.plan == "pro"


def test_license_lifetime_never_expires():
    key = issue_license("team", "user1234", expires_at=0, secret="s")
    lic = verify_license(key, secret="s")
    assert lic.valid is True
    assert lic.expires_at == 0


def test_license_rejects_expired():
    key = issue_license("pro", "user1234",
                         expires_at=int(time.time()) - 10, secret="s")
    lic = verify_license(key, secret="s")
    assert lic.valid is False
    assert lic.reason == "expired"


def test_license_rejects_wrong_secret():
    key = issue_license("pro", "user1234", expires_at=int(time.time()) + 86400,
                         secret="real-secret")
    lic = verify_license(key, secret="fake-secret")
    assert lic.valid is False
    assert "signature" in lic.reason


def test_license_rejects_malformed():
    assert verify_license("garbage", secret="s").valid is False
    assert verify_license("SF-pro", secret="s").valid is False
    assert verify_license("", secret="s").valid is False


def test_license_rejects_invalid_plan():
    with pytest.raises(ValueError):
        issue_license("godmode", "user", 0)


# ─────────────────────────────────────────────────────────────────────
# Stripe webhook signature
# ─────────────────────────────────────────────────────────────────────

def _make_stripe_signed(payload: bytes, secret: str, ts: int | None = None) -> str:
    ts = ts or int(time.time())
    signed = f"{ts}.{payload.decode()}"
    sig = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def test_stripe_webhook_verifies_valid_signature():
    secret = "whsec_test123"
    payload = json.dumps({
        "type": "checkout.session.completed",
        "data": {"object": {"client_reference_id": "user_abc",
                            "metadata": {"plan": "pro"},
                            "customer": "cus_xyz"}},
    }).encode()
    sig = _make_stripe_signed(payload, secret)
    event = verify_webhook(payload, sig, secret)
    assert event is not None
    assert event["type"] == "checkout.session.completed"


def test_stripe_webhook_rejects_bad_signature():
    secret = "whsec_test123"
    payload = b'{"type":"checkout.session.completed"}'
    sig = _make_stripe_signed(payload, "wrong-secret")
    assert verify_webhook(payload, sig, secret) is None


def test_stripe_webhook_rejects_old_timestamp():
    secret = "whsec_test123"
    payload = b'{"type":"x"}'
    old_ts = int(time.time()) - 600  # 10 min old, default tolerance is 5min
    sig = _make_stripe_signed(payload, secret, ts=old_ts)
    assert verify_webhook(payload, sig, secret) is None


def test_stripe_handle_checkout_completed():
    ev = {
        "type": "checkout.session.completed",
        "data": {"object": {"client_reference_id": "user42",
                            "metadata": {"plan": "team"},
                            "customer": "cus_42"}},
    }
    action = handle_event(ev)
    assert action == {"action": "upgrade", "user_id": "user42",
                       "plan": "team", "stripe_customer_id": "cus_42"}


def test_stripe_handle_subscription_deleted():
    ev = {
        "type": "customer.subscription.deleted",
        "data": {"object": {"metadata": {"user_id": "user42"}}},
    }
    action = handle_event(ev)
    assert action == {"action": "downgrade", "user_id": "user42", "plan": "free"}


def test_stripe_handle_unknown_event_returns_none():
    ev = {"type": "charge.refunded", "data": {"object": {}}}
    assert handle_event(ev) is None


# ─────────────────────────────────────────────────────────────────────
# FastAPI integration — auth endpoints
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCFORGE_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("SYNCFORGE_JWT_SECRET", "test-jwt")
    # Force reimport so module-level globals re-read env
    import importlib
    import app.main
    importlib.reload(app.main)
    from app.main import app, JOBS
    JOBS.clear()
    with TestClient(app) as c:
        yield c


def test_register_login_me_flow(client):
    r = client.post("/api/auth/register",
                    json={"email": "test@test.com", "password": "supersecret123"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "test@test.com"
    assert me.json()["limits"]["videos_month"] == PLANS["free"]["videos_month"]


def test_register_short_password_rejected(client):
    r = client.post("/api/auth/register",
                    json={"email": "x@x.com", "password": "short"})
    assert r.status_code == 400


def test_login_wrong_password(client):
    client.post("/api/auth/register",
                json={"email": "u@u.com", "password": "password123"})
    r = client.post("/api/auth/login",
                    json={"email": "u@u.com", "password": "wrongone1"})
    assert r.status_code == 401


def test_me_without_token_fails(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_create_job_anonymous_works_in_dev(client, monkeypatch):
    """In dev (no SYNCFORGE_REQUIRE_AUTH), anon users can create jobs."""
    # Patch _run_job so we don't actually launch the pipeline
    async def fake_run(jid, req):
        pass
    monkeypatch.setattr("app.main._run_job", fake_run)
    r = client.post("/api/jobs",
                    json={"title": "anon job test", "voice": "en-US-AndrewNeural"})
    assert r.status_code == 200
    assert r.json()["status"] == "queued"


def test_create_job_quota_enforced(client, monkeypatch):
    async def fake_run(jid, req):
        pass
    monkeypatch.setattr("app.main._run_job", fake_run)
    # Free plan = 5 videos/month. Fire 6.
    headers = {"Content-Type": "application/json"}
    statuses = []
    for i in range(6):
        r = client.post("/api/jobs", json={"title": f"job-{i}"}, headers=headers)
        statuses.append(r.status_code)
    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429  # quota exhausted


def test_license_verify_endpoint(client, monkeypatch):
    monkeypatch.setenv("LICENSE_SECRET", "test-license-secret-public")
    key = issue_license("pro", "user1234", int(time.time()) + 86400,
                        secret="test-license-secret-public")
    r = client.post("/api/license/verify", params={"key": key})
    assert r.status_code == 200
    assert r.json()["valid"] is True
    assert r.json()["plan"] == "pro"
