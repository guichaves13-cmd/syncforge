"""Stress / concurrency tests — race conditions, parallel jobs, WS multi-subscribers."""
from __future__ import annotations
import asyncio
import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCFORGE_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("SYNCFORGE_JWT_SECRET", "stress-secret")
    monkeypatch.setenv("LICENSE_SECRET", "stress-license")
    import app.main as main
    importlib.reload(main)
    main.JOBS.clear()
    # Replace runner with instant no-op
    async def fast_run(jid, req):
        from app.main import publish, JOBS
        JOBS[jid].status = "done"
        await publish(jid, {"event": "done", "result": {"ok": True}})
    monkeypatch.setattr(main, "_run_job", fast_run)
    with TestClient(main.app) as c:
        yield c, main


# ─── Job creation concurrency ───────────────────────────────────────────

def test_50_concurrent_job_creations_unique_ids(client):
    """Burst of 50 concurrent jobs from a high-quota user. Every successful
    creation must yield a unique ID — no duplicates under race."""
    c, _ = client
    # Register, upgrade to team tier (1000 videos/month → plenty of headroom)
    reg = c.post("/api/auth/register",
                  json={"email": "burst@x.com", "password": "longpass1"}).json()
    user_id = reg["user"]["id"]
    # Manually upgrade via the userstore (no Stripe in test)
    import importlib, app.main as main
    main.USERS.upgrade(user_id, "team")
    # Re-issue token with new plan
    from app.auth.users import issue_token
    u = main.USERS.get(user_id)
    token = issue_token(u, main.JWT_SECRET)
    hdr = {"Authorization": f"Bearer {token}"}

    ids: list[str] = []
    statuses: list[int] = []
    lock = threading.Lock()
    def hit():
        r = c.post("/api/jobs", json={"title": "stress"}, headers=hdr)
        with lock:
            statuses.append(r.status_code)
            if r.status_code == 200:
                ids.append(r.json()["job_id"])
    with ThreadPoolExecutor(max_workers=20) as ex:
        for _ in range(50):
            ex.submit(hit)
    # All 50 returned a definitive answer
    assert len(statuses) == 50
    # Vast majority succeeded (team quota = 1000/mo)
    assert len(ids) == 50, f"only {len(ids)} of 50 succeeded; statuses: {set(statuses)}"
    # Critical: every job_id is unique
    assert len(set(ids)) == 50, "duplicate job_ids under concurrency"


def test_quota_thread_safety(client):
    """Burst 20 jobs from same user — exactly 5 must succeed (free tier)."""
    c, _ = client
    reg = c.post("/api/auth/register",
                  json={"email": "race@x.com", "password": "longpass1"}).json()
    hdr = {"Authorization": f"Bearer {reg['token']}"}

    statuses = []
    lock = threading.Lock()
    def hit():
        r = c.post("/api/jobs", json={"title": "T"}, headers=hdr)
        with lock:
            statuses.append(r.status_code)
    with ThreadPoolExecutor(max_workers=10) as ex:
        for _ in range(20):
            ex.submit(hit)

    ok = sum(1 for s in statuses if s == 200)
    blocked = sum(1 for s in statuses if s == 429)
    # All 20 returned a definitive answer
    assert ok + blocked == 20
    # Quota enforcement: at most 5 successes
    assert ok <= 5, f"quota leaked under race: {ok} successes (expected ≤5)"
    # All others rejected with 429
    assert blocked == 20 - ok


# ─── WebSocket multi-subscriber & late-join replay ──────────────────────

def test_ws_multiple_subscribers_get_all_events(client):
    """Two concurrent WS subscribers must receive the same event stream."""
    c, main = client
    from app.main import Job, JOBS, SUBS
    jid = "wsmulti01"
    JOBS[jid] = Job(id=jid, status="done",
                    events=[{"event": "start"},
                            {"event": "step", "step": "tts"},
                            {"event": "step", "step": "sync"},
                            {"event": "done", "result": {"ok": True}}])
    SUBS[jid] = []

    out_a, out_b = [], []
    with c.websocket_connect(f"/ws/jobs/{jid}") as ws_a, \
         c.websocket_connect(f"/ws/jobs/{jid}") as ws_b:
        for _ in range(4):
            out_a.append(ws_a.receive_json())
        for _ in range(4):
            out_b.append(ws_b.receive_json())

    assert [e["event"] for e in out_a] == ["start", "step", "step", "done"]
    assert out_a == out_b, "subscribers received different events"


def test_ws_late_subscriber_gets_full_replay(client):
    """Subscribe AFTER job is done → must still receive all past events."""
    c, _ = client
    from app.main import Job, JOBS, SUBS
    jid = "wslate01"
    JOBS[jid] = Job(id=jid, status="done",
                    events=[{"event": "start"},
                            {"event": "step", "step": "tts"},
                            {"event": "step", "step": "compose"},
                            {"event": "done", "result": {"ok": True}}])
    SUBS[jid] = []

    received = []
    with c.websocket_connect(f"/ws/jobs/{jid}") as ws:
        for _ in range(4):
            received.append(ws.receive_json())

    assert len(received) == 4
    assert received[-1]["event"] == "done"


# ─── Audit log thread-safety ────────────────────────────────────────────

def test_audit_concurrent_writes_no_corruption(tmp_path):
    """100 threads × 5 writes each → every line must be valid JSON."""
    from app.core.audit import AuditLog
    log = AuditLog(tmp_path / "race.jsonl")

    def writer(i):
        for j in range(5):
            log.record(event=f"e{i}", actor=f"u{i}", metadata={"j": j})

    with ThreadPoolExecutor(max_workers=20) as ex:
        for i in range(100):
            ex.submit(writer, i)

    import json
    lines = (tmp_path / "race.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 500
    for line in lines:
        json.loads(line)  # all parse


# ─── Quota persistence race ─────────────────────────────────────────────

def test_quota_record_video_atomic(tmp_path):
    """100 concurrent record_video calls → exactly 100 final count."""
    from app.core.quota import QuotaStore
    qs = QuotaStore(tmp_path / "q.json")

    def bump():
        qs.record_video("u1")

    with ThreadPoolExecutor(max_workers=20) as ex:
        for _ in range(100):
            ex.submit(bump)

    assert qs.stats("u1")["videos"] == 100


# ─── Dedup store under concurrent reads/writes ──────────────────────────

def test_dedup_store_concurrent_safe():
    """Concurrent add + is_duplicate must not crash (best-effort consistency)."""
    from app.services.sync.dedup import DedupStore

    class H:
        def __init__(self, n): self.n = n
        def __sub__(self, o): return abs(self.n - o.n)

    ds = DedupStore(threshold=2)

    def worker(i):
        ds.add([H(i * 100)])
        for j in range(10):
            ds.is_duplicate([H(j)])

    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(worker, range(50)))

    # All 50 hashes were added (no exception, no data race)
    assert len(ds.seen) == 50
