"""Test the FastAPI app (REST + WebSocket) without running real jobs."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import app, JOBS


@pytest.fixture()
def client():
    JOBS.clear()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["service"] == "syncforge"


def test_create_job_returns_id(client, monkeypatch):
    # Patch the runner to a no-op coroutine so we don't actually run the pipeline
    async def fake_run(jid, req):
        from app.main import publish, JOBS
        JOBS[jid].status = "done"
        await publish(jid, {"event": "done", "result": {"ok": True}})

    monkeypatch.setattr("app.main._run_job", fake_run)
    r = client.post("/api/jobs", json={
        "title": "Test Video",
        "theme": "test",
        "voice": "en-US-AndrewNeural",
    })
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_get_job_404(client):
    r = client.get("/api/jobs/nonexistent")
    assert r.status_code == 200
    assert "error" in r.json()


def test_list_jobs(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert "jobs" in r.json()


def test_websocket_replays_events(client, monkeypatch):
    # Pre-create a job with events
    from app.main import Job, JOBS, SUBS
    jid = "wsjob123"
    JOBS[jid] = Job(id=jid, status="done",
                    events=[{"event": "start"},
                            {"event": "step", "step": "tts"},
                            {"event": "done", "result": {"ok": True}}])
    SUBS[jid] = []

    with client.websocket_connect(f"/ws/jobs/{jid}") as ws:
        # Should receive all 3 past events, then close on 'done'
        e1 = ws.receive_json()
        e2 = ws.receive_json()
        e3 = ws.receive_json()
        assert e1["event"] == "start"
        assert e2["step"] == "tts"
        assert e3["event"] == "done"
