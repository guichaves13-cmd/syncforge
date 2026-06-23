"""SyncForge FastAPI app — REST + WebSocket for real-time job progress."""
from __future__ import annotations
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth.users import UserStore, issue_token, verify_token, PLANS
from .billing.stripe_handler import (
    create_checkout_session, handle_event, verify_webhook,
)
from .core.audit import AuditLog
from .core.license import issue_license, verify_license
from .core.quota import QuotaStore


app = FastAPI(title="SyncForge", version="0.4.0")

JWT_SECRET = os.getenv("SYNCFORGE_JWT_SECRET", "dev-jwt-secret-change-in-prod")
STORAGE = Path(os.getenv("SYNCFORGE_STORAGE", "storage"))
USERS = UserStore(STORAGE / "users.json")
QUOTAS = QuotaStore(STORAGE / "quota.json")
AUDIT = AuditLog(STORAGE / "audit.jsonl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory job store (single-process). Swap for Redis when scaling.
# ─────────────────────────────────────────────────────────────────────────────

class Job(BaseModel):
    id: str
    status: str = "queued"      # queued|running|done|failed
    progress: float = 0.0
    events: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None


JOBS: dict[str, Job] = {}
SUBS: dict[str, list[asyncio.Queue]] = {}  # job_id -> queues for websocket subscribers


async def publish(job_id: str, event: dict[str, Any]) -> None:
    if (job := JOBS.get(job_id)) is not None:
        job.events.append(event)
    for q in SUBS.get(job_id, []):
        await q.put(event)


# ─────────────────────────────────────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────────────────────────────────────

class CreateJobReq(BaseModel):
    title: str
    theme: str = ""
    language: str = "en"
    voice: str = "en-US-AndrewNeural"
    target_sec: int = 600
    mode: str = "tts_only"      # tts_only | avatar_overlay | avatar_full
    enable_vision_verify: bool = True
    enable_embeddings: bool = True
    enable_generative_fallback: bool = False


@app.get("/api/health")
def health() -> dict[str, Any]:
    from .updater.check import CURRENT_VERSION
    return {"ok": True, "service": "syncforge", "jobs": len(JOBS),
            "version": CURRENT_VERSION}


@app.get("/api/updates/check")
def updates_check(channel: str = "stable") -> dict[str, Any]:
    from .updater.check import check_for_update
    return check_for_update(channel=channel).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Auth — register / login / me
# ─────────────────────────────────────────────────────────────────────────────

class AuthReq(BaseModel):
    email: str
    password: str


def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    payload = verify_token(authorization[7:], JWT_SECRET)
    if not payload:
        raise HTTPException(401, "invalid or expired token")
    return payload


@app.post("/api/auth/register")
def auth_register(req: AuthReq, request: Request) -> dict[str, Any]:
    if len(req.password) < 8:
        raise HTTPException(400, "password must be ≥8 chars")
    try:
        u = USERS.create(req.email, req.password)
    except ValueError as e:
        raise HTTPException(409, str(e))
    AUDIT.record(event="user.register", actor=u.id,
                 metadata={"email": req.email},
                 ip=request.client.host if request.client else None)
    return {"token": issue_token(u, JWT_SECRET), "user": u.public()}


@app.post("/api/auth/login")
def auth_login(req: AuthReq, request: Request) -> dict[str, Any]:
    u = USERS.verify(req.email, req.password)
    if not u:
        AUDIT.record(event="user.login_failed", actor="unknown",
                     metadata={"email": req.email},
                     ip=request.client.host if request.client else None)
        raise HTTPException(401, "invalid credentials")
    AUDIT.record(event="user.login", actor=u.id, ip=request.client.host if request.client else None)
    return {"token": issue_token(u, JWT_SECRET), "user": u.public()}


@app.get("/api/auth/me")
def auth_me(user: dict = Depends(current_user)) -> dict[str, Any]:
    u = USERS.get(user["sub"])
    if not u:
        raise HTTPException(404, "user no longer exists")
    return {"user": u.public(),
            "quota": QUOTAS.stats(u.id),
            "limits": PLANS.get(u.plan, PLANS["free"])}


# ─────────────────────────────────────────────────────────────────────────────
# Billing
# ─────────────────────────────────────────────────────────────────────────────

class CheckoutReq(BaseModel):
    plan: str           # "pro" | "team"
    success_url: str
    cancel_url: str


@app.post("/api/billing/checkout")
def billing_checkout(req: CheckoutReq, user: dict = Depends(current_user)) -> dict[str, Any]:
    u = USERS.get(user["sub"])
    if not u:
        raise HTTPException(404, "user not found")
    if req.plan not in ("pro", "team"):
        raise HTTPException(400, "plan must be 'pro' or 'team'")
    sess = create_checkout_session(
        user_id=u.id, user_email=u.email, plan=req.plan,
        success_url=req.success_url, cancel_url=req.cancel_url,
    )
    if not sess:
        raise HTTPException(503, "Stripe not configured or unavailable")
    AUDIT.record(event="billing.checkout_created", actor=u.id,
                 metadata={"plan": req.plan, "session_id": sess.session_id})
    return {"url": sess.url, "session_id": sess.session_id}


@app.post("/api/billing/webhook")
async def billing_webhook(request: Request,
                          stripe_signature: Optional[str] = Header(default=None)):
    body = await request.body()
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    event = verify_webhook(body, stripe_signature or "", secret)
    if event is None:
        raise HTTPException(400, "invalid signature")
    action = handle_event(event)
    if not action:
        return {"ok": True, "noop": True}
    user_id = action.get("user_id")
    if not user_id:
        raise HTTPException(400, "no user_id in event metadata")
    plan = action["plan"]
    USERS.upgrade(user_id, plan,
                  stripe_customer_id=action.get("stripe_customer_id", ""))
    AUDIT.record(event=f"billing.{action['action']}", actor=user_id,
                 metadata={"plan": plan})
    return {"ok": True, "applied": action}


# ─────────────────────────────────────────────────────────────────────────────
# License (offline-capable)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/license/issue")
def license_issue(plan: str, user_id: str, days: int = 365,
                  user: dict = Depends(current_user)) -> dict[str, Any]:
    """Admin-ish: issue an offline license key. Only for self for now."""
    if user["sub"] != user_id:
        raise HTTPException(403, "can only issue for yourself")
    import time as _t
    expires_at = int(_t.time()) + days * 86400
    key = issue_license(plan, user_id, expires_at)
    AUDIT.record(event="license.issued", actor=user_id,
                 metadata={"plan": plan, "expires_at": expires_at})
    return {"key": key, "expires_at": expires_at}


@app.post("/api/license/verify")
def license_verify(key: str) -> dict[str, Any]:
    lic = verify_license(key)
    return {
        "valid": lic.valid, "plan": lic.plan, "user_id": lic.user_id,
        "expires_at": lic.expires_at, "reason": lic.reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Audit (admin-only — gated by env ADMIN_USER_ID for now)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/audit/tail")
def audit_tail(n: int = 100, event: Optional[str] = None,
               actor: Optional[str] = None,
               user: dict = Depends(current_user)) -> dict[str, Any]:
    if user["sub"] != os.getenv("ADMIN_USER_ID", ""):
        # non-admin: can only see own events
        actor = user["sub"]
    return {"events": AUDIT.tail(n=n, event=event, actor=actor)}


@app.post("/api/jobs")
async def create_job(req: CreateJobReq,
                     authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    # Auth is OPTIONAL in dev (no token → anon, default-quota); REQUIRED in prod
    # when SYNCFORGE_REQUIRE_AUTH=1 is set.
    user_id = "anon"
    user_plan = "free"
    if authorization and authorization.lower().startswith("bearer "):
        payload = verify_token(authorization[7:], JWT_SECRET)
        if payload:
            user_id = payload["sub"]
            user_plan = payload.get("plan", "free")
    elif os.getenv("SYNCFORGE_REQUIRE_AUTH") == "1":
        raise HTTPException(401, "auth required")

    # Quota check
    limit = PLANS.get(user_plan, PLANS["free"])["videos_month"]
    chk = QUOTAS.check(user_id, limit)
    if not chk.allowed:
        AUDIT.record(event="job.quota_exceeded", actor=user_id,
                     metadata={"used": chk.used_videos, "limit": chk.limit_videos})
        raise HTTPException(429, chk.reason)

    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = Job(id=jid)
    SUBS.setdefault(jid, [])
    QUOTAS.record_video(user_id)
    AUDIT.record(event="job.created", actor=user_id,
                 metadata={"job_id": jid, "title": req.title[:120]})
    asyncio.create_task(_run_job(jid, req))
    return {"job_id": jid, "status": "queued",
            "quota": QUOTAS.stats(user_id), "limit": limit}


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict[str, Any]:
    if (j := JOBS.get(jid)) is None:
        return {"error": "not found"}
    return j.model_dump()


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [{"id": j.id, "status": j.status, "progress": j.progress}
                     for j in JOBS.values()]}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — real-time progress per job
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/jobs/{jid}")
async def ws_job(ws: WebSocket, jid: str) -> None:
    await ws.accept()
    if jid not in JOBS:
        await ws.send_json({"error": "job not found"})
        await ws.close()
        return
    q: asyncio.Queue = asyncio.Queue()
    SUBS.setdefault(jid, []).append(q)
    # Replay past events for late subscribers
    for ev in JOBS[jid].events:
        await ws.send_json(ev)
    try:
        while True:
            ev = await q.get()
            await ws.send_json(ev)
            if ev.get("event") in ("done", "failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        SUBS[jid] = [x for x in SUBS.get(jid, []) if x is not q]


# ─────────────────────────────────────────────────────────────────────────────
# Background job runner — calls the real SyncEngine via runner.py
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(os.getenv("SYNCFORGE_OUTPUT_DIR", "storage/videos"))


async def _run_job(jid: str, req: CreateJobReq) -> None:
    from .services.runner import run_pipeline
    job = JOBS[jid]
    job.status = "running"
    await publish(jid, {"event": "start", "title": req.title})

    loop = asyncio.get_running_loop()

    def _progress(ev: dict) -> None:
        # Marshall sync callback → async publish via run_coroutine_threadsafe
        ev.setdefault("event", "step")
        asyncio.run_coroutine_threadsafe(publish(jid, ev), loop)

    # For now: script = title. Future: call a script-writer LLM if mode demands.
    script = req.title.strip()

    def _do_work() -> dict:
        return run_pipeline(
            script_text=script,
            voice=req.voice,
            title=req.title,
            theme=req.theme,
            output_dir=OUTPUT_DIR,
            enable_vision=req.enable_vision_verify,
            enable_embeddings=req.enable_embeddings,
            enable_generative_fallback=req.enable_generative_fallback,
            progress=_progress,
        )

    try:
        result = await loop.run_in_executor(None, _do_work)
        job.result = result
        if result.get("ok"):
            job.status = "done"
            await publish(jid, {"event": "done", "result": result})
        else:
            job.status = "failed"
            await publish(jid, {"event": "failed",
                                "error": result.get("error", "unknown")})
    except Exception as e:
        job.status = "failed"
        await publish(jid, {"event": "failed", "error": str(e)})
