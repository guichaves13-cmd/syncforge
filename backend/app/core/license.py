"""License key generation + offline verification.

License format:  `SF-{plan}-{user_id_short}-{expiry_unix}-{sig}`
Signed with HS256 over `{plan}|{user_id}|{expiry}` using LICENSE_SECRET.

A license is valid if:
  1. Signature verifies
  2. Now < expiry (or expiry == 0 for lifetime)
  3. Plan ∈ {free, pro, team}
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass


_VALID_PLANS = {"free", "pro", "team"}


@dataclass
class License:
    plan: str
    user_id: str
    expires_at: int   # unix seconds; 0 = lifetime
    valid: bool
    reason: str = ""


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _sign(payload: str, secret: str) -> str:
    return _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())[:16]


def issue_license(plan: str, user_id: str,
                  expires_at: int, secret: str | None = None) -> str:
    secret = secret or os.getenv("LICENSE_SECRET", "syncforge-default-license-secret-change-me")
    if plan not in _VALID_PLANS:
        raise ValueError(f"invalid plan: {plan}")
    user_id_short = (user_id or "")[:8] or "anon0000"
    body = f"{plan}|{user_id_short}|{expires_at}"
    sig = _sign(body, secret)
    return f"SF-{plan}-{user_id_short}-{expires_at}-{sig}"


def verify_license(key: str, secret: str | None = None) -> License:
    secret = secret or os.getenv("LICENSE_SECRET", "syncforge-default-license-secret-change-me")
    # split with limit=4 to get exactly 5 parts; the last part (sig) can itself
    # contain "-" because urlsafe-b64 uses "-" and "_"
    parts = (key or "").split("-", 4)
    if len(parts) != 5 or parts[0] != "SF":
        return License(plan="", user_id="", expires_at=0, valid=False,
                        reason="malformed key")
    _, plan, uid_short, exp_str, sig = parts
    if plan not in _VALID_PLANS:
        return License(plan=plan, user_id=uid_short, expires_at=0, valid=False,
                        reason=f"unknown plan {plan}")
    try:
        expires_at = int(exp_str)
    except ValueError:
        return License(plan=plan, user_id=uid_short, expires_at=0, valid=False,
                        reason="bad expiry")
    expected_sig = _sign(f"{plan}|{uid_short}|{expires_at}", secret)
    if not hmac.compare_digest(expected_sig, sig):
        return License(plan=plan, user_id=uid_short, expires_at=expires_at,
                        valid=False, reason="signature mismatch")
    if expires_at != 0 and expires_at < int(time.time()):
        return License(plan=plan, user_id=uid_short, expires_at=expires_at,
                        valid=False, reason="expired")
    return License(plan=plan, user_id=uid_short, expires_at=expires_at, valid=True)
