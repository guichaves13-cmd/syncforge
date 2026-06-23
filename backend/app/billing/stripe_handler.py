"""Stripe Checkout + Webhook handlers.

Webhook verifies signature with STRIPE_WEBHOOK_SECRET, then upgrades the
matching user via UserStore.upgrade().
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class CheckoutSession:
    url: str
    session_id: str


PRICE_MAP = {
    # Map your plan → Stripe Price ID via env, e.g. STRIPE_PRICE_PRO=price_abc123
    "pro":  os.getenv("STRIPE_PRICE_PRO",  "price_pro_placeholder"),
    "team": os.getenv("STRIPE_PRICE_TEAM", "price_team_placeholder"),
}


def create_checkout_session(*, user_id: str, user_email: str, plan: str,
                            success_url: str, cancel_url: str,
                            stripe_key: str = "") -> Optional[CheckoutSession]:
    """Returns a Stripe Checkout URL the client should redirect to."""
    key = stripe_key or os.getenv("STRIPE_SECRET_KEY", "")
    if not key:
        return None
    price_id = PRICE_MAP.get(plan)
    if not price_id:
        return None
    try:
        r = requests.post(
            "https://api.stripe.com/v1/checkout/sessions",
            auth=(key, ""),
            data={
                "mode": "subscription",
                "line_items[0][price]": price_id,
                "line_items[0][quantity]": 1,
                "customer_email": user_email,
                "client_reference_id": user_id,
                "metadata[user_id]": user_id,
                "metadata[plan]": plan,
                "success_url": success_url,
                "cancel_url": cancel_url,
            },
            timeout=15,
        )
        r.raise_for_status()
        j = r.json()
        return CheckoutSession(url=j["url"], session_id=j["id"])
    except Exception:
        return None


# ─── webhook signature verification ───────────────────────────────────

def verify_webhook(payload: bytes, sig_header: str, secret: str,
                   tolerance_s: int = 300) -> Optional[dict]:
    """Stripe sends `Stripe-Signature: t=...,v1=...,v1=...`. Verify HMAC-SHA256."""
    if not sig_header or not secret:
        return None
    parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
    timestamp = parts.get("t")
    signature = parts.get("v1")
    if not timestamp or not signature:
        return None
    if abs(time.time() - float(timestamp)) > tolerance_s:
        return None
    signed_payload = f"{timestamp}.{payload.decode('utf-8', errors='replace')}"
    expected = hmac.new(secret.encode(), signed_payload.encode(),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def handle_event(event: dict) -> dict | None:
    """Translate a verified Stripe event into a UserStore action.

    Returns: {"action": "upgrade", "user_id": ..., "plan": ..., "stripe_customer_id": ...}
             or None if event isn't actionable.
    """
    t = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}
    if t == "checkout.session.completed":
        return {
            "action": "upgrade",
            "user_id": obj.get("client_reference_id") or
                        (obj.get("metadata") or {}).get("user_id"),
            "plan": (obj.get("metadata") or {}).get("plan", "pro"),
            "stripe_customer_id": obj.get("customer", ""),
        }
    if t == "customer.subscription.deleted":
        return {
            "action": "downgrade",
            "user_id": (obj.get("metadata") or {}).get("user_id"),
            "plan": "free",
        }
    return None
