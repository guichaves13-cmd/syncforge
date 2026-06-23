"""Lightweight user store — file-backed JSON. Swap for Postgres when scaling.

Password hashing: PBKDF2-HMAC-SHA256 with 200k iterations + per-user salt.
Token: JWT-like (HS256) signed string {sub, exp, plan, tier}.
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock


_PBKDF2_ITERS = 200_000
_USERS_LOCK = Lock()


@dataclass
class User:
    id: str
    email: str
    password_salt: str           # hex
    password_hash: str           # hex
    plan: str = "free"           # free|pro|team
    quota_videos_month: int = 5  # default for free
    quota_storage_gb: int = 1
    stripe_customer_id: str = ""
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("password_salt"); d.pop("password_hash")
        return d


PLANS = {
    "free": {"videos_month": 5, "storage_gb": 1},
    "pro":  {"videos_month": 100, "storage_gb": 50},
    "team": {"videos_month": 1000, "storage_gb": 500},
}


class UserStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    # ── CRUD ──────────────────────────────────────────────────────────

    def get_by_email(self, email: str) -> User | None:
        for u in self._load().values():
            if u.email.lower() == email.lower():
                return u
        return None

    def get(self, user_id: str) -> User | None:
        return self._load().get(user_id)

    def create(self, email: str, password: str, plan: str = "free") -> User:
        if self.get_by_email(email):
            raise ValueError(f"email already registered: {email}")
        salt = secrets.token_hex(16)
        u = User(
            id=uuid.uuid4().hex[:12],
            email=email.strip(),
            password_salt=salt,
            password_hash=_hash_password(password, salt),
            plan=plan,
            quota_videos_month=PLANS[plan]["videos_month"],
            quota_storage_gb=PLANS[plan]["storage_gb"],
        )
        with _USERS_LOCK:
            data = self._load()
            data[u.id] = u
            self._save(data)
        return u

    def verify(self, email: str, password: str) -> User | None:
        u = self.get_by_email(email)
        if not u:
            return None
        if _hash_password(password, u.password_salt) == u.password_hash:
            return u
        return None

    def upgrade(self, user_id: str, plan: str,
                stripe_customer_id: str = "") -> User | None:
        with _USERS_LOCK:
            data = self._load()
            u = data.get(user_id)
            if not u:
                return None
            u.plan = plan
            u.quota_videos_month = PLANS[plan]["videos_month"]
            u.quota_storage_gb = PLANS[plan]["storage_gb"]
            if stripe_customer_id:
                u.stripe_customer_id = stripe_customer_id
            data[user_id] = u
            self._save(data)
            return u

    def all(self) -> list[User]:
        return list(self._load().values())

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> dict[str, User]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {uid: User(**d) for uid, d in raw.items()}

    def _save(self, data: dict[str, User]) -> None:
        out = {uid: asdict(u) for uid, u in data.items()}
        self.path.write_text(json.dumps(out, indent=2), encoding="utf-8")


# ─── password hashing ──────────────────────────────────────────────────

def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return h.hex()


# ─── JWT-like tokens (HS256, custom but minimal) ───────────────────────

def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def issue_token(user: User, secret: str, ttl_seconds: int = 7 * 24 * 3600) -> str:
    """Returns header.payload.signature"""
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({
        "sub": user.id, "email": user.email, "plan": user.plan,
        "iat": int(time.time()), "exp": int(time.time()) + ttl_seconds,
    }, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64(sig)}"


def verify_token(token: str, secret: str) -> dict | None:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        return None
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64d(sig_b64)):
        return None
    payload = json.loads(_b64d(payload_b64))
    if payload.get("exp", 0) < int(time.time()):
        return None
    return payload
