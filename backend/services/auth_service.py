import hashlib
import hmac
import json
import base64
import time

from sqlalchemy.orm import Session

from backend.models.user import User

SECRET_KEY = "cyberdrill-secret-key-change-in-production"


def _create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": int(time.time()) + 86400,  # 24 hours
    }
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(
        SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        expected_sig = hmac.new(
            SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload = json.loads(base64.b64decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def register_user(db: Session, username: str, email: str, password: str) -> User:
    existing = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()
    if existing:
        if existing.username == username:
            raise ValueError("Username already exists")
        raise ValueError("Email already exists")

    user = User(username=username, email=email)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login_user(db: Session, username: str, password: str) -> tuple[User, str]:
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.check_password(password):
        raise ValueError("Invalid username or password")
    token = _create_token(user.id, user.username)
    return user, token


def get_current_user(db: Session, token: str) -> User | None:
    payload = verify_token(token)
    if not payload:
        return None
    return db.query(User).filter(User.id == payload["user_id"]).first()
