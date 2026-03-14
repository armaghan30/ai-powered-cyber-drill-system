from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional

from backend.dependencies import get_db
from backend.schemas.auth import RegisterRequest, LoginRequest, AuthResponse
from backend.services import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    try:
        user = auth_service.register_user(
            db, payload.username, payload.email, payload.password
        )
        _, token = auth_service.login_user(db, payload.username, payload.password)
        return AuthResponse(
            token=token, username=user.username, message="Registration successful"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Login with username and password."""
    try:
        user, token = auth_service.login_user(db, payload.username, payload.password)
        return AuthResponse(
            token=token, username=user.username, message="Login successful"
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.get("/me")
def get_me(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(default=None),
):
    """Get current user info from token."""
    token = authorization.replace("Bearer ", "") if authorization else ""
    user = auth_service.get_current_user(db, token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"id": user.id, "username": user.username, "email": user.email}
