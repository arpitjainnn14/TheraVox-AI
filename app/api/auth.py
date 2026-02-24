"""
Authentication endpoints.

POST /api/auth/register  — create account
POST /api/auth/login     — obtain tokens
POST /api/auth/refresh   — rotate refresh token, get new access token
POST /api/auth/logout    — revoke refresh token
GET  /api/auth/me        — get current user profile (requires Bearer)
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.auth.utils import (
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
    verify_password,
    hash_password,
)
from app.core.config import get_settings
from app.db.models import RefreshToken, User
from app.models.schemas import (
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserRegisterRequest,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Cookie configuration
# ---------------------------------------------------------------------------

_COOKIE_NAME = "theravox_refresh"
_COOKIE_PATH = "/"          # Must be "/" so the browser sends it to all /api/auth/* paths
_COOKIE_SAMESITE = "lax"


def _set_refresh_cookie(response: Response, raw_token: str, expires_at: datetime) -> None:
    """Set the httpOnly refresh token cookie on the response."""
    max_age = int((expires_at - datetime.now(timezone.utc)).total_seconds())
    response.set_cookie(
        key=_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=False,          # Set True in production behind HTTPS
        samesite=_COOKIE_SAMESITE,
        path=_COOKIE_PATH,
        max_age=max_age,
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Clear the refresh token cookie."""
    response.delete_cookie(
        key=_COOKIE_NAME,
        path=_COOKIE_PATH,
        samesite=_COOKIE_SAMESITE,
    )


async def _issue_refresh_token(
    user: User,
    db: AsyncSession,
    response: Response,
) -> None:
    """Create a new refresh token, persist it, and set the cookie."""
    settings = get_settings()
    expire_days: int = settings.get("jwt_refresh_token_expire_days", 7)
    expires_at = datetime.now(timezone.utc) + timedelta(days=expire_days)

    raw_token, token_hash = create_refresh_token()

    db_token = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    db.add(db_token)
    # Caller is responsible for commit (managed by get_db dependency)

    _set_refresh_cookie(response, raw_token, expires_at)


# ---------------------------------------------------------------------------
# POST /api/auth/register
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    body: UserRegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    # Check for existing email — return a generic error to prevent user enumeration
    existing = await db.execute(select(User).where(User.email == body.email.lower()))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists",
        )

    user = User(
        email=body.email.lower(),
        full_name=body.full_name.strip(),
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()  # Get user.id without committing yet

    await _issue_refresh_token(user, db, response)
    # get_db commits after the route returns

    access_token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=access_token,
        user=UserProfileResponse.model_validate(user),
    )


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and obtain tokens",
)
async def login(
    body: UserLoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user: User | None = result.scalar_one_or_none()

    # Use constant-time comparison by always calling verify_password
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Revoke all existing refresh tokens for this user (single-session policy)
    existing_tokens = await db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == user.id,
            RefreshToken.revoked.is_(False),
        )
    )
    for rt in existing_tokens.scalars():
        rt.revoked = True

    await _issue_refresh_token(user, db, response)

    access_token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=access_token,
        user=UserProfileResponse.model_validate(user),
    )


# ---------------------------------------------------------------------------
# POST /api/auth/refresh
# ---------------------------------------------------------------------------

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate refresh token and issue new access token",
)
async def refresh(
    response: Response,
    db: AsyncSession = Depends(get_db),
    theravox_refresh: str | None = Cookie(default=None),
) -> TokenResponse:
    if not theravox_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    token_hash = hash_refresh_token(theravox_refresh)

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    rt: RefreshToken | None = result.scalar_one_or_none()

    if rt is None or not rt.is_valid:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired",
        )

    user: User | None = await db.get(User, rt.user_id)
    if user is None or not user.is_active:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or account disabled",
        )

    # Rotate: revoke old token, issue new one
    rt.revoked = True
    await _issue_refresh_token(user, db, response)

    access_token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=access_token,
        user=UserProfileResponse.model_validate(user),
    )


# ---------------------------------------------------------------------------
# POST /api/auth/logout
# ---------------------------------------------------------------------------

@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke refresh token and clear cookie",
)
async def logout(
    response: Response,
    db: AsyncSession = Depends(get_db),
    theravox_refresh: str | None = Cookie(default=None),
) -> None:
    if theravox_refresh:
        token_hash = hash_refresh_token(theravox_refresh)
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        rt: RefreshToken | None = result.scalar_one_or_none()
        if rt and not rt.revoked:
            rt.revoked = True

    _clear_refresh_cookie(response)


# ---------------------------------------------------------------------------
# GET /api/auth/me
# ---------------------------------------------------------------------------

@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get the current authenticated user's profile",
)
async def me(current_user: User = Depends(get_current_user)) -> UserProfileResponse:
    return UserProfileResponse.model_validate(current_user)
