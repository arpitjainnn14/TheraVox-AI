"""Pydantic models for request and response validation."""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, EmailStr, Field


class EmotionResponse(BaseModel):
    """Response model for emotion analysis."""
    emotion: str = Field(..., description="Detected emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    emoji: str = Field(..., description="Emoji representation")
    description: str = Field(..., description="Human-readable description")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    text_ready: bool
    audio_loaded: bool
    audio_libs: Any


class AudioStatusResponse(BaseModel):
    """Response model for audio analyzer status."""
    hf_libs_available: bool
    hf_model_id: str
    hf_loaded: bool
    device: Optional[str]
    soundfile_available: bool
    emotions_supported: list


class VisionAnalysisResponse(BaseModel):
    """Response model for vision analysis."""
    faces: list[EmotionResponse]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")


# ---------------------------------------------------------------------------
# Auth schemas
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    """Request body for POST /api/auth/register."""
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    """Request body for POST /api/auth/login."""
    email: EmailStr
    password: str


class UserProfileResponse(BaseModel):
    """Response shape for an authenticated user's profile."""
    id: uuid.UUID
    email: str
    full_name: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Response body for register / login / refresh endpoints."""
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse


# ---------------------------------------------------------------------------
# Wellness schemas
# ---------------------------------------------------------------------------

class WellnessEntryCreate(BaseModel):
    """Request body for POST /api/wellness/entries."""
    entry_type: str = Field(..., max_length=50, description="'journal' | 'mood_log' | 'gratitude' | 'activity'")
    content: str = Field(..., min_length=1)
    mood_score: Optional[float] = Field(default=None, ge=0, le=10)
    tags: Optional[List[str]] = None


class WellnessEntryResponse(BaseModel):
    """Response shape for a wellness entry."""
    id: uuid.UUID
    user_id: uuid.UUID
    entry_type: str
    content: str
    mood_score: Optional[float]
    tags: Optional[List[str]]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Feedback schemas
# ---------------------------------------------------------------------------

class FeedbackCreate(BaseModel):
    """Request body for POST /api/feedback."""
    category: str = Field(..., max_length=30, description="'bug' | 'suggestion' | 'general' | 'compliment'")
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1)
    rating: Optional[int] = Field(default=None, ge=1, le=5)


class FeedbackResponse(BaseModel):
    """Response shape for a feedback submission."""
    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    category: str
    subject: str
    message: str
    rating: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}
