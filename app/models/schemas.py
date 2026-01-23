"""Pydantic models for request and response validation."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


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
    librosa_available: bool
    emotions_supported: list


class VisionAnalysisResponse(BaseModel):
    """Response model for vision analysis."""
    faces: list[EmotionResponse]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
