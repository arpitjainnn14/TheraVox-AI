"""Main API router combining all route modules."""

from fastapi import APIRouter

from app.api.text import router as text_router
from app.api.audio import router as audio_router
from app.api.vision import router as vision_router
from app.api.system import router as system_router
from app.api.auth import router as auth_router
from app.api.wellness import router as wellness_router
from app.api.feedback import router as feedback_router

# Create main API router
api_router = APIRouter()

# Include API routes with /api prefix
api_router.include_router(text_router, prefix="/api", tags=["text"])
api_router.include_router(audio_router, prefix="/api", tags=["audio"])
api_router.include_router(vision_router, prefix="/api", tags=["vision"])
api_router.include_router(system_router, prefix="/api", tags=["system"])
api_router.include_router(auth_router, tags=["auth"])
api_router.include_router(wellness_router, tags=["wellness"])
api_router.include_router(feedback_router, tags=["feedback"])
