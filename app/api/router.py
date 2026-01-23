"""Main API router combining all route modules."""

from fastapi import APIRouter

from app.api import pages, text, audio, vision, system

# Create main API router
api_router = APIRouter()

# Include page routes (no prefix)
api_router.include_router(pages.router, tags=["pages"])

# Include API routes with /api prefix
api_router.include_router(text.router, prefix="/api", tags=["text"])
api_router.include_router(audio.router, prefix="/api", tags=["audio"])
api_router.include_router(vision.router, prefix="/api", tags=["vision"])
api_router.include_router(system.router, prefix="/api", tags=["system"])
