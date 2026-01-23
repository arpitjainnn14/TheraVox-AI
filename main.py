"""
TheraVox AI - Main Application Entry Point

A modular, scalable emotion analysis system supporting:
- Text emotion analysis
- Audio emotion recognition  
- Vision-based facial emotion detection
"""

import logging
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("🚀 Initializing TheraVox AI...")

# Import core components (lightweight)
from app.core.lifespan import lifespan
from app.core.config import get_settings

logger.info("✓ Core modules loaded")

# Get settings
settings = get_settings()
logger.info("✓ Configuration loaded")

# Create FastAPI application
logger.info("📦 Creating FastAPI application...")
app = FastAPI(
    title="TheraVox AI",
    description="Emotion Analysis API supporting text, audio, and vision",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=500)
logger.info("✓ Middleware configured")

# Mount static files
app.mount("/static", StaticFiles(directory=settings.get("static_dir", "static")), name="static")
logger.info("✓ Static files mounted")

# Import and include API router (deferred to avoid loading heavy dependencies)
logger.info("📡 Loading API routes...")
from app.api.router import api_router
app.include_router(api_router)
logger.info("✓ API routes loaded")

logger.info("✨ TheraVox AI ready! Models will load on first use.")


if __name__ == "__main__":
    import uvicorn
    
    host = settings.get("host", "127.0.0.1")
    port = settings.get("port", 8000)
    
    logger.info(f"Starting TheraVox AI on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
