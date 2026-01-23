"""Audio emotion analysis API endpoints."""

import os
import logging
import asyncio
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_audio_analyzer
from app.services import AudioAnalyzerService
from app.models.schemas import EmotionResponse, AudioStatusResponse
from app.utils.emotion_utils import get_emotion_emoji, get_emotion_description

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze_audio", response_model=EmotionResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    analyzer: AudioAnalyzerService = Depends(get_audio_analyzer)
):
    """Analyze audio file for emotion."""
    try:
        # Save uploaded file
        os.makedirs("logs", exist_ok=True)
        file_path = os.path.join("logs", file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run analysis in executor
        loop = asyncio.get_event_loop()
        emotion, confidence = await loop.run_in_executor(
            None,
            analyzer.analyze,
            file_path
        )
        
        # Get additional info
        emoji = get_emotion_emoji(emotion)
        description = get_emotion_description(emotion, confidence)
        
        return EmotionResponse(
            emotion=emotion,
            confidence=confidence,
            emoji=emoji,
            description=description
        )
        
    except Exception as e:
        logger.error(f"Audio analysis error: {str(e)}")
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"},
            status_code=500
        )


@router.get("/audio_status", response_model=AudioStatusResponse)
async def audio_status(
    analyzer: AudioAnalyzerService = Depends(get_audio_analyzer)
):
    """Get audio analyzer runtime status."""
    try:
        status = analyzer.get_status()
        return AudioStatusResponse(**status)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
