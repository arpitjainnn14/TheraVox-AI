"""Text emotion analysis API endpoints."""

import logging
import asyncio
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_text_analyzer
from app.services import TextAnalyzerService
from app.models.schemas import EmotionResponse, ErrorResponse
from app.utils.emotion_utils import get_emotion_emoji, get_emotion_description

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze_text", response_model=EmotionResponse)
async def analyze_text(
    request: Request,
    analyzer: TextAnalyzerService = Depends(get_text_analyzer)
):
    """
    Analyze text for emotion.
    
    Accepts both JSON and form data.
    """
    try:
        # Handle both JSON and form data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            data = await request.json()
            text = data.get("text", "")
        else:
            form_data = await request.form()
            text = form_data.get("text", "")
        
        # Run analysis in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        emotion, confidence = await loop.run_in_executor(
            None,
            analyzer.analyze,
            text
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
        logger.error(f"Text analysis error: {str(e)}")
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"},
            status_code=500
        )
