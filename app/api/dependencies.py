"""Dependency injection for API routes."""

from functools import lru_cache
from app.services import TextAnalyzerService, AudioAnalyzerService, VisionAnalyzerService
from app.core.config import get_settings


# Singleton instances for services (lazy-loaded)
_text_analyzer = None
_audio_analyzer = None
_vision_analyzer = None


def get_text_analyzer() -> TextAnalyzerService:
    """Get or create text analyzer instance."""
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzerService()
    return _text_analyzer


def get_audio_analyzer() -> AudioAnalyzerService:
    """Get or create audio analyzer instance."""
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AudioAnalyzerService()
    return _audio_analyzer


def get_vision_analyzer() -> VisionAnalyzerService:
    """Get or create vision analyzer instance."""
    global _vision_analyzer
    if _vision_analyzer is None:
        settings = get_settings()
        _vision_analyzer = VisionAnalyzerService(settings)
    return _vision_analyzer
