"""Application configuration management."""

import os
from functools import lru_cache
from typing import Any, Dict
import json


class Settings:
    """Application settings with environment-aware configuration."""
    
    def __init__(self, settings_file: str = "settings.json"):
        self.settings_file = settings_file
        
        # Default configuration
        self._config: Dict[str, Any] = {
            # Server settings
            "host": os.getenv("HOST", "127.0.0.1"),
            "port": int(os.getenv("PORT", "8000")),
            
            # Model settings
            "emotion_smoothing": 3,
            "detection_quality": "balanced",  # performance, balanced, quality
            "min_face_size": 50,
            
            # Performance settings
            "opencv_threads": int(os.getenv("OPENCV_THREADS", "2")),
            "torch_threads": int(os.getenv("TORCH_NUM_THREADS", "2")),
            
            # Paths
            "logs_dir": "logs",
            "screenshots_dir": "screenshots",
            "static_dir": "static",
            "templates_dir": "templates",
            
            # Feature flags
            "enable_audio": True,
            "enable_vision": True,
            "enable_text": True,
        }
        
        # Load from file if exists
        self._load_from_file()
    
    def _load_from_file(self) -> None:
        """Load settings from JSON file if it exists."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
            except Exception:
                pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        except Exception:
            pass
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
