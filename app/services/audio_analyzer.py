"""Audio emotion analysis service."""

import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Lazy imports - only import when needed
_np = None
_librosa = None
_sf = None
_torch = None
_AutoModelForAudioClassification = None
_AutoFeatureExtractor = None
_LIBROSA_AVAILABLE = None
_HF_AVAILABLE = None


def _get_np():
    """Lazy import numpy."""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_librosa():
    """Lazy import librosa."""
    global _librosa, _sf, _LIBROSA_AVAILABLE
    if _LIBROSA_AVAILABLE is None:
        try:
            import librosa
            import soundfile as sf
            _librosa = librosa
            _sf = sf
            _LIBROSA_AVAILABLE = True
        except ImportError:
            _librosa = None
            _sf = None
            _LIBROSA_AVAILABLE = False
            logger.warning("librosa not available - audio analysis disabled")
    return _librosa, _sf, _LIBROSA_AVAILABLE


def _get_hf():
    """Lazy import HuggingFace transformers."""
    global _torch, _AutoModelForAudioClassification, _AutoFeatureExtractor, _HF_AVAILABLE
    if _HF_AVAILABLE is None:
        try:
            import torch
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            _torch = torch
            _AutoModelForAudioClassification = AutoModelForAudioClassification
            _AutoFeatureExtractor = AutoFeatureExtractor
            _HF_AVAILABLE = True
        except ImportError:
            _torch = None
            _AutoModelForAudioClassification = None
            _AutoFeatureExtractor = None
            _HF_AVAILABLE = False
            logger.info("HuggingFace transformers not available for audio")
    return _torch, _AutoModelForAudioClassification, _AutoFeatureExtractor, _HF_AVAILABLE


class AudioAnalyzerService:
    """
    Audio emotion analyzer using Hugging Face Whisper SER model.
    Falls back to acoustic feature analysis if model unavailable.
    """
    
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Lazy-loaded HF model
        self._hf_model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
        self._hf_model = None
        self._hf_extractor = None
        self._hf_id2label = None
    
    def _ensure_hf_model(self) -> bool:
        """Lazy load the HuggingFace Whisper SER model."""
        torch, AutoModelForAudioClassification, AutoFeatureExtractor, hf_available = _get_hf()
        librosa, sf, librosa_available = _get_librosa()
        
        if not hf_available or not librosa_available:
            return False
            
        if self._hf_model is not None:
            return True
            
        try:
            import os
            torch, AutoModelForAudioClassification, AutoFeatureExtractor, _ = _get_hf()
            
            # Select device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model and feature extractor
            model = AutoModelForAudioClassification.from_pretrained(self._hf_model_id)
            extractor = AutoFeatureExtractor.from_pretrained(self._hf_model_id, do_normalize=True)
            
            model.to(device)
            model.eval()
            
            self._hf_model = model
            self._hf_extractor = extractor
            self._hf_id2label = getattr(model.config, "id2label", None)
            
            logger.info(f"Loaded HF Whisper SER model: {self._hf_model_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load HF SER model: {e}")
            return False
    
    def _predict_with_hf(self, audio_path: str) -> Optional[Tuple[str, float]]:
        """Predict emotion using HuggingFace model."""
        if not self._ensure_hf_model():
            return None
            
        try:
            librosa, _, _ = _get_librosa()
            np = _get_np()
            torch, _, _, _ = _get_hf()
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            if y is None or len(y) == 0:
                return None
            
            # Prepare audio for model
            sampling_rate = getattr(self._hf_extractor, "sampling_rate", 16000)
            max_length = int(sampling_rate * 30.0)  # 30 seconds max
            
            # Resample if needed
            if sr != sampling_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sampling_rate)
            
            # Truncate or pad
            if len(y) > max_length:
                y = y[:max_length]
            else:
                y = np.pad(y, (0, max_length - len(y)))
            
            # Extract features
            inputs = self._hf_extractor(
                y,
                sampling_rate=sampling_rate,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Move to device
            device = next(self._hf_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                pred_id = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[pred_id].item())
            
            # Get label
            if self._hf_id2label and pred_id in self._hf_id2label:
                label = self._hf_id2label[pred_id]
            else:
                return None
            
            # Normalize label to our emotion set
            label_map = {
                'angry': 'angry',
                'disgust': 'disgust',
                'fearful': 'fear',
                'happy': 'happy',
                'neutral': 'neutral',
                'sad': 'sad',
                'surprised': 'surprise',
            }
            
            emotion = label_map.get(label.lower(), 'neutral')
            return emotion, confidence
            
        except Exception as e:
            logger.warning(f"HF prediction failed: {e}")
            return None
    
    def _predict_with_acoustics(self, audio_path: str) -> Tuple[str, float]:
        """Fallback acoustic-based prediction."""
        librosa, _, librosa_available = _get_librosa()
        np = _get_np()
        
        if not librosa_available:
            return 'neutral', 0.0
            
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            if y is None or len(y) == 0:
                return 'neutral', 0.0
            
            # Extract basic acoustic features
            rms = np.mean(librosa.feature.rms(y=y)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            
            # Simple rule-based emotion detection
            if rms > 0.1:  # High energy
                if spectral_centroid > 3000:
                    return 'angry', 0.6
                else:
                    return 'happy', 0.6
            elif rms < 0.05:  # Low energy
                return 'sad', 0.6
            else:
                if zcr > 0.1:
                    return 'fear', 0.5
                else:
                    return 'neutral', 0.5
                    
        except Exception as e:
            logger.error(f"Acoustic analysis failed: {e}")
            return 'neutral', 0.0
    
    def analyze(self, audio_path: str) -> Tuple[str, float]:
        """
        Analyze audio file for emotion.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            tuple: (emotion, confidence)
        """
        if not audio_path:
            return 'neutral', 0.0
        
        # Try HF model first
        result = self._predict_with_hf(audio_path)
        if result is not None:
            return result
        
        # Fallback to acoustic analysis
        return self._predict_with_acoustics(audio_path)
    
    def get_status(self) -> Dict:
        """Get runtime status information."""
        _, _, _, hf_available = _get_hf()
        _, _, librosa_available = _get_librosa()
        
        return {
            "hf_libs_available": hf_available,
            "hf_model_id": self._hf_model_id,
            "hf_loaded": self._hf_model is not None,
            "device": str(next(self._hf_model.parameters()).device) if self._hf_model else None,
            "librosa_available": librosa_available,
            "emotions_supported": self.emotions,
        }
