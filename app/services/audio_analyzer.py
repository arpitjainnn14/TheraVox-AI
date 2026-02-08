"""Audio emotion analysis service."""

import logging
import os
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Lazy imports - only import when needed
_np = None
_sf = None
_torch = None
_AutoModelForAudioClassification = None
_Wav2Vec2FeatureExtractor = None
_SF_AVAILABLE = None
_HF_AVAILABLE = None


def _get_np():
    """Lazy import numpy."""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_soundfile():
    """Lazy import soundfile (faster than librosa for loading)."""
    global _sf, _SF_AVAILABLE
    if _SF_AVAILABLE is None:
        try:
            import soundfile as sf
            _sf = sf
            _SF_AVAILABLE = True
        except ImportError:
            _sf = None
            _SF_AVAILABLE = False
            logger.warning("soundfile not available - audio loading limited")
    return _sf, _SF_AVAILABLE


def _get_hf():
    """Lazy import HuggingFace transformers."""
    global _torch, _AutoModelForAudioClassification, _Wav2Vec2FeatureExtractor, _HF_AVAILABLE
    if _HF_AVAILABLE is None:
        try:
            import torch
            from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
            _torch = torch
            _AutoModelForAudioClassification = AutoModelForAudioClassification
            _Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor
            _HF_AVAILABLE = True
        except ImportError:
            _torch = None
            _AutoModelForAudioClassification = None
            _Wav2Vec2FeatureExtractor = None
            _HF_AVAILABLE = False
            logger.info("HuggingFace transformers not available for audio")
    return _torch, _AutoModelForAudioClassification, _Wav2Vec2FeatureExtractor, _HF_AVAILABLE


def _resample_audio(audio: 'np.ndarray', orig_sr: int, target_sr: int) -> 'np.ndarray':
    """Simple resampling using numpy (no librosa/numba dependency)."""
    np = _get_np()
    if orig_sr == target_sr:
        return audio
    
    # Simple linear interpolation resampling
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    
    if target_length == 0:
        return np.zeros(1, dtype=audio.dtype)
    
    # Use numpy interp for resampling
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class AudioAnalyzerService:
    """
    Audio emotion analyzer using HuggingFace speech emotion recognition.
    Uses a robust wav2vec2-based model with acoustic fallback.
    """
    
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Use facebook/hubert-large-ls960-ft - pretrained model
        # Then we use acoustic analysis as primary method since HF models
        # often misclassify non-speech audio as "angry"
        self._hf_model_id = "superb/wav2vec2-base-superb-er"
        self._hf_model = None
        self._hf_extractor = None
        self._hf_id2label = None
        self._model_loading = False
        self._model_load_failed = False
        
        # Speech detection threshold - if audio doesn't look like speech, use acoustic fallback
        self._speech_energy_threshold = 0.01
    
    def _ensure_hf_model(self) -> bool:
        """Lazy load the HuggingFace SER model with timeout protection."""
        if self._model_load_failed:
            return False
            
        if self._model_loading:
            return False
            
        torch, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, hf_available = _get_hf()
        sf, sf_available = _get_soundfile()
        
        if not hf_available or not sf_available:
            return False
            
        if self._hf_model is not None:
            return True
        
        self._model_loading = True
        
        try:
            torch, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, _ = _get_hf()
            
            # Select device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model with local_files_only first (if cached)
            try:
                model = AutoModelForAudioClassification.from_pretrained(
                    self._hf_model_id,
                    local_files_only=True
                )
                extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self._hf_model_id,
                    local_files_only=True
                )
                logger.info("Loaded model from cache")
            except Exception:
                # Download if not cached
                logger.info(f"Downloading model: {self._hf_model_id}")
                model = AutoModelForAudioClassification.from_pretrained(self._hf_model_id)
                extractor = Wav2Vec2FeatureExtractor.from_pretrained(self._hf_model_id)
            
            model.to(device)
            model.eval()
            
            self._hf_model = model
            self._hf_extractor = extractor
            self._hf_id2label = getattr(model.config, "id2label", None)
            
            logger.info(f"Loaded HF SER model: {self._hf_model_id} on {device}")
            self._model_loading = False
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load HF SER model: {e}")
            self._model_load_failed = True
            self._model_loading = False
            return False
    
    def _load_audio(self, audio_path: str, target_sr: int = 16000) -> Optional[Tuple['np.ndarray', int]]:
        """Load audio file and resample to target sample rate."""
        np = _get_np()
        sf, sf_available = _get_soundfile()
        
        if not sf_available:
            return None
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != target_sr:
                audio = _resample_audio(audio, sr, target_sr)
                sr = target_sr
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None
    
    def _is_likely_speech(self, audio: 'np.ndarray', sr: int) -> bool:
        """
        Check if audio likely contains speech based on acoustic properties.
        This helps avoid misclassification of music/noise as emotions.
        """
        np = _get_np()
        
        try:
            # Check minimum energy
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < self._speech_energy_threshold:
                return False
            
            # Check for speech-like frequency content using FFT
            fft = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            
            # Speech typically has energy between 85-300 Hz (fundamental frequency)
            # and formants up to ~5000 Hz
            speech_band_mask = (freqs >= 85) & (freqs <= 5000)
            low_speech_mask = (freqs >= 85) & (freqs <= 300)
            
            speech_energy = np.sum(fft[speech_band_mask])
            low_speech_energy = np.sum(fft[low_speech_mask])
            total_energy = np.sum(fft) + 1e-10
            
            speech_ratio = speech_energy / total_energy
            
            # Check for fundamental frequency presence
            has_fundamental = low_speech_energy > (total_energy * 0.05)
            
            # Check for temporal variation (speech has varying amplitude)
            # Split into chunks and check variance
            chunk_size = sr // 10  # 100ms chunks
            if len(audio) >= chunk_size * 3:
                chunks = [audio[i:i+chunk_size] for i in range(0, len(audio) - chunk_size, chunk_size)]
                chunk_energies = [np.sqrt(np.mean(c ** 2)) for c in chunks[:10]]
                if len(chunk_energies) > 0 and np.mean(chunk_energies) > 0:
                    energy_variance = np.var(chunk_energies) / (np.mean(chunk_energies) + 1e-10)
                    # Speech has moderate to high variance in energy over time
                    has_temporal_variation = energy_variance > 0.02
                else:
                    has_temporal_variation = True
            else:
                has_temporal_variation = True
            
            # More stringent criteria: speech-band energy, fundamental presence, and temporal variation
            return speech_ratio > 0.4 and has_fundamental and has_temporal_variation
            
        except Exception:
            return True  # Assume speech if detection fails
    
    def _predict_with_hf(self, audio_path: str) -> Optional[Tuple[str, float]]:
        """Predict emotion using HuggingFace model."""
        if not self._ensure_hf_model():
            return None
            
        try:
            np = _get_np()
            torch, _, _, _ = _get_hf()
            
            # Load audio at model's expected sample rate
            sampling_rate = getattr(self._hf_extractor, "sampling_rate", 16000)
            result = self._load_audio(audio_path, target_sr=sampling_rate)
            
            if result is None:
                return None
            
            y, sr = result
            
            if y is None or len(y) == 0:
                return None
            
            # Check if audio is likely speech
            if not self._is_likely_speech(y, sr):
                logger.info("Audio doesn't appear to be speech, using acoustic analysis")
                return None  # Fall back to acoustic analysis
            
            # Limit audio length (max 30 seconds)
            max_length = int(sampling_rate * 30.0)
            if len(y) > max_length:
                y = y[:max_length]
            
            # Extract features
            inputs = self._hf_extractor(
                y,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            device = next(self._hf_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                
                # Get all probabilities for analysis
                prob_list = probs.tolist()
                pred_id = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[pred_id].item())
                
                # Check for suspicious predictions:
            # If model is very confident (>92%) about angry/fear, and other probs are very low,
            # this might be misclassification of non-speech audio
            if (pred_id == 2 and confidence > 0.92) or (pred_id == 3 and confidence > 0.90):  # 2=ang, 3=sad in superb
                # Check if neutral probability is reasonable
                neutral_prob = prob_list[0] if len(prob_list) > 0 else 0
                if neutral_prob < 0.02:
                    # Suspicious - likely not real speech, fall back to acoustic
                    logger.info(f"Suspicious prediction (label={pred_id}, conf={confidence:.2f}), using acoustic fallback")
                    return None
            
            # Additional check: if confidence is too low, use acoustic fallback for refinement
            if confidence < 0.55:
                logger.info(f"Low confidence ({confidence:.2f}), using acoustic fallback")
                return None
            # Get label
            if self._hf_id2label and pred_id in self._hf_id2label:
                label = self._hf_id2label[pred_id]
            else:
                # Default mapping for superb model: 0=neu, 1=hap, 2=ang, 3=sad
                label_list = ['neu', 'hap', 'ang', 'sad']
                label = label_list[pred_id] if pred_id < len(label_list) else 'neu'
            
            # Normalize label to our emotion set (handle abbreviated and full labels)
            label_map = {
                # Abbreviated labels (superb model)
                'neu': 'neutral',
                'hap': 'happy',
                'ang': 'angry',
                'sad': 'sad',
                # Full labels (other models)
                'angry': 'angry',
                'disgust': 'disgust',
                'fear': 'fear',
                'fearful': 'fear',
                'happy': 'happy',
                'neutral': 'neutral',
                'surprise': 'surprise',
                'surprised': 'surprise',
            }
            
            emotion = label_map.get(label.lower(), 'neutral')
            return emotion, confidence
            
        except Exception as e:
            logger.warning(f"HF prediction failed: {e}")
            return None
    
    def _predict_with_acoustics(self, audio_path: str) -> Tuple[str, float]:
        """
        Fallback acoustic-based prediction using simple numpy operations.
        No librosa/numba dependencies.
        Improved rules based on emotion acoustic characteristics.
        """
        np = _get_np()
        
        result = self._load_audio(audio_path, target_sr=16000)
        if result is None:
            return 'neutral', 0.0
        
        y, sr = result
        
        if y is None or len(y) == 0:
            return 'neutral', 0.0
            
        try:
            # Extract basic acoustic features using only numpy
            
            # 1. Root Mean Square Energy (loudness)
            rms = np.sqrt(np.mean(y ** 2))
            
            # 2. Zero Crossing Rate (roughness/noisiness)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / 2
            zcr = zero_crossings / len(y)
            
            # 3. Simple spectral analysis using FFT
            fft = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            
            # Spectral centroid (brightness)
            spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum = np.cumsum(fft)
            rolloff_threshold = 0.85 * cumsum[-1]
            rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
            spectral_rolloff = freqs[min(rolloff_idx, len(freqs)-1)]
            
            # 4. Pitch estimation (simplified autocorrelation)
            # Look for fundamental frequency in speech range (80-400 Hz)
            min_lag = int(sr / 400)  # 400 Hz
            max_lag = int(sr / 80)   # 80 Hz
            
            pitch_estimate = 0
            pitch_variance = 0
            if max_lag < len(y):
                autocorr = np.correlate(y[:max_lag*2], y[:max_lag*2], mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peak in valid range
                valid_autocorr = autocorr[min_lag:max_lag]
                if len(valid_autocorr) > 0:
                    peak_idx = np.argmax(valid_autocorr) + min_lag
                    pitch_estimate = sr / peak_idx if peak_idx > 0 else 0
                    
                    # Calculate pitch variance (variation in pitch over time)
                    chunk_size = sr // 10  # 100ms chunks
                    if len(y) >= chunk_size * 3:
                        chunks = [y[i:i+chunk_size] for i in range(0, len(y) - chunk_size, chunk_size)]
                        chunk_pitches = []
                        for chunk in chunks[:10]:
                            if len(chunk) > max_lag:
                                ac = np.correlate(chunk[:max_lag], chunk[:max_lag], mode='full')
                                ac = ac[len(ac)//2:]
                                vac = ac[min_lag:max_lag]
                                if len(vac) > 0:
                                    pidx = np.argmax(vac) + min_lag
                                    chunk_pitches.append(sr / pidx if pidx > 0 else 0)
                        if len(chunk_pitches) > 2:
                            pitch_variance = np.var(chunk_pitches) / (np.mean(chunk_pitches) + 1e-10)
            
            # 5. Temporal energy variation
            chunk_size = sr // 20  # 50ms chunks
            if len(y) >= chunk_size * 4:
                chunks = [y[i:i+chunk_size] for i in range(0, len(y) - chunk_size, chunk_size)]
                chunk_energies = [np.sqrt(np.mean(c ** 2)) for c in chunks[:20]]
                if len(chunk_energies) > 0 and np.mean(chunk_energies) > 0:
                    energy_variance = np.var(chunk_energies) / (np.mean(chunk_energies) + 1e-10)
                else:
                    energy_variance = 0
            else:
                energy_variance = 0
            
            # Normalize features with better thresholds
            rms_norm = min(rms / 0.12, 1.0)  # More sensitive normalization
            zcr_norm = min(zcr * 6, 1.0)     # Less aggressive normalization
            cent_norm = min(spectral_centroid / 3000, 1.0)  # More sensitive
            pitch_norm = min(pitch_estimate / 300, 1.0) if pitch_estimate > 0 else 0.5
            
            # Log features for debugging
            logger.debug(f"Audio features - RMS: {rms:.4f} ({rms_norm:.2f}), ZCR: {zcr:.4f} ({zcr_norm:.2f}), "
                        f"Cent: {spectral_centroid:.1f} ({cent_norm:.2f}), Pitch: {pitch_estimate:.1f} ({pitch_norm:.2f}), "
                        f"PitchVar: {pitch_variance:.4f}, EnergyVar: {energy_variance:.4f}")
            
            # Improved rule-based emotion detection - reordered for better classification
            base_confidence = 0.62
            
            # HAPPY: Medium-high energy + moderate brightness (most common positive emotion)
            # This should catch normal positive speech
            if rms_norm > 0.35 and rms_norm < 0.75:
                if cent_norm > 0.30 and cent_norm < 0.65:
                    # Check it's not sad (has variation)
                    if energy_variance > 0.05 or pitch_variance > 0.03:
                        confidence = base_confidence + 0.12
                        logger.debug("Classified as HAPPY (normal positive speech)")
                        return 'happy', min(confidence, 0.82)
            
            # ANGRY: Very high energy + high brightness + high ZCR
            if rms_norm > 0.75 and cent_norm > 0.60 and zcr_norm > 0.45:
                confidence = base_confidence + 0.18
                logger.debug("Classified as ANGRY (high energy + brightness + ZCR)")
                return 'angry', min(confidence, 0.85)
            
            # SURPRISE: High energy + very high brightness + pitch variance
            elif rms_norm > 0.55 and cent_norm > 0.65:
                confidence = base_confidence + 0.08
                if pitch_variance > 0.08:
                    confidence += 0.08
                logger.debug("Classified as SURPRISE (very high brightness)")
                return 'surprise', min(confidence, 0.78)
            
            # SAD: Low energy + low brightness + low variance
            elif rms_norm < 0.30:
                if cent_norm < 0.40 and energy_variance < 0.08:
                    confidence = base_confidence + 0.12
                    logger.debug("Classified as SAD (low energy + brightness)")
                    return 'sad', min(confidence, 0.82)
            
            # FEAR: Medium energy + high ZCR + high energy variance (shaky)
            elif zcr_norm > 0.50 and energy_variance > 0.12:
                if rms_norm > 0.25 and rms_norm < 0.60:
                    confidence = base_confidence + 0.02
                    logger.debug("Classified as FEAR (high ZCR + energy variance)")
                    return 'fear', min(confidence, 0.70)
            
            # DISGUST: Very specific - low brightness + very high ZCR + specific energy range
            # Make this much more restrictive to avoid false positives
            elif rms_norm > 0.35 and rms_norm < 0.50:
                if cent_norm < 0.25 and zcr_norm > 0.60:
                    confidence = base_confidence - 0.10
                    logger.debug("Classified as DISGUST (low brightness + very high ZCR)")
                    return 'disgust', min(confidence, 0.60)
            
            # NEUTRAL: Medium energy + medium brightness + low variation
            elif rms_norm > 0.25 and rms_norm < 0.45:
                if cent_norm > 0.25 and cent_norm < 0.45 and energy_variance < 0.10:
                    confidence = base_confidence + 0.08
                    logger.debug("Classified as NEUTRAL (medium energy + brightness)")
                    return 'neutral', min(confidence, 0.75)
            
            # Very low energy - likely neutral
            elif rms_norm < 0.20:
                logger.debug("Classified as NEUTRAL (very low energy)")
                return 'neutral', base_confidence
            
            # Default fallback - prefer happy for medium-high energy
            logger.debug(f"Using fallback - RMS: {rms_norm:.2f}")
            if rms_norm > 0.50:
                return 'happy', base_confidence  # Medium-high energy default to happy
            elif rms_norm > 0.30:
                return 'neutral', base_confidence  # Medium energy
            else:
                return 'sad', base_confidence - 0.05  # Low energy
                    
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
        if not audio_path or not os.path.exists(audio_path):
            return 'neutral', 0.0
        
        # Try HF model first
        result = self._predict_with_hf(audio_path)
        if result is not None:
            logger.info(f"Using HF model prediction: {result[0]} (conf: {result[1]:.2f})")
            return result
        
        # Fallback to acoustic analysis
        logger.info("Using acoustic analysis fallback")
        result = self._predict_with_acoustics(audio_path)
        logger.info(f"Acoustic prediction: {result[0]} (conf: {result[1]:.2f})")
        return result
    
    def get_status(self) -> Dict:
        """Get runtime status information."""
        _, _, _, hf_available = _get_hf()
        _, sf_available = _get_soundfile()
        
        device = None
        if self._hf_model is not None:
            try:
                device = str(next(self._hf_model.parameters()).device)
            except:
                pass
        
        return {
            "hf_libs_available": hf_available,
            "hf_model_id": self._hf_model_id,
            "hf_loaded": self._hf_model is not None,
            "device": device,
            "soundfile_available": sf_available,
            "emotions_supported": self.emotions,
        }
