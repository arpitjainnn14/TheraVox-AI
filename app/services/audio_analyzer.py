"""Audio emotion analysis service."""

import logging
import os
import threading
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Lazy imports
_np = None
_sf = None
_torch = None
_AutoModelForAudioClassification = None
_AutoFeatureExtractor = None
_SF_AVAILABLE = None
_HF_AVAILABLE = None


def _get_np():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_soundfile():
    global _sf, _SF_AVAILABLE
    if _SF_AVAILABLE is None:
        try:
            import soundfile as sf
            _sf = sf
            _SF_AVAILABLE = True
        except ImportError:
            _sf = None
            _SF_AVAILABLE = False
            logger.warning("soundfile not available – audio loading limited")
    return _sf, _SF_AVAILABLE


def _get_hf():
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
            _torch = _AutoModelForAudioClassification = _AutoFeatureExtractor = None
            _HF_AVAILABLE = False
            logger.info("HuggingFace transformers not available for audio")
    return _torch, _AutoModelForAudioClassification, _AutoFeatureExtractor, _HF_AVAILABLE


def _resample_audio(audio: 'np.ndarray', orig_sr: int, target_sr: int) -> 'np.ndarray':
    """Simple linear-interpolation resampling (no librosa/numba dep)."""
    np = _get_np()
    if orig_sr == target_sr:
        return audio
    duration      = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    if target_length == 0:
        return np.zeros(1, dtype=np.float32)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class AudioAnalyzerService:
    """
    Audio emotion analyzer.

    Primary model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
      - Wav2Vec2 XLS-R large, fine-tuned on RAVDESS
      - 8 emotions: angry, calm, disgust, fearful, happy, neutral, sad, surprised
      - Outperforms wav2vec2-base-superb-er significantly

    Falls back to lightweight acoustic rule-based analysis when the HF model
    is unavailable or fails.
    """

    # Maps model output labels → canonical emotion names
    _LABEL_MAP = {
        # ehcalabres model labels
        'angry':    'angry',
        'calm':     'neutral',   # calm ≈ neutral for our 7-class scheme
        'disgust':  'disgust',
        'fearful':  'fear',
        'happy':    'happy',
        'neutral':  'neutral',
        'sad':      'sad',
        'surprised':'surprise',
        # abbreviated labels (superb-style fallback)
        'neu': 'neutral',
        'hap': 'happy',
        'ang': 'angry',
        'exc': 'happy',         # excited → happy
        # generic full-word labels other models may emit
        'fear':     'fear',
        'surprise': 'surprise',
        'disgust':  'disgust',
    }

    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self._hf_model_id      = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self._hf_model         = None
        self._hf_extractor     = None
        self._hf_id2label: Optional[Dict[int, str]] = None
        self._model_loading    = False
        self._model_load_failed = False
        self._load_lock        = threading.Lock()

        # Minimum energy to bother running the HF model
        self._silence_threshold = 0.005

    # ------------------------------------------------------------------ #
    #  HF model loading                                                   #
    # ------------------------------------------------------------------ #

    def _ensure_hf_model(self) -> bool:
        # Fast path without lock
        if self._model_load_failed:
            return False
        if self._hf_model is not None:
            return True

        torch, AutoModelForAudioClassification, AutoFeatureExtractor, hf_available = _get_hf()
        if not hf_available:
            return False

        with self._load_lock:
            # Re-check after acquiring lock (another thread may have loaded it)
            if self._model_load_failed:
                return False
            if self._hf_model is not None:
                return True
            if self._model_loading:
                return False

            self._model_loading = True
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Try local cache first, then download
            try:
                model     = AutoModelForAudioClassification.from_pretrained(
                    self._hf_model_id, local_files_only=True)
                extractor = AutoFeatureExtractor.from_pretrained(
                    self._hf_model_id, local_files_only=True)
                logger.info("Loaded audio model from local cache")
            except Exception:
                logger.info(f"Downloading audio model: {self._hf_model_id}")
                model     = AutoModelForAudioClassification.from_pretrained(self._hf_model_id)
                extractor = AutoFeatureExtractor.from_pretrained(self._hf_model_id)

            model.to(device).eval()

            self._hf_model     = model
            self._hf_extractor = extractor
            self._hf_id2label  = getattr(model.config, "id2label", None)

            logger.info(f"Audio SER model ready on {device}. Labels: {self._hf_id2label}")
            self._model_loading = False
            return True

        except Exception as e:
            logger.warning(f"Failed to load HF audio model: {e}")
            self._model_load_failed = True
            self._model_loading     = False
            return False

    # ------------------------------------------------------------------ #
    #  Audio I/O                                                          #
    # ------------------------------------------------------------------ #

    def _load_audio(self, audio_path: str, target_sr: int = 16000) -> Optional[Tuple['np.ndarray', int]]:
        """Load audio file → (float32 mono array, sample_rate)."""
        np = _get_np()
        sf, sf_available = _get_soundfile()

        # 1. soundfile (supports WAV, FLAC, OGG …)
        if sf_available:
            try:
                audio, sr = sf.read(audio_path, dtype='float32')
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if sr != target_sr:
                    audio = _resample_audio(audio, sr, target_sr)
                return audio.astype(np.float32), target_sr
            except Exception as e:
                logger.warning(f"soundfile failed ({e}), trying wave fallback")

        # 2. stdlib wave (WAV-only)
        try:
            import wave as _wave
            with _wave.open(audio_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth  = wf.getsampwidth()
                orig_sr    = wf.getframerate()
                raw        = wf.readframes(wf.getnframes())

            if   sampwidth == 1:
                audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)  / 32768.0
            elif sampwidth == 4:
                audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)  / 2147483648.0
            else:
                logger.warning(f"Unsupported WAV sample width: {sampwidth}")
                return None

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
            if orig_sr != target_sr:
                audio = _resample_audio(audio, orig_sr, target_sr)
            return audio.astype(np.float32), target_sr

        except Exception as e:
            logger.error(f"All audio loading methods failed for {audio_path}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  HF inference                                                        #
    # ------------------------------------------------------------------ #

    def _predict_with_hf(self, audio_path: str) -> Optional[Tuple[str, float]]:
        """Run HF SER model; returns (emotion, confidence) or None."""
        if not self._ensure_hf_model():
            return None

        try:
            np    = _get_np()
            torch, _, __, _ = _get_hf()

            sampling_rate = getattr(self._hf_extractor, "sampling_rate", 16000)
            result        = self._load_audio(audio_path, target_sr=sampling_rate)
            if result is None:
                return None

            y, sr = result
            if y is None or len(y) == 0:
                return None

            # Skip truly silent files
            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms < self._silence_threshold:
                logger.info("Audio is (near) silent – returning neutral")
                return 'neutral', 0.60

            # Cap at 30 s to keep inference time reasonable
            max_samples = int(sampling_rate * 30.0)
            if len(y) > max_samples:
                y = y[:max_samples]

            inputs = self._hf_extractor(
                y,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            device = next(self._hf_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._hf_model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1).squeeze(0)

            pred_id    = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())

            # Resolve label
            if self._hf_id2label and pred_id in self._hf_id2label:
                raw_label = self._hf_id2label[pred_id]
            else:
                raw_label = str(pred_id)

            emotion = self._LABEL_MAP.get(raw_label.lower(), 'neutral')
            logger.debug(f"HF SER: raw={raw_label} → {emotion} ({confidence:.3f})")
            return emotion, confidence

        except Exception as e:
            logger.warning(f"HF prediction failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Acoustic fallback                                                   #
    # ------------------------------------------------------------------ #

    def _predict_with_acoustics(self, audio_path: str) -> Tuple[str, float]:
        """
        Lightweight numpy-only acoustic rule engine.
        Used only when the HF model is unavailable.
        """
        np = _get_np()
        result = self._load_audio(audio_path, target_sr=16000)
        if result is None:
            return 'neutral', 0.50

        y, sr = result
        if y is None or len(y) == 0:
            return 'neutral', 0.50

        try:
            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms < self._silence_threshold:
                return 'neutral', 0.60

            # Zero-crossing rate
            zcr = float(np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y)))

            # Spectral centroid
            fft   = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), 1 / sr)
            sc    = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-10))

            # Temporal energy variance
            chunk = sr // 20   # 50 ms
            if len(y) >= chunk * 4:
                rms_chunks  = [float(np.sqrt(np.mean(y[i:i+chunk] ** 2)))
                               for i in range(0, len(y) - chunk, chunk)]
                mean_e = float(np.mean(rms_chunks)) + 1e-10
                ev     = float(np.var(rms_chunks)) / mean_e
            else:
                ev = 0.0

            # Normalise
            rms_n = min(rms / 0.12, 1.0)
            zcr_n = min(zcr * 6,    1.0)
            sc_n  = min(sc / 3000,  1.0)

            logger.debug(f"Acoustics – rms={rms_n:.2f} zcr={zcr_n:.2f} sc={sc_n:.2f} ev={ev:.3f}")

            BASE = 0.55

            if rms_n > 0.75 and sc_n > 0.60 and zcr_n > 0.45:
                return 'angry',   min(BASE + 0.20, 0.80)
            if rms_n > 0.55 and sc_n > 0.65:
                return 'surprise',min(BASE + 0.10 + (0.08 if ev > 0.08 else 0), 0.75)
            if rms_n < 0.30 and sc_n < 0.40 and ev < 0.08:
                return 'sad',     min(BASE + 0.15, 0.75)
            if zcr_n > 0.50 and ev > 0.12 and 0.25 < rms_n < 0.60:
                return 'fear',    min(BASE + 0.05, 0.65)
            if 0.35 < rms_n < 0.75 and 0.30 < sc_n < 0.65 and ev > 0.05:
                return 'happy',   min(BASE + 0.12, 0.75)
            if 0.25 < rms_n < 0.45 and 0.25 < sc_n < 0.45 and ev < 0.10:
                return 'neutral', min(BASE + 0.10, 0.72)

            # Soft default
            if rms_n > 0.50:
                return 'happy',   BASE
            if rms_n > 0.25:
                return 'neutral', BASE
            return 'sad', BASE - 0.05

        except Exception as e:
            logger.error(f"Acoustic analysis failed: {e}")
            return 'neutral', 0.50

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(self, audio_path: str) -> Tuple[str, float]:
        """
        Analyze audio file for emotion.

        Args:
            audio_path: Path to audio file.

        Returns:
            tuple: (emotion, confidence)  where confidence ∈ [0, 1]
        """
        if not audio_path or not os.path.exists(audio_path):
            return 'neutral', 0.50

        result = self._predict_with_hf(audio_path)
        if result is not None:
            logger.info(f"HF SER result: {result[0]} ({result[1]:.2f})")
            return result

        logger.info("Falling back to acoustic analysis")
        result = self._predict_with_acoustics(audio_path)
        logger.info(f"Acoustic result: {result[0]} ({result[1]:.2f})")
        return result

    def get_status(self) -> Dict:
        _, _, _, hf_available = _get_hf()
        _, sf_available       = _get_soundfile()
        device = None
        if self._hf_model is not None:
            try:
                device = str(next(self._hf_model.parameters()).device)
            except Exception:
                pass
        return {
            "hf_libs_available":  hf_available,
            "hf_model_id":        self._hf_model_id,
            "hf_loaded":          self._hf_model is not None,
            "device":             device,
            "soundfile_available":sf_available,
            "emotions_supported": self.emotions,
        }
