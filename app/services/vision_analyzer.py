"""Vision-based emotion analysis service (face detection + emotion recognition)."""

import logging
from typing import Tuple, List, Dict
from collections import deque

logger = logging.getLogger(__name__)

# Lazy imports - only import when needed
_cv2 = None
_np = None
_DeepFace = None
_DEEPFACE_AVAILABLE = None


def _get_cv2():
    """Lazy import cv2."""
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_np():
    """Lazy import numpy."""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_deepface():
    """Lazy import DeepFace."""
    global _DeepFace, _DEEPFACE_AVAILABLE
    if _DEEPFACE_AVAILABLE is None:
        try:
            from deepface import DeepFace
            _DeepFace = DeepFace
            _DEEPFACE_AVAILABLE = True
        except ImportError:
            _DeepFace = None
            _DEEPFACE_AVAILABLE = False
            logger.warning("DeepFace not available - vision analysis disabled")
    return _DeepFace, _DEEPFACE_AVAILABLE


class FaceDetector:
    """Face detection using OpenCV Haar Cascades."""
    
    def __init__(self):
        cv2 = _get_cv2()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of face locations (x, y, w, h)
        """
        cv2 = _get_cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
        )
        
        return list(faces)
    
    def extract_face(self, frame, location: Tuple[int, int, int, int]):
        """Extract face region from frame."""
        x, y, w, h = location
        return frame[y:y + h, x:x + w]
    
    def is_valid_face(self, face_img, min_size: int = 30) -> bool:
        """Check if detected face is valid."""
        if face_img is None or face_img.size == 0:
            return False
        height, width = face_img.shape[:2]
        return height >= min_size and width >= min_size


class EmotionRecognizer:
    """Emotion recognition using DeepFace."""
    
    def __init__(self, settings=None):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.settings = settings
        
        # Smoothing for temporal consistency
        smoothing = 3
        if settings:
            try:
                smoothing = max(1, int(settings.get('emotion_smoothing', 3)))
            except:
                pass
        
        self.emotion_history = deque(maxlen=smoothing)
        self.confidence_history = deque(maxlen=smoothing)
        
        # Emotion weights and thresholds
        self.emotion_weights = {
            'happy': 1.0, 'surprise': 1.0, 'angry': 1.0,
            'fear': 1.0, 'sad': 1.0, 'disgust': 1.0, 'neutral': 0.9
        }
        
        self.confidence_thresholds = {
            'happy': 0.20, 'surprise': 0.20, 'neutral': 0.50, 'default': 0.20
        }
    
    def _get_backend_settings(self) -> Tuple[str, bool, bool]:
        """Get DeepFace backend settings."""
        quality = 'balanced'
        if self.settings:
            try:
                quality = self.settings.get('detection_quality', 'balanced') or 'balanced'
            except:
                pass
        
        if quality == 'performance':
            return 'opencv', False, False
        elif quality == 'quality':
            return 'retinaface', True, True
        else:
            return 'mediapipe', True, True
    
    def _preprocess_face(self, face_img):
        """Preprocess face for better analysis."""
        cv2 = _get_cv2()
        # Resize if too small
        if face_img.shape[0] < 96 or face_img.shape[1] < 96:
            face_img = cv2.resize(face_img, (96, 96))
        
        # Enhance contrast
        if len(face_img.shape) == 3:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            face_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        return face_img
    
    def analyze_emotion(self, face_img) -> Tuple[str, float]:
        """
        Analyze emotion from face image.
        
        Args:
            face_img: Face image (BGR)
            
        Returns:
            tuple: (emotion, confidence)
        """
        DeepFace, available = _get_deepface()
        if not available:
            return 'neutral', 0.0
            
        if face_img is None or face_img.size == 0:
            return 'neutral', 0.0
        
        try:
            # Preprocess
            face_img = self._preprocess_face(face_img)
            
            # Get backend settings
            backend, align, enforce = self._get_backend_settings()
            
            # Analyze with DeepFace
            try:
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=enforce,
                    align=align,
                    detector_backend=backend,
                    silent=True
                )
            except Exception:
                # Fallback to OpenCV backend
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    align=False,
                    detector_backend='opencv',
                    silent=True
                )
            
            # Parse result
            analysis = result[0] if isinstance(result, (list, tuple)) else result
            emotions = analysis.get('emotion', {})
            
            # Apply weights
            weighted_emotions = {
                emo: emotions[emo] * self.emotion_weights.get(emo, 1.0)
                for emo in self.emotions
            }
            
            # Get top emotion
            sorted_emotions = sorted(weighted_emotions.items(), key=lambda x: x[1], reverse=True)
            top_emotion, _ = sorted_emotions[0]
            confidence = emotions[top_emotion] / 100.0
            
            # Temporal smoothing
            self.emotion_history.append(top_emotion)
            self.confidence_history.append(confidence)
            
            # Use most recent if consistent
            if len(self.emotion_history) >= 2:
                if self.emotion_history[-1] == self.emotion_history[-2]:
                    return top_emotion, confidence
            
            return top_emotion, confidence
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return 'neutral', 0.1


class VisionAnalyzerService:
    """Vision-based emotion analysis combining face detection and emotion recognition."""
    
    def __init__(self, settings=None):
        self.face_detector = FaceDetector()
        self.emotion_recognizer = EmotionRecognizer(settings)
    
    def analyze_frame(self, frame) -> List[Dict]:
        """
        Analyze a video frame for faces and emotions.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of dictionaries with emotion data for each face
        """
        if frame is None or frame.size == 0:
            return []
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        results = []
        for face_location in faces:
            # Extract face
            face_img = self.face_detector.extract_face(frame, face_location)
            
            # Validate face
            if not self.face_detector.is_valid_face(face_img):
                continue
            
            # Analyze emotion
            emotion, confidence = self.emotion_recognizer.analyze_emotion(face_img)
            
            results.append({
                'location': face_location,
                'emotion': emotion,
                'confidence': confidence
            })
        
        return results
