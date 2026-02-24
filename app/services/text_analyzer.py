"""Text emotion analysis service."""

import re
import logging
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers library not available, using fallback lexicon-based analysis")


class TextAnalyzerService:
    """
    Text emotion analyzer using Hugging Face transformer model with lexicon-based fallback.
    
    Supports 7 emotions: happy, sad, angry, fear, disgust, surprise, neutral
    """
    
    def __init__(self):
        self._hf_pipe = None
        self.emotions = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
        self.neutral_threshold = 0.35
        
        # Lexicon for fallback analysis
        self.lexicon: Dict[str, set] = {
            'happy': {'happy', 'joy', 'joyful', 'glad', 'excited', 'love', 'lovely', 'awesome', 'great', 
                     'amazing', 'fantastic', 'wonderful', 'delighted', 'pleased', 'cheerful', 'proud'},
            'sad': {'sad', 'unhappy', 'down', 'depressed', 'cry', 'crying', 'tears', 'heartbroken', 
                   'miserable', 'sorrow', 'grief', 'lonely', 'hurt', 'disappointed'},
            'angry': {'angry', 'furious', 'annoyed', 'irritated', 'rage', 'mad', 'livid', 'outraged', 
                     'resentful', 'hate', 'frustrated', 'offended', 'bitter'},
            'fear': {'afraid', 'scared', 'terrified', 'anxious', 'anxiety', 'worried', 'worry', 'panic', 
                    'nervous', 'fear', 'fearful', 'frightened', 'concerned'},
            'disgust': {'disgust', 'disgusted', 'disgusting', 'gross', 'nasty', 'revolting', 'repulsed', 
                       'sickened', 'vomit', 'yuck'},
            'surprise': {'surprise', 'surprised', 'surprising', 'astonished', 'wow', 'shocked', 
                        'unbelievable', 'unexpected', 'amazed', 'omg'},
        }
        
        self.emoji_map = {
            ':)': 'happy', '😊': 'happy', '❤️': 'happy',
            ':(': 'sad', '😢': 'sad', '😭': 'sad',
            '>:(': 'angry', '😠': 'angry', '😡': 'angry',
            ':o': 'surprise', '😮': 'surprise', '😲': 'surprise',
            '🤢': 'disgust', '🤮': 'disgust',
            '😨': 'fear', '😱': 'fear'
        }
        
        self.negations = {'not', 'no', 'never', 'dont', 'doesnt', 'didnt', 'cant', 'wont', 'isnt'}
        self.intensifiers = {'very': 1.5, 'extremely': 1.8, 'so': 1.4, 'really': 1.3, 'super': 1.5}
        self.opposite = {'happy': 'sad', 'sad': 'happy', 'angry': 'happy', 'fear': 'happy'}
    
    def _ensure_hf_model(self) -> bool:
        """Lazy load the Hugging Face model."""
        if not HF_AVAILABLE:
            return False
            
        if self._hf_pipe is not None:
            return True
            
        try:
            model_name = "michellejieli/emotion_text_classifier"
            device = 0 if torch.cuda.is_available() else -1
            self._hf_pipe = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
                device=device
            )
            logger.info(f"Loaded transformer model: {model_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            return False
    
    def _analyze_with_transformer(self, text: str) -> Optional[Tuple[str, float]]:
        """Analyze text using transformer model."""
        if not self._ensure_hf_model():
            return None
            
        try:
            outputs = self._hf_pipe(text[:512])
            scores = outputs[0]
            
            best_label = None
            best_score = -1.0
            
            for item in scores:
                label = item.get('label', '').lower()
                score = float(item.get('score', 0.0))
                
                if score > best_score:
                    best_score = score
                    best_label = label
            
            if best_label is None:
                return None
            
            if best_score < self.neutral_threshold:
                # No emotion scored highly — treat as neutral with inverse confidence
                return 'neutral', max(0.5, 1.0 - best_score)
            
            # Map model labels to standard emotions
            label_map = {
                'sadness': 'sad',
                'joy': 'happy',
                'love': 'happy',
                'anger': 'angry',
                'fear': 'fear',
                'surprise': 'surprise'
            }
            
            emotion = label_map.get(best_label, best_label)
            return emotion, max(0.0, min(1.0, best_score))
            
        except Exception as e:
            logger.error(f"Transformer analysis failed: {e}")
            return None
    
    def _analyze_with_lexicon(self, text: str) -> Tuple[str, float]:
        """Fallback lexicon-based analysis."""
        text_lower = text.lower()
        scores = {k: 0.0 for k in self.lexicon.keys()}
        
        # Check emojis
        for emoji, emotion in self.emoji_map.items():
            if emoji in text:
                scores[emotion] += 1.0
        
        # Tokenize and score
        tokens = re.findall(r'\w+', text_lower)
        
        for i, token in enumerate(tokens):
            # Find matching emotion
            base_emotion = None
            for emotion, words in self.lexicon.items():
                if token in words:
                    base_emotion = emotion
                    break
            
            if not base_emotion:
                continue
            
            weight = 1.0
            
            # Check for modifiers in window
            window = tokens[max(0, i-3):i]
            for w in window:
                if w in self.intensifiers:
                    weight *= self.intensifiers[w]
                if w in self.negations:
                    # Flip to opposite emotion
                    opp = self.opposite.get(base_emotion, 'neutral')
                    if opp != 'neutral':
                        scores[opp] += 0.8 * weight
                    weight *= 0.1
            
            scores[base_emotion] += weight
        
        # Calculate final emotion
        total = sum(scores.values())
        if total <= 0.1:
            return 'neutral', 0.5
        
        probs = {k: v/total for k, v in scores.items()}
        best_emotion = max(probs.items(), key=lambda x: x[1])
        
        # Ensure minimum confidence
        confidence = min(0.95, max(0.15, best_emotion[1]))
        return best_emotion[0], confidence
    
    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Analyze text for emotion.
        
        Args:
            text: Input text to analyze
            
        Returns:
            tuple: (emotion, confidence)
        """
        text = (text or '').strip()
        if not text:
            return 'neutral', 0.0
        
        # Try transformer first
        result = self._analyze_with_transformer(text)
        if result is not None:
            return result
        
        # Fallback to lexicon
        return self._analyze_with_lexicon(text)
