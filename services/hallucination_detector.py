"""
Hallucination Detection Service
Uses DeBERTa-v3-small NLI model to detect hallucinations
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detects hallucinations by comparing response to context using NLI"""
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        """Singleton pattern to avoid reloading model multiple times"""
        if cls._instance is None:
            cls._instance = super(HallucinationDetector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the DeBERTa model (only once due to singleton)"""
        if self._model is None:
            logger.info("Loading hallucination detection model...")
            model_name = 'cross-encoder/nli-deberta-v3-small'
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._model.eval()  # Set to evaluation mode
                logger.info("Hallucination detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load hallucination model: {e}", exc_info=True)
                raise
        
        # Use class-level model and tokenizer
        self.tokenizer = self._tokenizer
        self.model = self._model
    
    def check(self, context: str, answer: str) -> dict:
        """
        Check if answer is a hallucination based on context
        
        Args:
            context: The context/ground truth information
            answer: The model's response to check
            
        Returns:
            dict with is_hallucination, confidence_score, truth_score, reasoning
        """
        try:
            # Tokenize context and answer together
            inputs = self.tokenizer(
                context,
                answer,
                return_tensors='pt',
                truncation=True,
                max_length=512  # Limit to prevent OOM
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.numpy()[0]
            
            # Convert logits to probabilities
            probabilities = self._softmax(logits)
            
            # Label Map:
            # 0 = Contradiction (HALLUCINATION)
            # 1 = Entailment (TRUTH)
            # 2 = Neutral (IRRELEVANT)
            
            contradiction_prob = float(probabilities[0])
            entailment_prob = float(probabilities[1])
            neutral_prob = float(probabilities[2])
            
            # Hallucination if contradiction probability is highest
            is_hallucination = contradiction_prob > entailment_prob
            is_correct = entailment_prob > contradiction_prob and entailment_prob > neutral_prob
            
            return {
                "is_hallucination": bool(is_hallucination),
                "is_correct": bool(is_correct),
                "confidence_score": contradiction_prob,
                "truth_score": entailment_prob,
                "reasoning": f"Contradiction: {contradiction_prob:.2%}, Entailment: {entailment_prob:.2%}, Neutral: {neutral_prob:.2%}"
            }
        except Exception as e:
            logger.error(f"Hallucination detection error: {e}")
            # Return safe default
            return {
                "is_hallucination": False,
                "is_correct": False,
                "confidence_score": 0.0,
                "truth_score": 0.0,
                "reasoning": f"Error during detection: {str(e)}"
            }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

