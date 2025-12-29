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
            # Validate inputs
            if not context or not answer:
                logger.warning("Empty context or answer provided for hallucination detection")
                return {
                    "is_hallucination": False,
                    "is_correct": False,
                    "confidence_score": 0.0,
                    "truth_score": 0.0,
                    "reasoning": "Empty context or answer provided"
                }
            
            # Tokenize context and answer together
            # For NLI, we pass them as a pair (premise, hypothesis)
            inputs = self.tokenizer(
                context,
                answer,
                return_tensors='pt',
                truncation=True,
                max_length=512,  # Limit to prevent OOM
                padding=True
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Handle both tensor and numpy outputs
                if hasattr(outputs.logits, 'numpy'):
                    logits = outputs.logits.numpy()[0]
                else:
                    logits = outputs.logits.detach().cpu().numpy()[0]
            
            # Convert logits to probabilities
            probabilities = self._softmax(logits)
            
            # Label Map for NLI models:
            # 0 = Contradiction (HALLUCINATION) - answer contradicts context
            # 1 = Entailment (TRUTH) - answer follows from context
            # 2 = Neutral (IRRELEVANT) - answer is unrelated to context
            
            contradiction_prob = float(probabilities[0])
            entailment_prob = float(probabilities[1])
            neutral_prob = float(probabilities[2])
            
            # Improved hallucination detection:
            # 1. Contradiction is highest probability, OR
            # 2. Contradiction is > 0.5 (high confidence), OR
            # 3. Contradiction is significantly higher than entailment (>0.2 difference)
            is_hallucination = (
                contradiction_prob > entailment_prob or
                contradiction_prob > 0.5 or
                (contradiction_prob - entailment_prob) > 0.2
            )
            is_correct = entailment_prob > 0.5 and entailment_prob > contradiction_prob
            
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

