"""
Hallucination Detection Service (Alternative Implementation)
Uses sentence-transformers for NLI to avoid tokenizer compatibility issues
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HallucinationDetectorV2:
    """Detects hallucinations using sentence-transformers semantic similarity"""
    
    _instance = None
    _model = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(HallucinationDetectorV2, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model (only once)"""
        if not HallucinationDetectorV2._initialized:
            logger.info("Loading hallucination detection model (sentence-transformers)...")
            try:
                # Use a lighter model to avoid OOM issues
                # all-MiniLM-L6-v2 is smaller (~80MB) and faster, already used by context detector
                # This reuses the same model instance if available, saving memory
                try:
                    # Try to reuse context detector's model if it exists
                    from services.context_detector import ContextDetector
                    context_detector = ContextDetector()
                    if hasattr(context_detector, 'model'):
                        HallucinationDetectorV2._model = context_detector.model
                        logger.info("Reusing context detector model for hallucination detection (memory efficient)")
                    else:
                        raise AttributeError("Context detector model not available")
                except Exception:
                    # Fallback: load our own instance
                    HallucinationDetectorV2._model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Loaded all-MiniLM-L6-v2 model for hallucination detection")
                
                HallucinationDetectorV2._initialized = True
                logger.info("Hallucination detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}", exc_info=True)
                raise
        
        self.model = HallucinationDetectorV2._model
    
    def check(self, context: str, answer: str) -> dict:
        """
        Check if answer is a hallucination based on context
        Uses semantic similarity: if answer contradicts context, similarity will be low
        
        Args:
            context: The context/ground truth information
            answer: The model's response to check
            
        Returns:
            dict with is_hallucination, confidence_score, truth_score, reasoning
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {
                "is_hallucination": False,
                "is_correct": False,
                "confidence_score": 0.0,
                "truth_score": 0.0,
                "reasoning": "Model not loaded"
            }
        
        try:
            # Validate inputs
            if not context or not answer:
                logger.warning("Empty context or answer")
                return {
                    "is_hallucination": False,
                    "is_correct": False,
                    "confidence_score": 0.0,
                    "truth_score": 0.0,
                    "reasoning": "Empty context or answer"
                }
            
            # Generate embeddings
            context_embedding = self.model.encode(context, convert_to_numpy=True)
            answer_embedding = self.model.encode(answer, convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarity = float(np.dot(context_embedding, answer_embedding) / 
                             (np.linalg.norm(context_embedding) * np.linalg.norm(answer_embedding)))
            
            # For hallucination detection:
            # Low similarity (< 0.5) suggests contradiction/hallucination
            # High similarity (> 0.7) suggests answer follows from context
            # Medium similarity (0.5-0.7) is ambiguous
            
            # Hallucination if similarity is very low (contradiction)
            is_hallucination = similarity < 0.5
            is_correct = similarity > 0.7
            
            # Confidence: inverse of similarity (lower similarity = higher hallucination confidence)
            hallucination_confidence = 1.0 - similarity
            truth_confidence = similarity
            
            return {
                "is_hallucination": bool(is_hallucination),
                "is_correct": bool(is_correct),
                "confidence_score": hallucination_confidence,
                "truth_score": truth_confidence,
                "reasoning": f"Semantic similarity: {similarity:.2%}. Low similarity (<50%) indicates potential contradiction/hallucination."
            }
        except Exception as e:
            logger.error(f"Hallucination detection error: {e}", exc_info=True)
            return {
                "is_hallucination": False,
                "is_correct": False,
                "confidence_score": 0.0,
                "truth_score": 0.0,
                "reasoning": f"Error: {str(e)}"
            }

