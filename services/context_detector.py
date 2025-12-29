"""
Context Drop Detection Service
Detects when retrieved context doesn't match the query
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ContextDetector:
    """Detects context drop by measuring semantic similarity between query and context"""
    
    def __init__(self):
        """Initialize the sentence transformer model"""
        logger.info("Loading context detection model...")
        # Use a lightweight model for fast inference
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Context detection model loaded")
    
    def check(self, query: str, context: str) -> dict:
        """
        Check if context is relevant to the query
        
        Args:
            query: The user's query
            context: The retrieved context to check
            
        Returns:
            dict with has_context_drop, relevance_score
        """
        try:
            # Generate embeddings
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            context_embedding = self.model.encode(context, convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, context_embedding)
            
            # Threshold: < 0.5 means context is likely irrelevant
            has_context_drop = similarity < 0.5
            
            return {
                "has_context_drop": bool(has_context_drop),
                "relevance_score": float(similarity),
                "threshold": 0.5
            }
        except Exception as e:
            logger.error(f"Context detection error: {e}")
            return {
                "has_context_drop": False,
                "relevance_score": 0.0,
                "threshold": 0.5
            }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

