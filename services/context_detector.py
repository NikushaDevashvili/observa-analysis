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
    
    def check(self, query: str, context: str, response: str = None) -> dict:
        """
        Check if context is relevant to the query AND if response uses the context
        
        Args:
            query: The user's query
            context: The retrieved context to check
            response: The model's response (optional, for better detection)
            
        Returns:
            dict with has_context_drop, relevance_score
        """
        try:
            # Generate embeddings
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            context_embedding = self.model.encode(context, convert_to_numpy=True)
            
            # Calculate cosine similarity between query and context
            query_context_similarity = self._cosine_similarity(query_embedding, context_embedding)
            
            # If response is provided, also check if response uses the context
            if response:
                response_embedding = self.model.encode(response, convert_to_numpy=True)
                response_context_similarity = self._cosine_similarity(response_embedding, context_embedding)
                
                # Context drop if:
                # 1. Query and context are not similar (context doesn't match query)
                # 2. Response and context are not similar (response doesn't use context)
                # Use the lower of the two similarities as the relevance score
                relevance_score = min(query_context_similarity, response_context_similarity)
            else:
                relevance_score = query_context_similarity
            
            # Threshold: < 0.6 means potential context drop
            # Lower threshold because we want to catch cases where context was provided but not used
            has_context_drop = relevance_score < 0.6
            
            return {
                "has_context_drop": bool(has_context_drop),
                "relevance_score": float(relevance_score),
                "threshold": 0.6
            }
        except Exception as e:
            logger.error(f"Context detection error: {e}")
            return {
                "has_context_drop": False,
                "relevance_score": 0.0,
                "threshold": 0.6
            }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

