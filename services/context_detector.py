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
            
            # If response is provided, check if response uses the context
            if response:
                response_embedding = self.model.encode(response, convert_to_numpy=True)
                response_context_similarity = self._cosine_similarity(response_embedding, context_embedding)
                
                # Context drop detection logic:
                # 1. Response-Context similarity is most important (does response use context?)
                # 2. If response doesn't match context well (< 0.7), it's a context drop
                # 3. Also consider query-context similarity (should context have been retrieved?)
                
                # Primary indicator: response doesn't use context
                # Secondary: context might not be relevant to query
                # Use weighted average: 70% response-context, 30% query-context
                relevance_score = (response_context_similarity * 0.7) + (query_context_similarity * 0.3)
                
                # More aggressive threshold: < 0.7 means context drop
                # This catches cases where response doesn't properly use the provided context
                has_context_drop = response_context_similarity < 0.7 or relevance_score < 0.65
            else:
                relevance_score = query_context_similarity
                # Without response, use stricter threshold for query-context match
                has_context_drop = query_context_similarity < 0.6
            
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

