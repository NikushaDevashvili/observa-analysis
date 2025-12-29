"""
Model Drift Detection Service
Detects when model behavior changes over time
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects model drift by comparing response patterns over time"""
    
    def __init__(self):
        """Initialize the sentence transformer model"""
        logger.info("Loading drift detection model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_url = os.getenv("DATABASE_URL")
        logger.info("Drift detection model loaded")
    
    def _get_db_connection(self):
        """Get database connection"""
        if not self.db_url:
            return None
        return psycopg2.connect(self.db_url)
    
    def check(self, tenant_id: str, current_response: str, model: str) -> Dict[str, Any]:
        """
        Check if current response shows drift compared to baseline
        
        Args:
            tenant_id: Tenant ID for historical comparison
            current_response: Current response to check
            model: Model name
            
        Returns:
            dict with has_model_drift, drift_score
        """
        try:
            if not self.db_url:
                return {
                    "has_model_drift": False,
                    "drift_score": 0.0
                }
            
            # Get baseline responses (first month of data)
            baseline_responses = self._get_baseline_responses(tenant_id, model)
            
            if len(baseline_responses) < 10:  # Need at least 10 baseline samples
                return {
                    "has_model_drift": False,
                    "drift_score": 0.0,
                    "reason": "Insufficient baseline data"
                }
            
            # Calculate baseline embedding
            baseline_embeddings = self.model.encode(baseline_responses, convert_to_numpy=True)
            baseline_mean = np.mean(baseline_embeddings, axis=0)
            
            # Calculate current response embedding
            current_embedding = self.model.encode(current_response, convert_to_numpy=True)
            
            # Calculate similarity to baseline
            similarity = self._cosine_similarity(current_embedding, baseline_mean)
            
            # Drift if similarity is low (< 0.7)
            drift_score = 1.0 - similarity
            has_model_drift = similarity < 0.7
            
            return {
                "has_model_drift": bool(has_model_drift),
                "drift_score": float(drift_score),
                "similarity": float(similarity),
                "threshold": 0.7
            }
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            return {
                "has_model_drift": False,
                "drift_score": 0.0
            }
    
    def _get_baseline_responses(self, tenant_id: str, model: str) -> List[str]:
        """Get baseline responses from first month of data"""
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get responses from first 30 days (baseline period)
            # This assumes we have a traces table - for now, we'll use analysis_results
            # In production, you'd query the actual traces table
            cursor.execute("""
                SELECT response
                FROM traces
                WHERE tenant_id = %s
                AND model = %s
                AND timestamp >= (
                    SELECT MIN(timestamp) FROM traces WHERE tenant_id = %s
                )
                AND timestamp <= (
                    SELECT MIN(timestamp) FROM traces WHERE tenant_id = %s
                ) + INTERVAL '30 days'
                ORDER BY timestamp
                LIMIT 100
            """, (tenant_id, model, tenant_id, tenant_id))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [row["response"] for row in rows if row["response"]]
        except Exception as e:
            logger.error(f"Error getting baseline responses: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

