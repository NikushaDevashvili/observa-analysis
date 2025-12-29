"""
Cost Anomaly Detection Service
Detects unusual cost spikes in token usage
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# Model pricing (per 1K tokens)
MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
}


class CostAnalyzer:
    """Analyzes token usage and detects cost anomalies"""
    
    def __init__(self):
        """Initialize database connection"""
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            logger.warning("DATABASE_URL not set, cost analysis will be limited")
    
    def _get_db_connection(self):
        """Get database connection"""
        if not self.db_url:
            return None
        return psycopg2.connect(self.db_url)
    
    def calculate_cost(self, tokens_prompt: int, tokens_completion: int, model: str) -> float:
        """
        Calculate cost for a trace
        
        Args:
            tokens_prompt: Number of input tokens
            tokens_completion: Number of output tokens
            model: Model name
            
        Returns:
            Cost in USD
        """
        # Normalize model name (handle variations)
        model_key = model.lower()
        for key in MODEL_PRICING:
            if key in model_key:
                pricing = MODEL_PRICING[key]
                input_cost = (tokens_prompt / 1000) * pricing["input"]
                output_cost = (tokens_completion / 1000) * pricing["output"]
                return input_cost + output_cost
        
        # Default pricing if model not found (use GPT-3.5-turbo as baseline)
        default_pricing = MODEL_PRICING["gpt-3.5-turbo"]
        input_cost = (tokens_prompt / 1000) * default_pricing["input"]
        output_cost = (tokens_completion / 1000) * default_pricing["output"]
        return input_cost + output_cost
    
    def analyze(self, tokens_total: int, model: str, tenant_id: str) -> Dict[str, Any]:
        """
        Analyze cost and detect anomalies
        
        Args:
            tokens_total: Total tokens used
            model: Model name
            tenant_id: Tenant ID for historical comparison
            
        Returns:
            dict with has_anomaly, anomaly_score, cost
        """
        try:
            # Estimate tokens (split 70/30 if not provided separately)
            tokens_prompt = int(tokens_total * 0.7)
            tokens_completion = int(tokens_total * 0.3)
            
            # Calculate cost
            cost = self.calculate_cost(tokens_prompt, tokens_completion, model)
            
            # Get historical data for comparison
            if self.db_url:
                anomaly_score = self._detect_anomaly(tenant_id, tokens_total, cost)
                has_anomaly = anomaly_score > 2.0  # More than 2 standard deviations
            else:
                # Without DB, can't do statistical analysis
                anomaly_score = 0.0
                has_anomaly = False
            
            return {
                "has_anomaly": has_anomaly,
                "anomaly_score": float(anomaly_score),
                "cost": float(cost),
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion
            }
        except Exception as e:
            logger.error(f"Cost analysis error: {e}")
            return {
                "has_anomaly": False,
                "anomaly_score": 0.0,
                "cost": 0.0
            }
    
    def _detect_anomaly(self, tenant_id: str, tokens_total: int, cost: float) -> float:
        """
        Detect if current usage is anomalous compared to historical data
        
        Returns z-score (number of standard deviations from mean)
        """
        try:
            conn = self._get_db_connection()
            if not conn:
                return 0.0
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get historical token usage for this tenant (last 30 days)
            cursor.execute("""
                SELECT tokens_total, tokens_prompt, tokens_completion
                FROM traces
                WHERE tenant_id = %s
                AND timestamp > NOW() - INTERVAL '30 days'
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (tenant_id,))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if len(rows) < 10:  # Need at least 10 data points
                return 0.0
            
            # Calculate statistics
            historical_tokens = [row["tokens_total"] for row in rows if row["tokens_total"]]
            
            if not historical_tokens:
                return 0.0
            
            mean = np.mean(historical_tokens)
            std = np.std(historical_tokens)
            
            if std == 0:
                return 0.0
            
            # Calculate z-score
            z_score = abs((tokens_total - mean) / std)
            
            return float(z_score)
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return 0.0

