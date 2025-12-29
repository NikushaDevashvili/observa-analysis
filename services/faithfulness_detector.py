"""
Answer Faithfulness Detection Service
Detects when answer doesn't match the provided context
"""
from services.hallucination_detector import HallucinationDetector
import logging

logger = logging.getLogger(__name__)


class FaithfulnessDetector:
    """Detects if answer is faithful to the provided context"""
    
    def __init__(self):
        """Initialize using hallucination detector (same NLI model)"""
        logger.info("Loading faithfulness detection model...")
        self.detector = HallucinationDetector()
        logger.info("Faithfulness detection model loaded")
    
    def check(self, context: str, answer: str) -> dict:
        """
        Check if answer is faithful to the context
        
        Args:
            context: The provided context
            answer: The model's answer
            
        Returns:
            dict with has_faithfulness_issue, faithfulness_score
        """
        try:
            # Use NLI to check if answer entails context
            # For faithfulness, we want high entailment (answer should follow from context)
            result = self.detector.check(context, answer)
            
            # Faithfulness issue if answer doesn't entail context well
            # Low entailment score means answer doesn't match context
            faithfulness_score = result["truth_score"]  # Entailment probability
            has_faithfulness_issue = faithfulness_score < 0.5
            
            return {
                "has_faithfulness_issue": bool(has_faithfulness_issue),
                "faithfulness_score": faithfulness_score,
                "threshold": 0.5
            }
        except Exception as e:
            logger.error(f"Faithfulness detection error: {e}")
            return {
                "has_faithfulness_issue": False,
                "faithfulness_score": 0.0,
                "threshold": 0.5
            }

