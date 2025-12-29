"""
Observa ML Analysis Service
FastAPI service for analyzing LLM traces using ML models
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Observa ML Analysis Service", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TraceAnalysisRequest(BaseModel):
    """Request model for trace analysis"""
    trace_id: str
    tenant_id: str
    project_id: str
    query: str
    context: Optional[str] = None
    response: str
    model: Optional[str] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    tokens_total: Optional[int] = None
    latency_ms: int


class AnalysisResult(BaseModel):
    """Analysis result model"""
    trace_id: str
    tenant_id: str
    project_id: str
    
    # Hallucination Detection
    is_hallucination: bool = False
    hallucination_confidence: Optional[float] = None
    hallucination_reasoning: Optional[str] = None
    
    # Quality Scores
    quality_score: Optional[int] = None
    coherence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    helpfulness_score: Optional[float] = None
    
    # Issue Flags
    has_context_drop: bool = False
    has_model_drift: bool = False
    has_prompt_injection: bool = False
    has_context_overflow: bool = False
    has_faithfulness_issue: bool = False
    has_cost_anomaly: bool = False
    has_latency_anomaly: bool = False
    has_quality_degradation: bool = False
    
    # Detailed Metrics
    context_relevance_score: Optional[float] = None
    answer_faithfulness_score: Optional[float] = None
    drift_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    
    # Metadata
    analysis_model: Optional[str] = None
    analysis_version: str = "0.1.0"
    processing_time_ms: Optional[int] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "observa-analysis"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_trace(request: TraceAnalysisRequest):
    """
    Analyze a trace for various issues:
    - Hallucinations
    - Context drop
    - Answer faithfulness
    - Model drift
    - Cost anomalies
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Analyzing trace {request.trace_id} for tenant {request.tenant_id}")
        
        # Import analysis services (lazy import to avoid loading models on startup)
        from services.hallucination_detector import HallucinationDetector
        from services.context_detector import ContextDetector
        from services.faithfulness_detector import FaithfulnessDetector
        from services.cost_analyzer import CostAnalyzer
        from services.drift_detector import DriftDetector
        
        result = AnalysisResult(
            trace_id=request.trace_id,
            tenant_id=request.tenant_id,
            project_id=request.project_id,
        )
        
        # 1. Hallucination Detection (if context is provided)
        if request.context:
            try:
                detector = HallucinationDetector()
                hallucination_result = detector.check(request.context, request.response)
                result.is_hallucination = hallucination_result["is_hallucination"]
                result.hallucination_confidence = hallucination_result["confidence_score"]
                result.hallucination_reasoning = hallucination_result["reasoning"]
                result.analysis_model = "deberta-v3-small"
            except Exception as e:
                logger.error(f"Hallucination detection failed: {e}")
        
        # 2. Context Drop Detection (if context is provided)
        if request.context and request.query:
            try:
                context_detector = ContextDetector()
                context_result = context_detector.check(request.query, request.context)
                result.has_context_drop = context_result["has_context_drop"]
                result.context_relevance_score = context_result["relevance_score"]
            except Exception as e:
                logger.error(f"Context drop detection failed: {e}")
        
        # 3. Answer Faithfulness Detection (if context is provided)
        if request.context:
            try:
                faithfulness_detector = FaithfulnessDetector()
                faithfulness_result = faithfulness_detector.check(request.context, request.response)
                result.has_faithfulness_issue = faithfulness_result["has_faithfulness_issue"]
                result.answer_faithfulness_score = faithfulness_result["faithfulness_score"]
            except Exception as e:
                logger.error(f"Faithfulness detection failed: {e}")
        
        # 4. Cost Anomaly Detection
        if request.tokens_total and request.model:
            try:
                cost_analyzer = CostAnalyzer()
                cost_result = cost_analyzer.analyze(
                    tokens_total=request.tokens_total,
                    model=request.model,
                    tenant_id=request.tenant_id
                )
                result.has_cost_anomaly = cost_result["has_anomaly"]
                result.anomaly_score = cost_result["anomaly_score"]
            except Exception as e:
                logger.error(f"Cost anomaly detection failed: {e}")
        
        # 5. Model Drift Detection
        if request.model:
            try:
                drift_detector = DriftDetector()
                drift_result = drift_detector.check(
                    tenant_id=request.tenant_id,
                    current_response=request.response,
                    model=request.model
                )
                result.has_model_drift = drift_result["has_model_drift"]
                result.drift_score = drift_result["drift_score"]
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        result.processing_time_ms = processing_time
        
        logger.info(f"Analysis completed for trace {request.trace_id} in {processing_time}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed for trace {request.trace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

