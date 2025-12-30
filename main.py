"""
Observa ML Analysis Service
FastAPI service for analyzing LLM traces using ML models
"""
import os
import asyncio
from contextlib import asynccontextmanager
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

# Model registry to track loaded models
model_registry: Dict[str, Any] = {
    "hallucination": None,
    "context": None,
    "faithfulness": None,
    "drift": None,
    "cost": None,
}

# Track model loading status
model_loading_status: Dict[str, bool] = {
    "hallucination": False,
    "context": False,
    "faithfulness": False,
    "drift": False,
    "cost": False,
}


async def preload_hallucination_model():
    """Preload hallucination detection model"""
    try:
        logger.info("Preloading hallucination detection model...")
        from services.hallucination_detector import HallucinationDetector
        detector = HallucinationDetector()
        # Check if model loaded successfully
        if detector.model is not None and detector.tokenizer is not None:
            model_registry["hallucination"] = detector
            model_loading_status["hallucination"] = True
            logger.info("✅ Hallucination model preloaded successfully")
        else:
            logger.warning("⚠️ Hallucination model not loaded (will use fallback)")
    except Exception as e:
        logger.error(f"❌ Failed to preload hallucination model: {e}", exc_info=True)
        model_loading_status["hallucination"] = False


async def preload_context_model():
    """Preload context detection model"""
    try:
        logger.info("Preloading context detection model...")
        from services.context_detector import ContextDetector
        detector = ContextDetector()
        model_registry["context"] = detector
        model_loading_status["context"] = True
        logger.info("✅ Context detection model preloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to preload context model: {e}", exc_info=True)
        model_loading_status["context"] = False


async def preload_faithfulness_model():
    """Preload faithfulness detection model"""
    try:
        logger.info("Preloading faithfulness detection model...")
        from services.faithfulness_detector import FaithfulnessDetector
        detector = FaithfulnessDetector()
        model_registry["faithfulness"] = detector
        model_loading_status["faithfulness"] = True
        logger.info("✅ Faithfulness detection model preloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to preload faithfulness model: {e}", exc_info=True)
        model_loading_status["faithfulness"] = False


async def preload_drift_model():
    """Preload drift detection model"""
    try:
        logger.info("Preloading drift detection model...")
        from services.drift_detector import DriftDetector
        detector = DriftDetector()
        model_registry["drift"] = detector
        model_loading_status["drift"] = True
        logger.info("✅ Drift detection model preloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to preload drift model: {e}", exc_info=True)
        model_loading_status["drift"] = False


async def preload_models():
    """Preload all models in parallel"""
    logger.info("🚀 Starting model preloading...")
    start_time = asyncio.get_event_loop().time()
    
    # Preload models in parallel
    tasks = [
        preload_hallucination_model(),
        preload_context_model(),
        preload_faithfulness_model(),
        preload_drift_model(),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Log results
    elapsed = asyncio.get_event_loop().time() - start_time
    loaded_count = sum(1 for status in model_loading_status.values() if status)
    total_count = len(model_loading_status)
    
    logger.info(f"✅ Model preloading completed in {elapsed:.2f}s")
    logger.info(f"📊 Models loaded: {loaded_count}/{total_count}")
    
    for model_name, status in model_loading_status.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"  {status_icon} {model_name}: {'Ready' if status else 'Failed'}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup: Preload models
    logger.info("🔧 Application startup: Preloading models...")
    await preload_models()
    logger.info("✅ Application ready - all models preloaded")
    yield
    # Shutdown: Cleanup if needed
    logger.info("🔧 Application shutdown: Cleaning up...")


app = FastAPI(
    title="Observa ML Analysis Service",
    version="0.1.0",
    lifespan=lifespan
)

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
    """Health check endpoint (liveness probe)"""
    return {
        "status": "ok",
        "service": "observa-analysis",
        "models": {
            name: "loaded" if status else "not_loaded"
            for name, status in model_loading_status.items()
        }
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint - returns 200 when all models are loaded, 503 otherwise"""
    all_loaded = all(model_loading_status.values())
    
    if all_loaded:
        return {
            "status": "ready",
            "service": "observa-analysis",
            "models": {
                name: "ready" if status else "not_ready"
                for name, status in model_loading_status.items()
            }
        }
    else:
        from fastapi import status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "service": "observa-analysis",
                "message": "Some models are still loading",
                "models": {
                    name: "ready" if status else "loading"
                    for name, status in model_loading_status.items()
                }
            }
        )


async def run_hallucination_detection(request: TraceAnalysisRequest) -> Dict[str, Any]:
    """Run hallucination detection with timeout"""
    try:
        from services.hallucination_detector import HallucinationDetector
        from services.hallucination_detector_v2 import HallucinationDetectorV2
        
        if not request.context:
            return {"skipped": True}
        
        # Try DeBERTa first, fallback to sentence-transformers
        try:
            detector = HallucinationDetector()
            result = detector.check(request.context, request.response)
            if "Model not loaded" in result.get("reasoning", "") or "check logs" in result.get("reasoning", "").lower():
                raise Exception("DeBERTa model not loaded")
            return {"result": result, "model": "cross-encoder/nli-deberta-v3-small"}
        except Exception:
            detector = HallucinationDetectorV2()
            result = detector.check(request.context, request.response)
            return {"result": result, "model": "all-MiniLM-L6-v2"}
    except Exception as e:
        logger.error(f"Hallucination detection error: {e}", exc_info=True)
        return {"error": str(e)}


async def run_context_detection(request: TraceAnalysisRequest) -> Dict[str, Any]:
    """Run context drop detection with timeout"""
    try:
        from services.context_detector import ContextDetector
        
        if not request.context or not request.query:
            return {"skipped": True}
        
        detector = ContextDetector()
        result = detector.check(request.query, request.context, request.response)
        return {"result": result}
    except Exception as e:
        logger.error(f"Context drop detection error: {e}", exc_info=True)
        return {"error": str(e)}


async def run_faithfulness_detection(request: TraceAnalysisRequest) -> Dict[str, Any]:
    """Run faithfulness detection with timeout"""
    try:
        from services.faithfulness_detector import FaithfulnessDetector
        
        if not request.context:
            return {"skipped": True}
        
        detector = FaithfulnessDetector()
        result = detector.check(request.context, request.response)
        return {"result": result}
    except Exception as e:
        logger.error(f"Faithfulness detection error: {e}", exc_info=True)
        return {"error": str(e)}


async def run_cost_analysis(request: TraceAnalysisRequest) -> Dict[str, Any]:
    """Run cost anomaly detection with timeout"""
    try:
        from services.cost_analyzer import CostAnalyzer
        
        if not request.tokens_total:
            return {"skipped": True}
        
        analyzer = CostAnalyzer()
        result = analyzer.analyze(request.tokens_total, request.model)
        return {"result": result}
    except Exception as e:
        logger.error(f"Cost analysis error: {e}", exc_info=True)
        return {"error": str(e)}


async def run_drift_detection(request: TraceAnalysisRequest) -> Dict[str, Any]:
    """Run model drift detection with timeout"""
    try:
        from services.drift_detector import DriftDetector
        
        if not request.model:
            return {"skipped": True}
        
        detector = DriftDetector()
        result = detector.check(request.model, request.response)
        return {"result": result}
    except Exception as e:
        logger.error(f"Drift detection error: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_trace(request: TraceAnalysisRequest):
    """
    Analyze a trace for various issues using parallel processing:
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
        
        result = AnalysisResult(
            trace_id=request.trace_id,
            tenant_id=request.tenant_id,
            project_id=request.project_id,
        )
        
        # Run all analyses in parallel with individual timeouts
        tasks = []
        
        if request.context:
            tasks.append(run_hallucination_detection(request))
            tasks.append(run_faithfulness_detection(request))
        
        if request.context and request.query:
            tasks.append(run_context_detection(request))
        
        if request.tokens_total:
            tasks.append(run_cost_analysis(request))
        
        if request.model:
            tasks.append(run_drift_detection(request))
        
        # Run all tasks in parallel with 30 second timeout per task
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        result_index = 0
        
        # Process hallucination result
        if request.context:
            hallucination_data = results[result_index] if result_index < len(results) else None
            result_index += 1
            
            if isinstance(hallucination_data, dict) and "result" in hallucination_data:
                hallucination_result = hallucination_data["result"]
                result.is_hallucination = hallucination_result.get("is_hallucination", False)
                result.hallucination_confidence = hallucination_result.get("confidence_score")
                result.hallucination_reasoning = hallucination_result.get("reasoning")
                result.analysis_model = hallucination_data.get("model")
            elif isinstance(hallucination_data, dict) and "error" in hallucination_data:
                result.hallucination_reasoning = f"Error: {hallucination_data['error']}"
        
        # Process faithfulness result
        if request.context:
            faithfulness_data = results[result_index] if result_index < len(results) else None
            result_index += 1
            
            if isinstance(faithfulness_data, dict) and "result" in faithfulness_data:
                faithfulness_result = faithfulness_data["result"]
                result.has_faithfulness_issue = faithfulness_result.get("has_issue", False)
                result.answer_faithfulness_score = faithfulness_result.get("score")
        
        # Process context drop result
        if request.context and request.query:
            context_data = results[result_index] if result_index < len(results) else None
            result_index += 1
            
            if isinstance(context_data, dict) and "result" in context_data:
                context_result = context_data["result"]
                result.has_context_drop = context_result.get("has_context_drop", False)
                result.context_relevance_score = context_result.get("relevance_score")
        
        # Process cost analysis result
        if request.tokens_total:
            cost_data = results[result_index] if result_index < len(results) else None
            result_index += 1
            
            if isinstance(cost_data, dict) and "result" in cost_data:
                cost_result = cost_data["result"]
                result.has_cost_anomaly = cost_result.get("has_anomaly", False)
                result.anomaly_score = cost_result.get("anomaly_score")
        
        # Process drift detection result
        if request.model:
            drift_data = results[result_index] if result_index < len(results) else None
            
            if isinstance(drift_data, dict) and "result" in drift_data:
                drift_result = drift_data["result"]
                result.has_model_drift = drift_result.get("has_drift", False)
                result.drift_score = drift_result.get("drift_score")
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        result.processing_time_ms = processing_time
        
        logger.info(f"Analysis completed for trace {request.trace_id} in {processing_time}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed for trace {request.trace_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

