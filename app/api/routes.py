"""FastAPI routes for the Free-Roam Inference Service"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from app.api.schemas import (
    InferenceStartRequest,
    InferenceStartResponse,
    InferenceStatusResponse,
    InferenceResultsResponse,
    PromptUpdateRequest,
    PromptUpdateResponse,
    PromptCurrentResponse,
    PromptHistoryResponse,
    HealthResponse,
    ReadinessResponse,
    LivenessResponse,
    ErrorResponse
)
from app.services.inference_engine import InferenceEngine
from app.core.state_manager import PromptStateManager
from app.utils.monitoring import health_monitor
from app.core.exceptions import (
    FreeRoamException,
    InferenceException,
    PromptException,
    StreamException
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Dependency to get inference engine (will be injected in main.py)
def get_inference_engine() -> InferenceEngine:
    """Dependency to get inference engine"""
    # This will be overridden in main.py with actual instance
    raise RuntimeError("Inference engine not initialized")


def get_state_manager() -> PromptStateManager:
    """Dependency to get state manager"""
    # This will be overridden in main.py with actual instance
    raise RuntimeError("State manager not initialized")


# Health Check Endpoints

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Overall health check endpoint"""
    health_status = health_monitor.get_health_status()
    return HealthResponse(**health_status)


@router.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
async def readiness_check():
    """Readiness check endpoint (for orchestration)"""
    readiness = health_monitor.get_readiness_status()
    return ReadinessResponse(**readiness)


@router.get("/health/live", response_model=LivenessResponse, tags=["health"])
async def liveness_check():
    """Liveness check endpoint"""
    liveness = health_monitor.get_liveness_status()
    return LivenessResponse(**liveness)


# Inference Endpoints

@router.post(
    "/api/v1/inference/start",
    response_model=InferenceStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["inference"]
)
async def start_inference(
    request: InferenceStartRequest,
    engine: InferenceEngine = Depends(get_inference_engine)
):
    """Start inference on a stream"""
    try:
        job_id = await engine.start_inference(
            stream_ref=request.stream_ref,
            prompt=request.prompt,
            fps=request.fps,
            job_id=request.job_id
        )
        
        return InferenceStartResponse(
            job_id=job_id,
            status="pending",
            message=f"Inference job {job_id} started"
        )
    except InferenceException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to start inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start inference: {str(e)}"
        )


@router.get(
    "/api/v1/inference/status/{job_id}",
    response_model=InferenceStatusResponse,
    tags=["inference"]
)
async def get_inference_status(
    job_id: str,
    engine: InferenceEngine = Depends(get_inference_engine)
):
    """Get inference job status"""
    job_status = await engine.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    return InferenceStatusResponse(**job_status)


@router.get(
    "/api/v1/inference/results/{job_id}",
    response_model=InferenceResultsResponse,
    tags=["inference"]
)
async def get_inference_results(
    job_id: str,
    limit: Optional[int] = None,
    engine: InferenceEngine = Depends(get_inference_engine)
):
    """Get inference results for a job"""
    job_status = await engine.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    results = await engine.get_job_results(job_id, limit=limit)
    
    if results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    return InferenceResultsResponse(
        job_id=job_id,
        status=job_status["status"],
        results=results,
        total_results=len(results)
    )


@router.delete(
    "/api/v1/inference/cancel/{job_id}",
    status_code=status.HTTP_200_OK,
    tags=["inference"]
)
async def cancel_inference(
    job_id: str,
    engine: InferenceEngine = Depends(get_inference_engine)
):
    """Cancel an inference job"""
    cancelled = await engine.cancel_job(job_id)
    
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or cannot be cancelled"
        )
    
    return {"message": f"Job {job_id} cancelled", "job_id": job_id}


@router.get(
    "/api/v1/inference/jobs",
    tags=["inference"]
)
async def list_jobs(
    status: Optional[str] = None,
    engine: InferenceEngine = Depends(get_inference_engine)
):
    """List all inference jobs"""
    from app.services.inference_engine import JobStatus
    
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}"
            )
    
    jobs = await engine.list_jobs(status=status_filter)
    return {"jobs": jobs, "total": len(jobs)}


# Prompt Management Endpoints

@router.post(
    "/api/v1/prompt/update",
    response_model=PromptUpdateResponse,
    tags=["prompt"]
)
async def update_prompt(
    request: PromptUpdateRequest,
    state_manager: PromptStateManager = Depends(get_state_manager)
):
    """Update the active prompt"""
    try:
        prompt_state = await state_manager.update_prompt(
            new_prompt=request.prompt,
            job_id=request.job_id,
            preserve_previous=request.preserve_previous
        )
        
        response_data = prompt_state.to_dict()
        if request.job_id:
            response_data["job_id"] = request.job_id
        
        return PromptUpdateResponse(**response_data)
    except PromptException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update prompt: {str(e)}"
        )


@router.get(
    "/api/v1/prompt/current",
    response_model=PromptCurrentResponse,
    tags=["prompt"]
)
async def get_current_prompt(
    job_id: Optional[str] = None,
    state_manager: PromptStateManager = Depends(get_state_manager)
):
    """Get the current prompt"""
    prompt_state = await state_manager.get_prompt_state(job_id=job_id)
    
    if not prompt_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prompt set"
        )
    
    response_data = prompt_state.to_dict()
    if job_id:
        response_data["job_id"] = job_id
    
    return PromptCurrentResponse(**response_data)


@router.get(
    "/api/v1/prompt/history",
    response_model=PromptHistoryResponse,
    tags=["prompt"]
)
async def get_prompt_history(
    limit: int = 10,
    job_id: Optional[str] = None,
    state_manager: PromptStateManager = Depends(get_state_manager)
):
    """Get prompt history"""
    history = await state_manager.get_prompt_history(job_id=job_id, limit=limit)
    
    return PromptHistoryResponse(history=history, total=len(history))


# Error Handler

@router.exception_handler(FreeRoamException)
async def free_roam_exception_handler(request, exc: FreeRoamException):
    """Handle Free-Roam exceptions"""
    logger.error(f"Free-Roam exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": str(exc), "type": type(exc).__name__}
    )

