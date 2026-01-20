"""Ray Serve application for Free-Roam Inference Service"""

import asyncio
import logging
from typing import Dict, Any, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse

from ray import serve
from ray.serve import Application

from app.config import settings
from app.utils.logging import setup_logging, get_logger
from app.utils.monitoring import health_monitor
from app.models.example_vlm import ExampleVLM
from app.services.stream_processor import StreamProcessor
from app.services.inference_service import InferenceService, JobStatus
from app.core.state_manager import PromptStateManager
from app.core.exceptions import (
    FreeRoamException,
    InferenceException,
    PromptException,
    ValidationException
)
from app.api.handlers import parse_json_body, create_error_response, create_success_response

# Setup logging
setup_logging()
logger = get_logger(__name__)


# Model Deployment
@serve.deployment(
    name="model",
    num_replicas=settings.num_replicas,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.5 if settings.device == "cuda" else 0}
)
class ModelDeployment:
    """Ray Serve deployment for VLM model inference"""
    
    def __init__(self):
        """Initialize model deployment"""
        self.model: Optional[ExampleVLM] = None
        self._initialized = False
        self._init_task = None
        logger.info(f"Initializing ModelDeployment with model_path: {settings.model_path}")
        self.model = ExampleVLM()
        # Schedule async initialization
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            self._init_task = asyncio.create_task(self._async_init())
        else:
            loop.run_until_complete(self._async_init())
    
    async def _async_init(self):
        """Async initialization"""
        await self.model.initialize(
            model_path=settings.model_path or "placeholder",
            device=settings.device
        )
        self._initialized = True
        health_monitor.set_model_loaded(True)
        logger.info("ModelDeployment initialized successfully")
    
    async def predict(self, frames: list, prompt: str) -> Dict[str, Any]:
        """Run inference on frames
        
        Args:
            frames: List of frames (will be converted from dict/list format)
            prompt: Text prompt
            
        Returns:
            Predictions dictionary
        """
        # Wait for initialization if still in progress
        if self._init_task and not self._init_task.done():
            await self._init_task
        
        if not self._initialized or not self.model:
            raise InferenceException("Model not initialized")
        
        # Convert frames if needed (Ray Serve may serialize/deserialize)
        import numpy as np
        if frames and isinstance(frames[0], (dict, list)):
            frames = [np.array(f) for f in frames]
        
        return await self.model.predict(frames, prompt)


# Inference Service Deployment (singleton for state management)
@serve.deployment(name="inference_service")
class InferenceServiceDeployment:
    """Ray Serve deployment for inference service (job tracking)"""
    
    def __init__(self):
        """Initialize inference service"""
        self.stream_processor = StreamProcessor()
        self.state_manager = PromptStateManager()
        self.inference_service = InferenceService(
            stream_processor=self.stream_processor,
            state_manager=self.state_manager
        )
        
        # Initialize Redis if configured
        if settings.redis_host:
            asyncio.create_task(self._init_redis())
        
        logger.info("InferenceServiceDeployment initialized")
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            await self.stream_processor.initialize_redis()
        except Exception as e:
            logger.warning(f"Redis initialization failed (will retry on use): {e}")
    
    async def start_inference(self, stream_ref: str, prompt: Optional[str] = None, 
                             fps: Optional[float] = None, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Start inference job"""
        try:
            job_id = await self.inference_service.start_inference(
                stream_ref=stream_ref,
                prompt=prompt,
                fps=fps,
                job_id=job_id
            )
            return {"job_id": job_id, "status": "pending", "message": f"Inference job {job_id} started"}
        except InferenceException as e:
            raise
        except Exception as e:
            logger.error(f"Failed to start inference: {e}", exc_info=True)
            raise InferenceException(f"Failed to start inference: {e}") from e
    
    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        status = await self.inference_service.get_job_status(job_id)
        if not status:
            raise InferenceException(f"Job {job_id} not found")
        return status
    
    async def get_results(self, job_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get job results"""
        status = await self.inference_service.get_job_status(job_id)
        if not status:
            raise InferenceException(f"Job {job_id} not found")
        
        results = await self.inference_service.get_job_results(job_id, limit=limit)
        return {
            "job_id": job_id,
            "status": status["status"],
            "results": results or [],
            "total_results": len(results) if results else 0
        }
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel job"""
        cancelled = await self.inference_service.cancel_job(job_id)
        if not cancelled:
            raise InferenceException(f"Job {job_id} not found or cannot be cancelled")
        return {"message": f"Job {job_id} cancelled", "job_id": job_id}
    
    async def list_jobs(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """List all jobs"""
        job_status = None
        if status_filter:
            try:
                job_status = JobStatus(status_filter.lower())
            except ValueError:
                raise ValidationException(f"Invalid status: {status_filter}")
        
        jobs = await self.inference_service.list_jobs(status=job_status)
        return {"jobs": jobs, "total": len(jobs)}
    
    async def update_prompt(self, new_prompt: str, job_id: Optional[str] = None, 
                           preserve_previous: bool = True) -> Dict[str, Any]:
        """Update prompt"""
        prompt_state = await self.state_manager.update_prompt(
            new_prompt=new_prompt,
            job_id=job_id,
            preserve_previous=preserve_previous
        )
        result = prompt_state.to_dict()
        if job_id:
            result["job_id"] = job_id
        return result
    
    async def get_current_prompt(self, job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get current prompt"""
        prompt_state = await self.state_manager.get_prompt_state(job_id=job_id)
        if not prompt_state:
            return None
        result = prompt_state.to_dict()
        if job_id:
            result["job_id"] = job_id
        return result
    
    async def get_prompt_history(self, job_id: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Get prompt history"""
        history = await self.state_manager.get_prompt_history(job_id=job_id, limit=limit)
        return {"history": history, "total": len(history)}


# API Deployment
@serve.deployment(route_prefix="/api/v1", name="api")
class APIDeployment:
    """Ray Serve deployment for API endpoints"""
    
    def __init__(self):
        """Initialize API deployment"""
        self._inference_service_handle = None
        self._model_handle = None
        logger.info("APIDeployment initialized")
    
    def _get_inference_service_handle(self):
        """Get inference service handle (lazy initialization)"""
        if self._inference_service_handle is None:
            self._inference_service_handle = serve.get_deployment("inference_service").get_handle()
        return self._inference_service_handle
    
    def _get_model_handle(self):
        """Get model handle (lazy initialization)"""
        if self._model_handle is None:
            self._model_handle = serve.get_deployment("model").get_handle()
        return self._model_handle
    
    async def __call__(self, request: Request):
        """Handle HTTP requests"""
        path = request.url.path
        method = request.method
        
        try:
            # Inference endpoints
            if path.startswith("/api/v1/inference"):
                return await self._handle_inference(request, path, method)
            
            # Prompt endpoints
            elif path.startswith("/api/v1/prompt"):
                return await self._handle_prompt(request, path, method)
            
            else:
                return create_error_response("Not found", f"Unknown endpoint: {path}", 404)
                
        except (InferenceException, PromptException, ValidationException) as e:
            return create_error_response(str(e), status_code=400)
        except FreeRoamException as e:
            logger.error(f"Free-Roam exception: {e}", exc_info=True)
            return create_error_response(str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return create_error_response("Internal server error", str(e) if settings.log_level == "DEBUG" else None, 500)
    
    async def _handle_inference(self, request: Request, path: str, method: str) -> JSONResponse:
        """Handle inference endpoints"""
        handle = self._get_inference_service_handle()
        
        if method == "POST" and path.endswith("/start"):
            data = await parse_json_body(request)
            result = await handle.start_inference.remote(
                stream_ref=data.get("stream_ref"),
                prompt=data.get("prompt"),
                fps=data.get("fps"),
                job_id=data.get("job_id")
            )
            return create_success_response(result, 202)
        
        elif method == "GET" and "/status/" in path:
            job_id = path.split("/status/")[-1]
            status = await handle.get_status.remote(job_id)
            return create_success_response(status)
        
        elif method == "GET" and "/results/" in path:
            job_id = path.split("/results/")[-1]
            limit = request.query_params.get("limit")
            limit = int(limit) if limit else None
            results = await handle.get_results.remote(job_id, limit=limit)
            return create_success_response(results)
        
        elif method == "DELETE" and "/cancel/" in path:
            job_id = path.split("/cancel/")[-1]
            result = await handle.cancel_job.remote(job_id)
            return create_success_response(result)
        
        elif method == "GET" and path.endswith("/jobs"):
            status_filter = request.query_params.get("status")
            result = await handle.list_jobs.remote(status_filter=status_filter)
            return create_success_response(result)
        
        else:
            return create_error_response("Not found", f"Unknown inference endpoint: {path}", 404)
    
    async def _handle_prompt(self, request: Request, path: str, method: str) -> JSONResponse:
        """Handle prompt endpoints"""
        handle = self._get_inference_service_handle()
        
        if method == "POST" and path.endswith("/update"):
            data = await parse_json_body(request)
            result = await handle.update_prompt.remote(
                new_prompt=data.get("prompt"),
                job_id=data.get("job_id"),
                preserve_previous=data.get("preserve_previous", True)
            )
            return create_success_response(result)
        
        elif method == "GET" and path.endswith("/current"):
            job_id = request.query_params.get("job_id")
            result = await handle.get_current_prompt.remote(job_id=job_id)
            if not result:
                return create_error_response("No prompt set", status_code=404)
            return create_success_response(result)
        
        elif method == "GET" and path.endswith("/history"):
            limit = request.query_params.get("limit", "10")
            limit = int(limit) if limit.isdigit() else 10
            job_id = request.query_params.get("job_id")
            result = await handle.get_prompt_history.remote(job_id=job_id, limit=limit)
            return create_success_response(result)
        
        else:
            return create_error_response("Not found", f"Unknown prompt endpoint: {path}", 404)


# Health Deployment
@serve.deployment(route_prefix="/health", name="health")
class HealthDeployment:
    """Ray Serve deployment for health checks"""
    
    async def __call__(self, request: Request):
        """Handle health check requests"""
        path = request.url.path
        
        if path == "/health" or path == "/health/":
            status = health_monitor.get_health_status()
            return create_success_response(status)
        
        elif path == "/health/ready" or path.endswith("/ready"):
            status = health_monitor.get_readiness_status()
            ready = status.get("ready", False)
            return create_success_response(status, 200 if ready else 503)
        
        elif path == "/health/live" or path.endswith("/live"):
            status = health_monitor.get_liveness_status()
            return create_success_response(status)
        
        else:
            return create_error_response("Not found", f"Unknown health endpoint: {path}", 404)


# Root endpoint
@serve.deployment(route_prefix="/", name="root")
class RootDeployment:
    """Root endpoint deployment"""
    
    async def __call__(self, request: Request):
        """Handle root requests"""
        return create_success_response({
            "service": "Free-Roam Inference Service (Ray Serve)",
            "version": settings.api_version,
            "status": "running",
            "docs": "See /health for health checks"
        })


# Build the application
def build_app() -> Application:
    """Build Ray Serve application"""
    # Create deployments (order matters - dependencies first)
    model_deployment = ModelDeployment.bind()
    inference_service_deployment = InferenceServiceDeployment.bind()
    api_deployment = APIDeployment.bind()
    health_deployment = HealthDeployment.bind()
    root_deployment = RootDeployment.bind()
    
    # Return main API deployment (Ray Serve will handle routing)
    # Health and root are separate deployments with their own routes
    return api_deployment


# For `serve run` command
app = build_app()

