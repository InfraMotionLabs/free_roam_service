"""Main FastAPI application for Free-Roam Inference Service"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.utils.logging import setup_logging, get_logger
from app.utils.monitoring import health_monitor
from app.api import routes
from app.services.stream_processor import StreamProcessor
from app.services.inference_engine import InferenceEngine
from app.core.state_manager import PromptStateManager
from app.models.example_vlm import ExampleVLM
from app.core.exceptions import FreeRoamException

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# Global instances (will be initialized in lifespan)
inference_engine: InferenceEngine = None
state_manager: PromptStateManager = None
stream_processor: StreamProcessor = None
model: ExampleVLM = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Free-Roam Inference Service...")
    
    try:
        # Initialize components
        global model, stream_processor, state_manager, inference_engine
        
        # Initialize model
        logger.info(f"Loading model from: {settings.model_path}")
        model = ExampleVLM()
        await model.initialize(
            model_path=settings.model_path or "placeholder",
            device=settings.device
        )
        health_monitor.set_model_loaded(True)
        logger.info("Model loaded successfully")
        
        # Initialize stream processor
        stream_processor = StreamProcessor()
        if settings.redis_host:
            try:
                await stream_processor.initialize_redis()
            except Exception as e:
                logger.warning(f"Redis initialization failed (will retry on use): {e}")
        logger.info("Stream processor initialized")
        
        # Initialize state manager
        state_manager = PromptStateManager()
        logger.info("State manager initialized")
        
        # Initialize inference engine
        inference_engine = InferenceEngine(
            model=model,
            stream_processor=stream_processor,
            state_manager=state_manager
        )
        logger.info("Inference engine initialized")
        
        # Override dependencies
        routes.get_inference_engine = lambda: inference_engine
        routes.get_state_manager = lambda: state_manager
        
        logger.info("Free-Roam Inference Service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Free-Roam Inference Service...")
    
    try:
        # Cancel all running jobs
        if inference_engine:
            jobs = await inference_engine.list_jobs()
            for job in jobs:
                if job["status"] in ["pending", "running"]:
                    await inference_engine.cancel_job(job["job_id"])
        
        # Close Redis connection
        if stream_processor:
            await stream_processor.close_redis()
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Free-Roam Inference Service for video stream analysis with VLM models",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = asyncio.get_event_loop().time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = asyncio.get_event_loop().time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    return response


# Error handlers
@app.exception_handler(FreeRoamException)
async def free_roam_exception_handler(request: Request, exc: FreeRoamException):
    """Handle Free-Roam exceptions"""
    logger.error(f"Free-Roam exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "detail": "An internal error occurred"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.log_level == "DEBUG" else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(routes.router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Free-Roam Inference Service",
        "version": settings.api_version,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False
    )

