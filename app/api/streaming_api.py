"""FastAPI streaming service for real-time inference results"""

import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.config import settings
from app.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Free-Roam Inference Streaming API",
    version=settings.api_version,
    description="Real-time inference results streaming service for frontend"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if hasattr(settings, 'cors_origins') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RayServe backend URL
RAY_SERVE_URL = settings.ray_serve_url


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "streaming_api"}


@app.get("/api/v1/inference/results/stream/{job_id}")
async def stream_results(
    job_id: str,
    limit: Optional[int] = Query(None, description="Maximum number of results to return"),
    last_frame: Optional[int] = Query(0, description="Last frame number received (for incremental polling)")
):
    """Stream real-time inference results for a job
    
    This endpoint polls the RayServe backend and returns incremental results.
    Use the `last_frame` parameter to get only new results since the last poll.
    
    Args:
        job_id: Job ID to get results for
        limit: Optional limit on number of results
        last_frame: Last frame number received (for incremental polling)
    
    Returns:
        JSON response with results including frame_number, predictions, and model_id
    """
    try:
        # Fetch results from RayServe backend
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{RAY_SERVE_URL}/api/v1/inference/results/{job_id}"
            params = {}
            if limit:
                params["limit"] = limit
            
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Extract results
        results = data.get("results", [])
        
        # Filter results to return only new ones (after last_frame)
        new_results = [
            result for result in results
            if result.get("frame_number", 0) > last_frame
        ]
        
        # If no new results, return empty list with current status
        if not new_results:
            return JSONResponse(
                status_code=200,
                content={
                    "job_id": job_id,
                    "status": data.get("status", "unknown"),
                    "results": [],
                    "total_results": len(results),
                    "new_results": 0,
                    "last_frame": last_frame
                }
            )
        
        # Return new results
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": data.get("status", "unknown"),
                "results": new_results,
                "total_results": len(results),
                "new_results": len(new_results),
                "last_frame": new_results[-1].get("frame_number", last_frame) if new_results else last_frame
            }
        )
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to RayServe backend: {e}")
        raise HTTPException(status_code=503, detail="RayServe backend unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in stream_results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.streaming_api:app",
        host="0.0.0.0",
        port=settings.streaming_api_port,
        log_level=settings.log_level.lower()
    )

