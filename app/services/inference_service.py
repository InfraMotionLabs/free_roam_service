"""Inference service for Ray Serve - job tracking and processing"""

import asyncio
import uuid
import time
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass, field
import logging
import gc

from app.services.stream_processor import StreamProcessor
from app.services.frame_sampler import FrameSampler
from app.core.state_manager import PromptStateManager
from app.core.exceptions import InferenceException, InferenceTimeoutException
from app.config import settings

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Inference job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceJob:
    """Represents an inference job"""
    job_id: str
    stream_ref: str
    prompt: str
    fps: float
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    frames_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "stream_ref": self.stream_ref,
            "prompt": self.prompt,
            "fps": self.fps,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results_count": len(self.results),
            "frames_processed": self.frames_processed,
            "error": self.error
        }


class InferenceService:
    """Inference service for managing jobs and processing streams"""
    
    def __init__(
        self,
        stream_processor: StreamProcessor,
        state_manager: PromptStateManager
    ):
        """Initialize inference service
        
        Args:
            stream_processor: Stream processor instance
            state_manager: Prompt state manager instance
        """
        self.stream_processor = stream_processor
        self.state_manager = state_manager
        
        self._jobs: Dict[str, InferenceJob] = {}
        self._job_lock = asyncio.Lock()
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._max_concurrent = settings.max_concurrent_streams
        
        logger.info(f"InferenceService initialized (max_concurrent={self._max_concurrent})")
    
    async def start_inference(
        self,
        stream_ref: str,
        prompt: Optional[str] = None,
        fps: Optional[float] = None,
        job_id: Optional[str] = None
    ) -> str:
        """Start an inference job
        
        Args:
            stream_ref: Stream reference (HLS URL or Redis key)
            prompt: Optional prompt (uses current prompt if not provided)
            fps: Optional FPS (uses default if not provided)
            job_id: Optional job ID (generated if not provided)
            
        Returns:
            Job ID
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        # Get prompt
        if prompt is None:
            prompt = await self.state_manager.get_current_prompt()
        
        # Get FPS
        if fps is None:
            fps = settings.default_fps
        
        # Create job
        job = InferenceJob(
            job_id=job_id,
            stream_ref=stream_ref,
            prompt=prompt,
            fps=fps
        )
        
        async with self._job_lock:
            # Check concurrent limit
            active_jobs = sum(
                1 for j in self._jobs.values()
                if j.status in [JobStatus.PENDING, JobStatus.RUNNING]
            )
            
            if active_jobs >= self._max_concurrent:
                raise InferenceException(
                    f"Maximum concurrent streams ({self._max_concurrent}) reached"
                )
            
            self._jobs[job_id] = job
            
            # Set job-specific prompt
            await self.state_manager.set_job_prompt(job_id, prompt)
        
        logger.info(f"Created inference job {job_id} for stream: {stream_ref}")
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary or None if not found
        """
        async with self._job_lock:
            job = self._jobs.get(job_id)
            if job:
                return job.to_dict()
            return None
    
    async def get_job_results(
        self,
        job_id: str,
        limit: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get job results with enhanced fields (frame_number, model_id)
        
        Args:
            job_id: Job ID
            limit: Optional limit on number of results
            
        Returns:
            List of results with frame_number and model_id, or None if job not found
        """
        async with self._job_lock:
            job = self._jobs.get(job_id)
            if job:
                results = job.results
                if limit is not None:
                    results = results[:limit]
                
                # Enhance results with frame_number and model_id
                enhanced_results = []
                for idx, result in enumerate(results, start=1):
                    enhanced_result = result.copy()
                    enhanced_result["frame_number"] = idx
                    
                    # Extract model_id from metadata if available, otherwise use model_type
                    if "metadata" in enhanced_result:
                        metadata = enhanced_result["metadata"]
                        enhanced_result["model_id"] = metadata.get("model_id") or metadata.get("model_type", "unknown")
                    else:
                        enhanced_result["model_id"] = "unknown"
                    
                    enhanced_results.append(enhanced_result)
                
                return enhanced_results
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job
        
        Args:
            job_id: Job ID
            
        Returns:
            True if cancelled, False if not found
        """
        async with self._job_lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            # Cancel processing task if exists
            if job_id in self._processing_tasks:
                task = self._processing_tasks[job_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self._processing_tasks[job_id]
            
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            
            # Cleanup job prompt
            await self.state_manager.clear_job_prompt(job_id)
            
            logger.info(f"Job {job_id} cancelled")
            return True
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from history
        
        Args:
            job_id: Job ID
            
        Returns:
            True if deleted, False if not found or cannot be deleted
        """
        async with self._job_lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            # Only allow deletion of completed, failed, or cancelled jobs
            if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            # Remove job from history
            del self._jobs[job_id]
            
            # Cleanup job prompt
            await self.state_manager.clear_job_prompt(job_id)
            
            logger.info(f"Job {job_id} deleted from history")
            return True
    
    async def list_jobs(self, status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """List all jobs
        
        Args:
            status: Optional status filter
            
        Returns:
            List of job dictionaries
        """
        async with self._job_lock:
            jobs = list(self._jobs.values())
            if status:
                jobs = [j for j in jobs if j.status == status]
            return [job.to_dict() for job in jobs]
    
    async def process_job_batch(
        self,
        job_id: str,
        frames: List,
        prompt: str,
        model_deployment
    ) -> List[Dict[str, Any]]:
        """Process a batch of frames for a job
        
        Args:
            job_id: Job ID
            frames: List of frames
            prompt: Prompt text
            model_deployment: Ray Serve model deployment handle
            
        Returns:
            List of predictions
        """
        try:
            # Call model deployment for inference
            predictions = await model_deployment.predict.remote(frames, prompt)
            
            # Format predictions
            if isinstance(predictions, dict):
                pred_list = predictions.get("predictions", [])
                if not pred_list and "predictions" not in predictions:
                    pred_list = [predictions]
                return pred_list
            elif isinstance(predictions, list):
                return predictions
            else:
                return [{"prediction": predictions}]
                
        except Exception as e:
            logger.error(f"Inference batch failed for job {job_id}: {e}")
            raise InferenceException(f"Inference batch failed: {e}") from e



