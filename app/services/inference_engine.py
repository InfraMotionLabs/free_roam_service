"""Async inference engine with queue management"""

import asyncio
import uuid
import time
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass, field
import logging
import gc

from app.models.base import BaseVLM
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


class InferenceEngine:
    """Async inference engine for processing video streams"""
    
    def __init__(
        self,
        model: BaseVLM,
        stream_processor: StreamProcessor,
        state_manager: PromptStateManager
    ):
        """Initialize inference engine
        
        Args:
            model: VLM model instance
            stream_processor: Stream processor instance
            state_manager: Prompt state manager instance
        """
        self.model = model
        self.stream_processor = stream_processor
        self.state_manager = state_manager
        
        self._jobs: Dict[str, InferenceJob] = {}
        self._job_lock = asyncio.Lock()
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._max_concurrent = settings.max_concurrent_streams
        
        logger.info(f"InferenceEngine initialized (max_concurrent={self._max_concurrent})")
    
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
        
        # Start processing task
        task = asyncio.create_task(self._process_job(job))
        self._processing_tasks[job_id] = task
        
        logger.info(f"Started inference job {job_id} for stream: {stream_ref}")
        
        return job_id
    
    async def _process_job(self, job: InferenceJob) -> None:
        """Process an inference job
        
        Args:
            job: Inference job to process
        """
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        
        try:
            # Create frame sampler
            sampler = FrameSampler(target_fps=job.fps)
            
            # Get frames from stream
            frame_batch = []
            batch_size = settings.max_batch_size
            frames_processed = 0
            
            async for frame in self.stream_processor.get_frames(
                stream_ref=job.stream_ref,
                fps=job.fps
            ):
                frame_batch.append(frame)
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    predictions = await self._process_batch(
                        frames=frame_batch,
                        prompt=job.prompt
                    )
                    job.results.extend(predictions)
                    frames_processed += len(frame_batch)
                    job.frames_processed = frames_processed
                    
                    # Cleanup batch
                    del frame_batch
                    frame_batch = []
                    gc.collect()
            
            # Process remaining frames
            if frame_batch:
                predictions = await self._process_batch(
                    frames=frame_batch,
                    prompt=job.prompt
                )
                job.results.extend(predictions)
                frames_processed += len(frame_batch)
                job.frames_processed = frames_processed
                
                del frame_batch
                gc.collect()
            
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            
            logger.info(
                f"Job {job.job_id} completed: {frames_processed} frames, "
                f"{len(job.results)} predictions"
            )
            
        except asyncio.TimeoutError:
            job.status = JobStatus.FAILED
            job.error = "Inference timeout"
            job.completed_at = time.time()
            logger.error(f"Job {job.job_id} timed out")
            raise InferenceTimeoutException(f"Job {job.job_id} timed out")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            raise InferenceException(f"Job {job.job_id} failed: {e}") from e
        
        finally:
            # Cleanup
            async with self._job_lock:
                if job.job_id in self._processing_tasks:
                    del self._processing_tasks[job.job_id]
    
    async def _process_batch(
        self,
        frames: List,
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Process a batch of frames
        
        Args:
            frames: List of frames
            prompt: Prompt text
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Run inference with timeout
            predictions = await asyncio.wait_for(
                self.model.predict(frames, prompt),
                timeout=settings.inference_timeout
            )
            
            # Format predictions
            if isinstance(predictions, dict):
                # Extract predictions list
                pred_list = predictions.get("predictions", [])
                if not pred_list and "predictions" not in predictions:
                    # Single prediction
                    pred_list = [predictions]
                
                return pred_list
            elif isinstance(predictions, list):
                return predictions
            else:
                return [{"prediction": predictions}]
                
        except asyncio.TimeoutError:
            logger.warning(f"Inference batch timed out after {settings.inference_timeout}s")
            raise InferenceTimeoutException("Inference batch timed out")
        except Exception as e:
            logger.error(f"Inference batch failed: {e}")
            raise InferenceException(f"Inference batch failed: {e}") from e
    
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
        """Get job results
        
        Args:
            job_id: Job ID
            limit: Optional limit on number of results
            
        Returns:
            List of results or None if job not found
        """
        async with self._job_lock:
            job = self._jobs.get(job_id)
            if job:
                results = job.results
                if limit is not None:
                    results = results[:limit]
                return results
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
            
            # Cancel processing task
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
    
    async def cleanup_old_jobs(self, max_age_seconds: float = 3600) -> int:
        """Clean up old completed/failed jobs
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of jobs cleaned up
        """
        current_time = time.time()
        cleaned = 0
        
        async with self._job_lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    age = current_time - (job.completed_at or job.created_at)
                    if age > max_age_seconds:
                        to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._jobs[job_id]
                await self.state_manager.clear_job_prompt(job_id)
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old jobs")
        
        return cleaned



