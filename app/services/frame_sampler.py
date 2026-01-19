"""Frame sampling logic with configurable FPS"""

import numpy as np
from typing import Iterator, List, Optional
import logging

from app.core.exceptions import FrameExtractionException

logger = logging.getLogger(__name__)


class FrameSampler:
    """Frame sampler for extracting frames at specified FPS"""
    
    def __init__(self, target_fps: float = 2.0):
        """Initialize frame sampler
        
        Args:
            target_fps: Target frames per second to sample
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than 0")
        
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # Time between frames in seconds
        logger.info(f"FrameSampler initialized with target_fps={target_fps}, interval={self.frame_interval:.3f}s")
    
    def sample_frames(
        self,
        frames: List[np.ndarray],
        source_fps: float,
        start_time: float = 0.0
    ) -> List[np.ndarray]:
        """Sample frames from a list at target FPS
        
        Args:
            frames: List of all frames
            source_fps: Original FPS of the video/stream
            start_time: Start time offset in seconds
            
        Returns:
            List of sampled frames
        """
        if not frames:
            return []
        
        if source_fps <= 0:
            raise ValueError("source_fps must be greater than 0")
        
        # Calculate frame indices to sample
        sampled_indices = self._calculate_sample_indices(
            num_frames=len(frames),
            source_fps=source_fps,
            start_time=start_time
        )
        
        # Extract sampled frames
        sampled_frames = [frames[i] for i in sampled_indices if i < len(frames)]
        
        logger.debug(
            f"Sampled {len(sampled_frames)} frames from {len(frames)} "
            f"(source_fps={source_fps:.2f}, target_fps={self.target_fps:.2f})"
        )
        
        return sampled_frames
    
    def sample_frames_generator(
        self,
        frame_generator: Iterator[np.ndarray],
        source_fps: float,
        start_time: float = 0.0,
        max_frames: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """Sample frames from a generator at target FPS (memory-efficient)
        
        Args:
            frame_generator: Generator yielding frames
            source_fps: Original FPS of the video/stream
            start_time: Start time offset in seconds
            max_frames: Maximum number of frames to sample (None for unlimited)
            
        Yields:
            Sampled frames
        """
        if source_fps <= 0:
            raise ValueError("source_fps must be greater than 0")
        
        frame_index = 0
        sampled_count = 0
        next_sample_time = start_time
        
        try:
            for frame in frame_generator:
                current_time = frame_index / source_fps
                
                # Check if we should sample this frame
                if current_time >= next_sample_time:
                    yield frame
                    sampled_count += 1
                    next_sample_time += self.frame_interval
                    
                    # Check max_frames limit
                    if max_frames is not None and sampled_count >= max_frames:
                        logger.debug(f"Reached max_frames limit: {max_frames}")
                        break
                
                frame_index += 1
                
        except Exception as e:
            logger.error(f"Error in frame sampling generator: {e}")
            raise FrameExtractionException(f"Frame sampling failed: {e}") from e
    
    def _calculate_sample_indices(
        self,
        num_frames: int,
        source_fps: float,
        start_time: float = 0.0
    ) -> List[int]:
        """Calculate which frame indices to sample
        
        Args:
            num_frames: Total number of frames
            source_fps: Original FPS
            start_time: Start time offset
            
        Returns:
            List of frame indices to sample
        """
        if num_frames == 0:
            return []
        
        indices = []
        current_time = start_time
        frame_interval_source = 1.0 / source_fps
        
        while current_time * source_fps < num_frames:
            frame_index = int(current_time * source_fps)
            if frame_index < num_frames:
                indices.append(frame_index)
            current_time += self.frame_interval
        
        return indices
    
    def set_target_fps(self, target_fps: float) -> None:
        """Update target FPS
        
        Args:
            target_fps: New target FPS
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than 0")
        
        old_fps = self.target_fps
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        logger.info(f"Updated target_fps from {old_fps} to {target_fps}")
    
    def calculate_expected_samples(self, duration: float) -> int:
        """Calculate expected number of samples for a given duration
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Expected number of samples
        """
        return int(duration * self.target_fps) + 1

