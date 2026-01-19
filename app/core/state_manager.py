"""Thread-safe prompt state management with versioning"""

import asyncio
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

from app.core.exceptions import PromptException

logger = logging.getLogger(__name__)


@dataclass
class PromptState:
    """Represents a prompt state with metadata"""
    prompt: str
    version: int
    timestamp: float
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "prompt": self.prompt,
            "version": self.version,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class PromptStateManager:
    """Thread-safe prompt state manager with versioning
    
    Manages prompt state for multiple concurrent streams/jobs.
    Each job can have its own prompt, and prompts are versioned
    for history tracking.
    """
    
    def __init__(self, default_prompt: str = ""):
        """Initialize the state manager
        
        Args:
            default_prompt: Default prompt to use initially
        """
        self._lock = asyncio.Lock()
        self._default_prompt = default_prompt
        self._global_prompt: Optional[PromptState] = None
        self._job_prompts: Dict[str, PromptState] = {}
        self._prompt_history: List[PromptState] = []
        
        # Initialize with default prompt if provided
        if default_prompt:
            self._global_prompt = PromptState(
                prompt=default_prompt,
                version=1,
                timestamp=time.time()
            )
            self._prompt_history.append(self._global_prompt)
        
        logger.info(f"PromptStateManager initialized with default prompt: {default_prompt[:50] if default_prompt else 'None'}")
    
    async def get_current_prompt(self, job_id: Optional[str] = None) -> str:
        """Get the current prompt for a job or global prompt
        
        Args:
            job_id: Optional job ID. If provided, returns job-specific prompt.
                   If None, returns global prompt.
                   
        Returns:
            Current prompt string
        """
        async with self._lock:
            if job_id and job_id in self._job_prompts:
                return self._job_prompts[job_id].prompt
            elif self._global_prompt:
                return self._global_prompt.prompt
            else:
                return self._default_prompt
    
    async def get_prompt_state(self, job_id: Optional[str] = None) -> Optional[PromptState]:
        """Get the full prompt state for a job or global
        
        Args:
            job_id: Optional job ID
            
        Returns:
            PromptState object or None
        """
        async with self._lock:
            if job_id and job_id in self._job_prompts:
                return self._job_prompts[job_id]
            else:
                return self._global_prompt
    
    async def update_prompt(
        self, 
        new_prompt: str, 
        job_id: Optional[str] = None,
        preserve_previous: bool = True
    ) -> PromptState:
        """Update the prompt for a job or globally
        
        Args:
            new_prompt: New prompt text
            job_id: Optional job ID. If provided, updates job-specific prompt.
                   If None, updates global prompt.
            preserve_previous: If True, preserves previous prompt in history
                   
        Returns:
            New PromptState object
        """
        if not new_prompt or not new_prompt.strip():
            raise PromptException("Prompt cannot be empty")
        
        async with self._lock:
            # Get current state to determine next version
            current_state = None
            if job_id and job_id in self._job_prompts:
                current_state = self._job_prompts[job_id]
            elif self._global_prompt:
                current_state = self._global_prompt
            
            next_version = (current_state.version + 1) if current_state else 1
            
            # Create new state
            new_state = PromptState(
                prompt=new_prompt.strip(),
                version=next_version,
                timestamp=time.time(),
                created_at=current_state.created_at if current_state else datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            
            # Preserve previous in history if requested
            if preserve_previous and current_state:
                self._prompt_history.append(current_state)
            
            # Update state
            if job_id:
                self._job_prompts[job_id] = new_state
                logger.info(f"Updated prompt for job {job_id} (version {next_version})")
            else:
                self._global_prompt = new_state
                self._prompt_history.append(new_state)
                logger.info(f"Updated global prompt (version {next_version})")
            
            return new_state
    
    async def set_job_prompt(self, job_id: str, prompt: str) -> PromptState:
        """Set a prompt for a specific job
        
        Args:
            job_id: Job identifier
            prompt: Prompt text
            
        Returns:
            PromptState object
        """
        return await self.update_prompt(prompt, job_id=job_id)
    
    async def clear_job_prompt(self, job_id: str) -> None:
        """Clear job-specific prompt (falls back to global)
        
        Args:
            job_id: Job identifier
        """
        async with self._lock:
            if job_id in self._job_prompts:
                del self._job_prompts[job_id]
                logger.info(f"Cleared prompt for job {job_id}")
    
    async def get_prompt_history(
        self, 
        job_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get prompt history
        
        Args:
            job_id: Optional job ID (not implemented for job-specific history yet)
            limit: Maximum number of history entries to return
            
        Returns:
            List of prompt state dictionaries
        """
        async with self._lock:
            # For now, return global history
            # In future, could track per-job history
            history = self._prompt_history[-limit:] if limit > 0 else self._prompt_history
            return [state.to_dict() for state in reversed(history)]
    
    async def get_all_job_prompts(self) -> Dict[str, Dict]:
        """Get all job-specific prompts
        
        Returns:
            Dictionary mapping job_id to prompt state
        """
        async with self._lock:
            return {
                job_id: state.to_dict() 
                for job_id, state in self._job_prompts.items()
            }
    
    async def reset_to_default(self, job_id: Optional[str] = None) -> None:
        """Reset prompt to default
        
        Args:
            job_id: Optional job ID. If provided, clears job-specific prompt.
                   If None, resets global prompt.
        """
        async with self._lock:
            if job_id:
                if job_id in self._job_prompts:
                    del self._job_prompts[job_id]
                    logger.info(f"Reset prompt for job {job_id} to default")
            else:
                if self._default_prompt:
                    self._global_prompt = PromptState(
                        prompt=self._default_prompt,
                        version=1,
                        timestamp=time.time()
                    )
                    logger.info("Reset global prompt to default")
                else:
                    self._global_prompt = None
                    logger.info("Cleared global prompt")

