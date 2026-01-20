"""Pydantic schemas for API request/response models"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


# Inference Schemas

class InferenceStartRequest(BaseModel):
    """Request to start inference"""
    stream_ref: str = Field(..., description="HLS URL or Redis stream key")
    prompt: Optional[str] = Field(None, description="Prompt text (uses current prompt if not provided)")
    fps: Optional[float] = Field(None, ge=0.1, le=60.0, description="Frames per second (uses default if not provided)")
    job_id: Optional[str] = Field(None, description="Optional job ID (generated if not provided)")
    
    @validator("stream_ref")
    def validate_stream_ref(cls, v):
        """Validate stream reference"""
        if not v or not v.strip():
            raise ValueError("stream_ref cannot be empty")
        return v.strip()


class InferenceStartResponse(BaseModel):
    """Response from starting inference"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class InferenceStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    stream_ref: str
    prompt: str
    fps: float
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    frames_processed: int = 0
    results_count: int = 0
    error: Optional[str] = None


class InferenceResultsResponse(BaseModel):
    """Inference results response"""
    job_id: str
    status: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_results: int = 0


# Prompt Schemas

class PromptUpdateRequest(BaseModel):
    """Request to update prompt"""
    prompt: str = Field(..., description="New prompt text")
    job_id: Optional[str] = Field(None, description="Optional job ID for job-specific prompt")
    preserve_previous: bool = Field(True, description="Preserve previous prompt in history")
    
    @validator("prompt")
    def validate_prompt(cls, v):
        """Validate prompt"""
        if not v or not v.strip():
            raise ValueError("prompt cannot be empty")
        return v.strip()


class PromptUpdateResponse(BaseModel):
    """Response from updating prompt"""
    prompt: str
    version: int
    timestamp: float
    created_at: str
    updated_at: str
    job_id: Optional[str] = None


class PromptCurrentResponse(BaseModel):
    """Current prompt response"""
    prompt: str
    version: int
    timestamp: float
    created_at: str
    updated_at: str
    job_id: Optional[str] = None


class PromptHistoryResponse(BaseModel):
    """Prompt history response"""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


# Health Check Schemas

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    uptime_seconds: float
    uptime_formatted: str
    timestamp: str
    model_loaded: bool


class ReadinessResponse(BaseModel):
    """Readiness check response"""
    ready: bool
    status: str
    checks: Dict[str, bool]
    timestamp: str


class LivenessResponse(BaseModel):
    """Liveness check response"""
    alive: bool
    timestamp: str


# Error Schemas

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


