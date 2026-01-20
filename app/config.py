"""Configuration management with environment variable support"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Model Configuration
    model_path: str = Field(default="", env="MODEL_PATH", description="Path to model directory")
    model_type: str = Field(default="example_vlm", env="MODEL_TYPE", description="Model type identifier")
    device: str = Field(default="auto", env="DEVICE", description="Device: 'auto', 'cuda', or 'cpu'")
    
    # Processing Configuration
    default_fps: float = Field(default=2.0, env="DEFAULT_FPS", ge=0.1, le=60.0, description="Default frames per second for sampling")
    max_batch_size: int = Field(default=8, env="MAX_BATCH_SIZE", ge=1, le=64, description="Maximum batch size for inference")
    inference_timeout: float = Field(default=30.0, env="INFERENCE_TIMEOUT", ge=1.0, description="Inference timeout in seconds")
    frame_timeout: float = Field(default=5.0, env="FRAME_TIMEOUT", ge=1.0, description="Frame extraction timeout in seconds")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST", description="Redis server host")
    redis_port: int = Field(default=6379, env="REDIS_PORT", ge=1, le=65535, description="Redis server port")
    redis_db: int = Field(default=0, env="REDIS_DB", ge=0, description="Redis database number")
    redis_stream_key: str = Field(default="video_frames", env="REDIS_STREAM_KEY", description="Redis stream key for video frames")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD", description="Redis password if required")
    redis_socket_timeout: float = Field(default=5.0, env="REDIS_SOCKET_TIMEOUT", ge=1.0, description="Redis socket timeout")
    
    # API Configuration
    api_key: Optional[str] = Field(default=None, env="API_KEY", description="Optional API key for authentication")
    api_title: str = Field(default="Free-Roam Inference Service (Ray Serve)", env="API_TITLE", description="API title")
    api_version: str = Field(default="1.0.0", env="API_VERSION", description="API version")
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS", description="CORS allowed origins")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE", ge=1, description="Rate limit per minute per IP")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL", description="Logging level")
    log_format: str = Field(default="json", env="LOG_FORMAT", description="Log format: 'json' or 'text'")
    
    # Ray Serve Configuration
    ray_serve_host: str = Field(default="0.0.0.0", env="RAY_SERVE_HOST", description="Ray Serve host")
    ray_serve_port: int = Field(default=8000, env="RAY_SERVE_PORT", ge=1, le=65535, description="Ray Serve port")
    num_replicas: int = Field(default=1, env="NUM_REPLICAS", ge=1, description="Number of deployment replicas")
    
    # Memory Management
    max_concurrent_streams: int = Field(default=10, env="MAX_CONCURRENT_STREAMS", ge=1, description="Maximum concurrent stream processing jobs")
    memory_cleanup_interval: int = Field(default=100, env="MEMORY_CLEANUP_INTERVAL", ge=1, description="Frames processed before cleanup")
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        """Validate device setting"""
        if v not in ["auto", "cuda", "cpu"]:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",")]
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()



