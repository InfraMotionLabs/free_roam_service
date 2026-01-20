"""Monitoring utilities for health checks and metrics"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import logging

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Health monitoring for the service"""
    
    def __init__(self):
        """Initialize health monitor"""
        self._start_time = time.time()
        self._model_loaded = False
        self._last_check = time.time()
    
    def set_model_loaded(self, loaded: bool = True) -> None:
        """Set model loaded status"""
        self._model_loaded = loaded
        logger.info(f"Model loaded status set to: {loaded}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status
        
        Returns:
            Dictionary with health information
        """
        uptime = time.time() - self._start_time
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": self._model_loaded
        }
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status (for /health/ready endpoint)
        
        Returns:
            Dictionary with readiness information
        """
        is_ready = self._model_loaded
        
        return {
            "ready": is_ready,
            "status": "ready" if is_ready else "not_ready",
            "checks": {
                "model_loaded": self._model_loaded
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_liveness_status(self) -> Dict[str, Any]:
        """Get liveness status (for /health/live endpoint)
        
        Returns:
            Dictionary with liveness information
        """
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics
        
        Returns:
            Dictionary with system metrics
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_total_mb": psutil.virtual_memory().total / 1024 / 1024,
                    "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
                    "memory_percent": psutil.virtual_memory().percent
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        
        return " ".join(parts)


# Global health monitor instance
health_monitor = HealthMonitor()



