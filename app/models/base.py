"""Abstract base class for Vision-Language Models"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from app.core.exceptions import ModelNotInitializedException, InferenceException


class BaseVLM(ABC):
    """Abstract interface for Vision-Language Models
    
    This class defines the interface that all VLM implementations must follow.
    Implementations should inherit from this class and provide concrete
    implementations of all abstract methods.
    """
    
    def __init__(self):
        """Initialize the VLM base class"""
        self._initialized = False
        self._device = None
        self._model = None
    
    @abstractmethod
    async def initialize(self, model_path: str, device: str = "auto") -> None:
        """Load and initialize the model
        
        Args:
            model_path: Path to the model directory or model identifier
            device: Device to run on ('auto', 'cuda', or 'cpu')
            
        Raises:
            ModelLoadException: If model loading fails
        """
        pass
    
    @abstractmethod
    async def predict(
        self, 
        frames: List[np.ndarray], 
        prompt: str
    ) -> Dict[str, Any]:
        """Run inference on frames with given prompt
        
        Args:
            frames: List of video frames as numpy arrays (H, W, C) in RGB format
            prompt: Text prompt describing what to detect/analyze
            
        Returns:
            Dictionary containing predictions with keys:
            - 'predictions': List of prediction results
            - 'confidence': Confidence scores
            - 'metadata': Additional metadata
            
        Raises:
            ModelNotInitializedException: If model is not initialized
            InferenceException: If inference fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata
        
        Returns:
            Dictionary containing model information:
            - 'model_type': Type of model
            - 'device': Device being used
            - 'initialized': Whether model is initialized
            - Additional model-specific metadata
        """
        pass
    
    def _check_initialized(self) -> None:
        """Check if model is initialized, raise exception if not"""
        if not self._initialized:
            raise ModelNotInitializedException(
                "Model must be initialized before use. Call initialize() first."
            )
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use
        
        Args:
            device: Device preference ('auto', 'cuda', or 'cpu')
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized"""
        return self._initialized
    
    @property
    def device(self) -> Optional[str]:
        """Get the device being used"""
        return self._device



