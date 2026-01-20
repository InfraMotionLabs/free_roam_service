"""Custom exception classes for the Free-Roam Inference Service"""


class FreeRoamException(Exception):
    """Base exception for all Free-Roam service errors"""
    pass


class StreamException(FreeRoamException):
    """Base exception for stream-related errors"""
    pass


class HLSStreamException(StreamException):
    """Exception raised when HLS stream operations fail"""
    pass


class RedisStreamException(StreamException):
    """Exception raised when Redis stream operations fail"""
    pass


class StreamConnectionException(StreamException):
    """Exception raised when stream connection fails"""
    pass


class StreamTimeoutException(StreamException):
    """Exception raised when stream operations timeout"""
    pass


class FrameExtractionException(FreeRoamException):
    """Exception raised when frame extraction fails"""
    pass


class ModelException(FreeRoamException):
    """Base exception for model-related errors"""
    pass


class ModelNotInitializedException(ModelException):
    """Exception raised when model is not initialized"""
    pass


class ModelLoadException(ModelException):
    """Exception raised when model loading fails"""
    pass


class InferenceException(ModelException):
    """Exception raised when inference fails"""
    pass


class InferenceTimeoutException(InferenceException):
    """Exception raised when inference times out"""
    pass


class StateException(FreeRoamException):
    """Exception raised when state management operations fail"""
    pass


class PromptException(StateException):
    """Exception raised when prompt operations fail"""
    pass


class ConfigurationException(FreeRoamException):
    """Exception raised when configuration is invalid"""
    pass


class ValidationException(FreeRoamException):
    """Exception raised when input validation fails"""
    pass



