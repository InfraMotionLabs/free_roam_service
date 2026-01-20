"""Request handlers for Ray Serve API endpoints"""

import json
from typing import Dict, Any, Optional
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.exceptions import FreeRoamException, InferenceException, PromptException

logger = logging.getLogger(__name__)


async def parse_json_body(request: Request) -> Dict[str, Any]:
    """Parse JSON body from request"""
    try:
        body = await request.body()
        if not body:
            return {}
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def create_error_response(error: str, detail: Optional[str] = None, status_code: int = 500) -> JSONResponse:
    """Create error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail
        }
    )


def create_success_response(data: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    """Create success response"""
    return JSONResponse(
        status_code=status_code,
        content=data
    )



