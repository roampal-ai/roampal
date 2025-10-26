"""
CORS Handler - Ensures all responses include proper CORS headers
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Dict, Any

def cors_response(
    content: Dict[str, Any], 
    status_code: int = 200,
    headers: Dict[str, str] = None
) -> JSONResponse:
    """Create a response with CORS headers"""
    default_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    
    if headers:
        default_headers.update(headers)
    
    return JSONResponse(
        content=content,
        status_code=status_code,
        headers=default_headers
    )

def cors_error_response(
    error: str,
    status_code: int = 400,
    details: Dict[str, Any] = None
) -> JSONResponse:
    """Create an error response with CORS headers"""
    content = {"error": error}
    if details:
        content["details"] = details
    
    return cors_response(content, status_code)