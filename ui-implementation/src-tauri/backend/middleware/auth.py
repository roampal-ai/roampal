"""
Authentication middleware for Roampal API
"""
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from typing import Optional
import logging
import os

from .security import verify_api_key, hash_api_key, RateLimiter

logger = logging.getLogger(__name__)

# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# Initialize rate limiter
rate_limiter = RateLimiter(
    max_requests=int(os.environ.get('ROAMPAL_RATE_LIMIT_REQUESTS', '100')),
    window_seconds=int(os.environ.get('ROAMPAL_RATE_LIMIT_PERIOD', '60'))
)

# Get API key from environment
ROAMPAL_API_KEY = os.environ.get('ROAMPAL_API_KEY')
REQUIRE_AUTH = os.environ.get('ROAMPAL_REQUIRE_AUTH', 'false').lower() == 'true'

# Hash the API key if provided
if ROAMPAL_API_KEY:
    HASHED_API_KEY = hash_api_key(ROAMPAL_API_KEY)
else:
    HASHED_API_KEY = None

async def get_api_key(
    api_key_from_header: Optional[str] = Security(api_key_header),
    api_key_from_query: Optional[str] = Security(api_key_query),
) -> Optional[str]:
    """Extract API key from request"""
    return api_key_from_header or api_key_from_query

async def verify_authentication(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key)
) -> bool:
    """
    Verify API authentication and rate limiting

    Args:
        request: FastAPI request object
        api_key: API key from request

    Returns:
        True if authenticated

    Raises:
        HTTPException: If authentication fails or rate limit exceeded
    """
    # Get client identifier (IP address)
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limit
    if not rate_limiter.check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    # Skip auth if not required
    if not REQUIRE_AUTH:
        return True

    # Check if API key is provided
    if not api_key:
        logger.warning(f"Missing API key from {client_ip}")
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": f"ApiKey realm=\"{API_KEY_NAME}\""}
        )

    # Verify API key
    if not HASHED_API_KEY or not verify_api_key(api_key, HASHED_API_KEY):
        logger.warning(f"Invalid API key from {client_ip}")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    logger.info(f"Authenticated request from {client_ip}")
    return True

# Optional: Dependency for protected endpoints
def require_auth(authenticated: bool = Depends(verify_authentication)):
    """Dependency to require authentication"""
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return authenticated

# IP whitelist checking
def check_ip_whitelist(request: Request) -> bool:
    """
    Check if client IP is in whitelist

    Args:
        request: FastAPI request object

    Returns:
        True if IP is allowed
    """
    allowed_ips = os.environ.get('ROAMPAL_ALLOWED_IPS', '*').split(',')

    # Allow all if wildcard
    if '*' in allowed_ips:
        return True

    client_ip = request.client.host if request.client else None
    if not client_ip:
        return False

    return client_ip in allowed_ips

async def verify_ip_whitelist(request: Request) -> bool:
    """
    Verify client IP is in whitelist

    Args:
        request: FastAPI request object

    Returns:
        True if allowed

    Raises:
        HTTPException: If IP not in whitelist
    """
    if not check_ip_whitelist(request):
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
        raise HTTPException(
            status_code=403,
            detail="Access denied from this IP address"
        )

    return True