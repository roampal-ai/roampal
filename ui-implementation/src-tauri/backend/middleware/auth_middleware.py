"""Simplified authentication middleware - API key only"""

import logging
from typing import Optional
from fastapi import Request

logger = logging.getLogger(__name__)

class SimplifiedAuthMiddleware:
    """Simplified middleware for optional API key authentication"""

    def __init__(self):
        pass

    async def __call__(self, request: Request) -> bool:
        """Check API key if required"""
        # For now, always return True (no auth)
        # Can be extended later to check API key from middleware/auth.py
        return True

def check_api_key(request: Request) -> bool:
    """Check if request has valid API key"""
    # Simplified - no authentication for single-user tool
    return True