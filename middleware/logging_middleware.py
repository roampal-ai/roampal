"""
Enhanced logging middleware with correlation IDs for request tracing
"""
import uuid
import time
import logging
from contextvars import ContextVar
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

logger = logging.getLogger(__name__)

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation IDs to all requests"""

    async def dispatch(self, request: Request, call_next):
        # Get or generate correlation ID
        correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        correlation_id_var.set(correlation_id)

        # Add correlation ID to request state
        request.state.correlation_id = correlation_id

        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started",
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'client_ip': request.client.host if request.client else 'unknown'
            }
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log unhandled exceptions
            logger.error(
                f"Unhandled exception: {str(e)}",
                extra={'correlation_id': correlation_id},
                exc_info=True
            )
            response = Response(
                content="Internal server error",
                status_code=500
            )

        # Calculate request duration
        duration = time.time() - start_time

        # Add correlation ID to response headers
        response.headers['X-Correlation-ID'] = correlation_id

        # Log request completion
        logger.info(
            f"Request completed",
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'duration': f"{duration:.3f}s"
            }
        )

        return response

class StructuredLogger:
    """Custom logger with structured output and correlation ID support"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: int, msg: str, **kwargs):
        """Internal log method with correlation ID injection"""
        correlation_id = correlation_id_var.get()
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = correlation_id
        kwargs['extra'] = extra
        self.logger.log(level, msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

def setup_structured_logging(log_level: str = "INFO"):
    """Configure structured logging with JSON output"""
    import json
    import sys

    class JSONFormatter(logging.Formatter):
        """JSON formatter for structured logs"""

        def format(self, record):
            log_obj = {
                'timestamp': self.formatTime(record),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'correlation_id': getattr(record, 'correlation_id', None),
            }

            # Add extra fields
            if hasattr(record, 'method'):
                log_obj['method'] = record.method
            if hasattr(record, 'path'):
                log_obj['path'] = record.path
            if hasattr(record, 'status_code'):
                log_obj['status_code'] = record.status_code
            if hasattr(record, 'duration'):
                log_obj['duration'] = record.duration
            if hasattr(record, 'client_ip'):
                log_obj['client_ip'] = record.client_ip

            # Add exception info if present
            if record.exc_info:
                log_obj['exception'] = self.formatException(record.exc_info)

            return json.dumps(log_obj)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add JSON formatter to console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    # Also add file handler with rotation
    from logging.handlers import RotatingFileHandler
    import os

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    file_handler = RotatingFileHandler(
        'logs/loopsmith.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context"""
    return correlation_id_var.get()

def sanitize_log_data(data: dict) -> dict:
    """Remove sensitive information from log data"""
    sensitive_keys = ['password', 'api_key', 'token', 'secret', 'authorization']
    sanitized = {}

    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value

    return sanitized