"""
Circuit Breaker Pattern Implementation
Prevents cascading failures in distributed services
"""

import time
import asyncio
import logging
from typing import Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded threshold
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing recovery with limited requests
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.half_open_attempts = 0
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.half_open_attempts = 0
        if self.state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker '{self.name}' recovered - closing circuit")
        self.state = CircuitState.CLOSED
    
    def _record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(f"Circuit breaker '{self.name}' opened - threshold exceeded")
            self.state = CircuitState.OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result if successful
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info(f"Circuit breaker '{self.name}' attempting recovery")
            else:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        # Limit attempts in half-open state
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
            if self.half_open_attempts > 1:
                self.state = CircuitState.OPEN
                raise CircuitOpenError(f"Circuit breaker '{self.name}' reopened - recovery failed")
        
        # Attempt the call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            logger.error(f"Circuit breaker '{self.name}' recorded failure: {e}")
            raise
    
    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time,
            "can_retry": self._should_attempt_reset() if self.state == CircuitState.OPEN else True
        }
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.half_open_attempts = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

# Global circuit breakers for different services
circuit_breakers = {
    "embedding_service": CircuitBreaker("embedding_service", failure_threshold=3, recovery_timeout=30),
    "chromadb": CircuitBreaker("chromadb", failure_threshold=5, recovery_timeout=60),
    "ollama": CircuitBreaker("ollama", failure_threshold=3, recovery_timeout=45),
    "playwright": CircuitBreaker("playwright", failure_threshold=2, recovery_timeout=30),
    "tools_service": CircuitBreaker("tools_service", failure_threshold=3, recovery_timeout=30),
}

def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get circuit breaker for a service."""
    if service_name not in circuit_breakers:
        circuit_breakers[service_name] = CircuitBreaker(service_name)
    return circuit_breakers[service_name]