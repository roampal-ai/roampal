"""
Security middleware and utilities for Roampal
"""
import os
import re
import hashlib
import secrets
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration and constants"""
    # Allowed directories for file operations (jail)
    ALLOWED_DIRS = [
        Path(os.environ.get('ROAMPAL_WORKSPACE', '/workspace')),
        Path(os.environ.get('ROAMPAL_PROJECTS', '/app/data/loopsmith/projects')),
    ]

    # Allowed git commands (whitelist)
    ALLOWED_GIT_COMMANDS = ['status', 'diff', 'log', 'branch', 'show', 'remote']

    # Dangerous bash commands to block
    BLOCKED_COMMANDS = [
        'rm -rf /', 'dd', 'mkfs', 'format', ';', '&&', '||', '`', '$(',
        '>', '>>', '<', '|', 'sudo', 'su', 'chmod 777', 'eval', 'exec'
    ]

    # Max file size for operations (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Max command length
    MAX_COMMAND_LENGTH = 1000

def validate_path(file_path: str, allowed_dirs: Optional[List[Path]] = None) -> Path:
    """
    Validate and sanitize file paths to prevent traversal attacks

    Args:
        file_path: The path to validate
        allowed_dirs: List of allowed base directories

    Returns:
        Validated Path object

    Raises:
        SecurityException: If path is invalid or outside allowed directories
    """
    if allowed_dirs is None:
        allowed_dirs = SecurityConfig.ALLOWED_DIRS

    # Convert to Path and resolve to absolute
    try:
        requested_path = Path(file_path).resolve()
    except (ValueError, RuntimeError) as e:
        raise SecurityException(f"Invalid path: {e}")

    # Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/etc') or file_path.startswith('/root'):
        raise SecurityException("Path traversal attempt detected")

    # Check if path is within allowed directories
    for allowed_dir in allowed_dirs:
        try:
            allowed_dir = allowed_dir.resolve()
            requested_path.relative_to(allowed_dir)

            # Additional checks
            if requested_path.is_symlink():
                raise SecurityException("Symbolic links not allowed")

            return requested_path
        except (ValueError, RuntimeError):
            continue

    raise SecurityException(f"Path outside allowed directories: {file_path}")

def validate_git_command(command: str, args: List[str]) -> bool:
    """
    Validate git commands against whitelist

    Args:
        command: Git subcommand
        args: Command arguments

    Returns:
        True if command is safe

    Raises:
        SecurityException: If command is not allowed
    """
    if command not in SecurityConfig.ALLOWED_GIT_COMMANDS:
        raise SecurityException(f"Git command not allowed: {command}")

    # Check for dangerous patterns in arguments
    dangerous_patterns = ['--upload-pack', '--receive-pack', '--exec', '-c', 'hooks']
    for arg in args or []:
        if any(pattern in arg for pattern in dangerous_patterns):
            raise SecurityException(f"Dangerous git argument detected: {arg}")

    return True

def sanitize_bash_command(command: str) -> str:
    """
    Sanitize bash commands to prevent injection

    Args:
        command: Command to sanitize

    Returns:
        Sanitized command

    Raises:
        SecurityException: If command contains dangerous patterns
    """
    if len(command) > SecurityConfig.MAX_COMMAND_LENGTH:
        raise SecurityException("Command too long")

    # Check for blocked commands
    command_lower = command.lower()
    for blocked in SecurityConfig.BLOCKED_COMMANDS:
        if blocked in command_lower:
            raise SecurityException(f"Blocked command pattern detected: {blocked}")

    # Check for command chaining attempts
    if any(char in command for char in [';', '&&', '||', '`', '$(']):
        raise SecurityException("Command chaining not allowed")

    # Check for redirection attempts
    if any(char in command for char in ['>', '<', '|']):
        raise SecurityException("Command redirection not allowed")

    return command

def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """Verify an API key against stored hash"""
    return hash_api_key(provided_key) == stored_hash

class SecurityException(Exception):
    """Custom exception for security violations"""
    pass

class PathValidator:
    """Context manager for safe file operations"""

    def __init__(self, base_path: Path):
        self.base_path = base_path.resolve()

    def validate(self, file_path: str) -> Path:
        """Validate a path is within base_path"""
        requested = Path(file_path).resolve()

        try:
            requested.relative_to(self.base_path)
            return requested
        except ValueError:
            raise SecurityException(f"Path outside allowed directory: {file_path}")

def safe_subprocess_args(args: List[str]) -> List[str]:
    """
    Sanitize subprocess arguments

    Args:
        args: List of arguments

    Returns:
        Sanitized argument list
    """
    sanitized = []
    for arg in args:
        # Remove shell metacharacters
        clean_arg = re.sub(r'[;&|`$()<>]', '', arg)
        # Limit length
        if len(clean_arg) > 1000:
            raise SecurityException("Argument too long")
        sanitized.append(clean_arg)

    return sanitized

# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}

    def check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier has exceeded rate limit"""
        import time
        current_time = time.time()

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            t for t in self.requests[identifier]
            if current_time - t < self.window_seconds
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Add current request
        self.requests[identifier].append(current_time)
        return True