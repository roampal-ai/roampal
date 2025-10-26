# backend/utils/logging_utils.py
import logging
import sys
import io

def setup_logging(log_level: str = "INFO") -> None: # Removed app_name for simplicity
    """
    Configures basic root logging for the application.
    This ensures all loggers in the application will use this base configuration
    unless they are specifically configured otherwise or have propagation disabled.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Warning: Invalid log level '{log_level}' in settings. Defaulting to INFO.", file=sys.stderr)
        numeric_level = logging.INFO

    # Get the root logger
    root_logger = logging.getLogger()

    # If handlers already exist (e.g., from Uvicorn default or previous setup),
    # clear them to avoid duplicate messages and ensure our format is used.
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Create a custom stream handler that can handle Unicode on Windows
    # Wrap stdout with UTF-8 encoding to handle Unicode characters
    if sys.platform == "win32":
        # For Windows, create a TextIOWrapper that can handle Unicode
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    else:
        stream = sys.stdout
    
    # Create a custom handler with the Unicode-safe stream
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(module)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # Configure the root logger with our custom handler
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    # After basicConfig, root_logger.level will be set.
    # Any child logger (e.g., logger = logging.getLogger(__name__)) will inherit this level
    # unless its level is explicitly set higher.
    print(f"--- Root logging configured by setup_logging: Level={logging.getLevelName(root_logger.getEffectiveLevel())} ---", file=sys.stderr)

    # Optionally, set levels for very verbose third-party loggers if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("playwright").setLevel(logging.WARNING) # Playwright can be very verbose at DEBUG
