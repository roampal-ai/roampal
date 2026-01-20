"""
Root pytest configuration for backend tests.

Sets up Python path so all test files can import modules correctly.
"""

import sys
from pathlib import Path

# Add backend directory to path for all tests - do this IMMEDIATELY at module load
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def pytest_configure(config):
    """Called after command line options have been parsed and before test collection."""
    # Ensure path is set before any imports happen
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
