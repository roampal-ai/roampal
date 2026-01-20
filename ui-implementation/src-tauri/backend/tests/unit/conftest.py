"""
Pytest configuration for unit tests.

Sets up Python path for module imports.
"""

import sys
from pathlib import Path

# Add backend directory to path for all tests
# Go up: unit -> tests -> backend
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
