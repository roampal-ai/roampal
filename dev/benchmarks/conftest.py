"""
Pytest configuration for Roampal benchmark tests.

These benchmarks test the memory system's:
- Learning effectiveness (outcome scoring)
- Adversarial resistance (poisoning, confusion)
- Performance under stress (high volume, concurrent access)
- Quality metrics (MRR, nDCG, precision)
"""
import pytest
import sys
import os

# Add backend to path for imports
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


@pytest.fixture(scope="session")
def benchmark_data_dir():
    """Return path to benchmark test data."""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(scope="session")
def ab_test_data_dir():
    """Return path to A/B test data."""
    return os.path.join(os.path.dirname(__file__), 'ab_test_data')

@pytest.fixture
def harness():
    """Provide TortureTestHarness for torture suite tests."""
    import time
    from test_torture_suite import TortureTestHarness
    h = TortureTestHarness()
    h.start_time = time.time()
    return h
