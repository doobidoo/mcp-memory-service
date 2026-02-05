import os
import shutil
import sys
import tempfile

import pytest

# Force CPU-only mode for tests (avoids CUDA compatibility issues)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for ChromaDB testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)
