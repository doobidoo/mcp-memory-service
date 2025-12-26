import pytest
import os
import sys
import tempfile
import shutil
import uuid
from typing import Callable

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

@pytest.fixture
def temp_db_path():
    '''Create a temporary directory for database testing.'''
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def unique_content() -> Callable[[str], str]:
    """
    Generate unique test content to avoid duplicate content errors.

    Usage:
        def test_example(unique_content):
            content = unique_content("Test memory about authentication")
            hash1 = store(content, tags=["test"])

    Returns:
        A function that takes a base string and returns a unique version.
    """
    def _generator(base: str = "test") -> str:
        return f"{base} [{uuid.uuid4()}]"
    return _generator
