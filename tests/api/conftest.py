# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
API test fixtures for test isolation.

Provides utilities to avoid duplicate content errors across tests.
"""

import pytest
import uuid
from typing import Callable


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
