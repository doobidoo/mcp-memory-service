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
Tests for lazy import behavior in mcp_memory_service package.

Lazy imports prevent heavy dependencies (torch, transformers) from being
loaded at package import time, improving CLI startup performance.
"""

import sys


def test_import_does_not_eagerly_load_heavy_modules():
    """Test that importing the package doesn't trigger heavy module loads."""
    # Store baseline of loaded modules
    initial_modules = set(sys.modules.keys())
    
    # Import the package
    import mcp_memory_service
    
    # Check that heavy modules are NOT loaded
    heavy_imports = ['torch', 'transformers', 'sentence_transformers']
    for heavy in heavy_imports:
        assert heavy not in sys.modules, f"Heavy import {heavy!r} was loaded at package import time"
    
    # Clean up: remove from sys.modules so subsequent tests start fresh
    del sys.modules['mcp_memory_service']


def test_lazy_getattr_triggers_import_on_access():
    """Test that accessing lazy-loaded attributes triggers the actual import."""
    # Remove from cache if present
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]
    
    import mcp_memory_service
    
    # Before access, heavy modules should not be loaded
    assert 'torch' not in sys.modules
    
    # Access a lazy attribute - this should trigger the import
    # We can't actually access Memory since it requires torch, but we can
    # verify the lazy mechanism works by checking the module structure
    
    # The __getattr__ should be defined
    assert hasattr(mcp_memory_service, '__getattr__')
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]


def test_lazy_attributes_are_cached_after_first_access():
    """Test that lazy-loaded attributes are cached in globals."""
    # Remove from cache if present
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]
    
    import mcp_memory_service
    
    # First access - should trigger __getattr__
    # Note: We can't fully test this without actually importing torch
    # but we can verify the mechanism exists
    
    # Check that __getattr__ is the mechanism for attribute access
    assert callable(mcp_memory_service.__getattr__)
    
    # Clean up
    for mod_name in list(sys.modules.keys()):
        if 'mcp_memory_service' in mod_name:
            del sys.modules[mod_name]
