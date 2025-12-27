#!/bin/bash
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

set -euo pipefail

# Validate that all handlers can be imported without errors
# Catches Issue #299 style bugs (ModuleNotFoundError, ImportError)
#
# Exit codes:
#   0 - All imports successful
#   1 - Import validation failed

echo "🔍 Validating handler imports..."

# Test all 17 memory handlers can be imported
python3 -c "
import sys
import traceback

try:
    from mcp_memory_service.server.handlers.memory import (
        handle_store_memory,
        handle_retrieve_memory,
        handle_retrieve_with_quality_boost,
        handle_search_by_tag,
        handle_delete_memory,
        handle_delete_by_tag,
        handle_delete_by_tags,
        handle_delete_by_all_tags,
        handle_cleanup_duplicates,
        handle_update_memory_metadata,
        handle_debug_retrieve,
        handle_exact_match_retrieve,
        handle_get_raw_embedding,
        handle_recall_memory,
        handle_recall_by_timeframe,
        handle_delete_by_timeframe,
        handle_delete_before_date,
    )
    print('✅ All 17 handler imports successful')
    sys.exit(0)
except ImportError as e:
    print(f'❌ Import validation failed: {e}', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error during import: {e}', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Handler import validation passed"
    exit 0
else
    echo "❌ Handler import validation failed" >&2
    echo "💡 This catches bugs like Issue #299 (relative import errors)" >&2
    exit 1
fi
