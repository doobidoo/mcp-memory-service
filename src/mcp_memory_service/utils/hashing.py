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

import hashlib
import json
from typing import Any


def generate_content_hash(content: str, metadata: dict[str, Any] | None = None) -> str:
    """
    Generate a unique hash for content and metadata.

    This improved version ensures consistent hashing by:
    1. Normalizing content (strip whitespace, lowercase)
    2. Sorting metadata keys
    3. Using a consistent JSON serialization
    """
    # Normalize content
    normalized_content = content.strip().lower()

    # Create hash content with normalized content
    hash_content = normalized_content

    # Add metadata if present
    if metadata:
        # Filter out timestamp and dynamic fields
        static_metadata = {k: v for k, v in metadata.items() if k not in ["timestamp", "content_hash", "embedding"]}
        if static_metadata:
            # Sort keys and use consistent JSON serialization
            hash_content += json.dumps(static_metadata, sort_keys=True, ensure_ascii=True)

    # Generate hash
    return hashlib.sha256(hash_content.encode("utf-8")).hexdigest()
