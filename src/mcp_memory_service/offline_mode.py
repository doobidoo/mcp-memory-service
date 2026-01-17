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

"""Offline mode configuration for HuggingFace models.

Call setup_offline_mode() before importing ML libraries to configure
offline behavior and cache paths.
"""

import os
import platform


def setup_offline_mode():
    """Setup offline mode environment variables to prevent model downloads.

    Offline mode is only enabled if:
    1. User explicitly sets MCP_MEMORY_OFFLINE=1, OR
    2. User has already set HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE

    This allows first-time installations to download models when needed.
    """
    # Configure cache paths first (always needed)
    username = os.environ.get('USERNAME', os.environ.get('USER', ''))
    if platform.system() == "Windows" and username:
        default_hf_home = f"C:\\Users\\{username}\\.cache\\huggingface"
        default_transformers_cache = f"C:\\Users\\{username}\\.cache\\huggingface\\transformers"
        default_sentence_transformers_home = f"C:\\Users\\{username}\\.cache\\torch\\sentence_transformers"
    else:
        default_hf_home = os.path.expanduser("~/.cache/huggingface")
        default_transformers_cache = os.path.expanduser("~/.cache/huggingface/transformers")
        default_sentence_transformers_home = os.path.expanduser("~/.cache/torch/sentence_transformers")

    # Set cache paths if not already set
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = default_hf_home
    if 'TRANSFORMERS_CACHE' not in os.environ:
        os.environ['TRANSFORMERS_CACHE'] = default_transformers_cache
    if 'SENTENCE_TRANSFORMERS_HOME' not in os.environ:
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = default_sentence_transformers_home

    # Only set offline mode if explicitly requested
    # This allows first-time installations to download models when network is available
    offline_requested = os.environ.get('MCP_MEMORY_OFFLINE', '').lower() in ('1', 'true', 'yes')
    already_offline = (
        os.environ.get('HF_HUB_OFFLINE', '').lower() in ('1', 'true', 'yes') or
        os.environ.get('TRANSFORMERS_OFFLINE', '').lower() in ('1', 'true', 'yes')
    )

    if offline_requested or already_offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
