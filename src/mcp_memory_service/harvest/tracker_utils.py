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

"""Utilities for harvest tracker persistence and management."""

import asyncio
import logging
from typing import Set, Optional

logger = logging.getLogger(__name__)

# Module-level lock for tracker operations when lock not provided
_TRACKER_LOCK = asyncio.Lock()


async def read_tracker_strict(memory_service) -> Optional[Set[str]]:
    """Read the harvest tracker, distinguishing empty from read failure.
    
    Returns:
        Set[str]: The session IDs from the tracker (empty set on first run)
        None: Read failure (database locked, connection error, etc.)
    """
    try:
        tracker = await memory_service.list_memories(page=1, page_size=1, tags=["harvest-tracker"])
        for mem in tracker.get("memories", []):
            content = mem.get("content", "")
            if content.startswith("harvested_sessions:"):
                ids_str = content.split(":", 1)[1]
                result_ids = {s for s in ids_str.split(",") if s}
                
                # Validate against stored metadata count to detect truncation
                expected_count = mem.get("metadata", {}).get("count")
                if expected_count is not None and len(result_ids) != expected_count:
                    logger.warning("Tracker truncation detected: parsed %d sessions but metadata shows %d - treating as read failure", 
                                  len(result_ids), expected_count)
                    return None
                    
                return result_ids
        # No tracker found - first run
        return set()
    except Exception as e:
        logger.warning("Failed to read harvest tracker: %s", e)
        return None


async def persist_tracker(memory_service, new_session_ids: Set[str], 
                         force_shrink: bool = False, lock=None) -> bool:
    """Persist harvest tracker with monotonic merge and safety guards.
    
    Args:
        memory_service: The memory service instance
        new_session_ids: New session IDs to add to the tracker
        force_shrink: If True, allows reducing tracker size 
        lock: Optional asyncio Lock for thread safety
        
    Returns:
        bool: True if successfully persisted, False if aborted
    """
    async def _do_persist():
        # 1. Read current tracker state
        current_ids = await read_tracker_strict(memory_service)
        
        if current_ids is None:
            # Read failed - abort to avoid data loss
            logger.error("Aborting tracker update - failed to read current state")
            return False
            
        # 2. Merge monotonically: new = current ∪ new
        all_ids = current_ids | new_session_ids
        
        # 3. Anti-shrink guard - Check if result would be smaller than current
        # For union operations, this should never trigger unless something is very wrong
        if len(all_ids) < len(current_ids) and not force_shrink:
            logger.warning("Tracker would shrink from %d to %d sessions without force_shrink - aborting", 
                          len(current_ids), len(all_ids))  
            return False
            
        # 4. Store-before-delete atomicity
        try:
            # Store new tracker first
            tracker_content = f"harvested_sessions:{','.join(sorted(all_ids))}"
            await memory_service.store_memory(
                content=tracker_content,
                tags=["harvest-tracker"],
                memory_type="observation", 
                metadata={"count": len(all_ids)},
            )
            
            # Only delete old tracker after successful store
            # FIX C1: Use page_size=1000 to fetch ALL trackers, not just 1
            old_tracker = await memory_service.list_memories(
                page=1, page_size=1000, tags=["harvest-tracker"]
            )
            
            # Delete all but the newest (the one we just stored) 
            memories = old_tracker.get("memories", [])
            if len(memories) > 1:
                # Sort by creation time and keep the newest
                sorted_mems = sorted(memories, key=lambda m: m.get("created_at", 0))
                for mem in sorted_mems[:-1]:  # Delete all but the last (newest)
                    await memory_service.storage.delete(mem["content_hash"])
                    
            return True
            
        except Exception as e:
            logger.error("Failed to store new tracker: %s", e)
            return False
    
    if lock:
        async with lock:
            return await _do_persist()
    else:
        # Use module-level lock when no lock provided to serialize across entry points
        async with _TRACKER_LOCK:
            return await _do_persist()