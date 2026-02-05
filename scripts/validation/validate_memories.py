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

# scripts/validate_memories.py

import argparse
import asyncio
import json
import logging

from mcp_memory_service.storage.chroma import ChromaMemoryStorage

logger = logging.getLogger(__name__)


async def validate_memory_data(storage):
    """Comprehensive validation of memory data with focus on tag formatting"""

    validation_results = {
        "total_memories": 0,
        "tag_format_issues": [],
        "missing_required_fields": [],
        "inconsistent_formats": [],
        "recommendations": [],
    }

    try:
        # Get all memories from the collection
        results = storage.collection.get(include=["metadatas", "documents"])

        validation_results["total_memories"] = len(results["ids"])

        for i, meta in enumerate(results["metadatas"]):
            memory_id = results["ids"][i]

            # 1. Check Required Fields
            for field in ["content_hash", "tags"]:
                if field not in meta:
                    validation_results["missing_required_fields"].append({"memory_id": memory_id, "missing_field": field})

            # 2. Validate Tag Format
            tags = meta.get("tags", "[]")
            try:
                if isinstance(tags, str):
                    parsed_tags = json.loads(tags)
                    if not isinstance(parsed_tags, list):
                        validation_results["tag_format_issues"].append(
                            {
                                "memory_id": memory_id,
                                "issue": "Tags not in list format after parsing",
                                "current_format": type(parsed_tags).__name__,
                            }
                        )
                elif isinstance(tags, list):
                    validation_results["tag_format_issues"].append(
                        {
                            "memory_id": memory_id,
                            "issue": "Tags stored as raw list instead of JSON string",
                            "current_format": "list",
                        }
                    )
            except json.JSONDecodeError:
                validation_results["tag_format_issues"].append(
                    {"memory_id": memory_id, "issue": "Invalid JSON in tags field", "current_value": tags}
                )

            # 3. Check Tag Content
            try:
                stored_tags = json.loads(tags) if isinstance(tags, str) else tags
                if isinstance(stored_tags, list):
                    for tag in stored_tags:
                        if not isinstance(tag, str):
                            validation_results["inconsistent_formats"].append(
                                {
                                    "memory_id": memory_id,
                                    "issue": f"Non-string tag found: {type(tag).__name__}",
                                    "value": str(tag),
                                }
                            )
            except Exception as e:
                validation_results["inconsistent_formats"].append(
                    {"memory_id": memory_id, "issue": f"Error processing tags: {str(e)}", "current_tags": tags}
                )

        # Generate Recommendations
        if validation_results["tag_format_issues"]:
            validation_results["recommendations"].append("Run tag format migration to normalize all tags to JSON strings")
        if validation_results["missing_required_fields"]:
            validation_results["recommendations"].append("Repair memories with missing required fields")
        if validation_results["inconsistent_formats"]:
            validation_results["recommendations"].append("Clean up non-string tags in affected memories")

        return validation_results

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        validation_results["error"] = str(e)
        return validation_results


async def run_validation_report(storage):
    """Generate a formatted validation report"""
    results = await validate_memory_data(storage)

    report = f"""
    Memory Data Validation Report
    ============================
    Total Memories: {results["total_memories"]}

    Issues Found:
    -------------
    1. Tag Format Issues: {len(results["tag_format_issues"])}
    2. Missing Fields: {len(results["missing_required_fields"])}
    3. Inconsistent Formats: {len(results["inconsistent_formats"])}

    Recommendations:
    ---------------
    {chr(10).join(f"- {r}" for r in results["recommendations"])}

    Detailed Issues:
    ---------------
    {json.dumps(results, indent=2)}
    """

    return report


async def main():
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.ERROR),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Initialize storage
    # storage = ChromaMemoryStorage("path/to/your/db")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Validate memory data tags")
    parser.add_argument("--db-path", required=True, help="Path to ChromaDB database")
    args = parser.parse_args()

    # Initialize storage with provided path
    logger.info(f"Connecting to database at: {args.db_path}")
    storage = ChromaMemoryStorage(args.db_path)

    # Run validation and get report
    report = await run_validation_report(storage)

    # Print report to console
    print(report)

    # Save report to file
    with open("validation_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    asyncio.run(main())
