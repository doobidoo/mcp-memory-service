#!/usr/bin/env python3
"""
Automated Cloudflare resource setup for MCP Memory Service.
This script creates the required Cloudflare resources using the HTTP API.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Load .env from project root when present
_project_root = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv

    load_dotenv(_project_root / ".env", override=False)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudflareSetup:
    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        self.client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
            self.client = httpx.AsyncClient(headers=headers, timeout=30.0)
        return self.client

    async def _make_request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        """Make authenticated request to Cloudflare API."""
        client = await self._get_client()
        response = await client.request(method, url, **kwargs)

        if response.status_code not in [200, 201]:
            logger.error(
                "API request failed: %s %s", response.status_code, response.text
            )
            response.raise_for_status()

        return response.json()

    async def create_vectorize_index(self, name: str = "mcp-memory-index") -> str:
        """Create Vectorize index and return its ID."""
        logger.info("Creating Vectorize index: %s", name)

        # Use Vectorize v2 API
        vectorize_base = f"{self.base_url}/vectorize/v2/indexes"

        # Check if index already exists
        try:
            url = f"{vectorize_base}/{name}"
            result = await self._make_request("GET", url)
            if result.get("success"):
                logger.info("Vectorize index %s already exists", name)
                return name
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise

        # Create new index
        url = vectorize_base
        payload = {"name": name, "config": {"dimensions": 768, "metric": "cosine"}}

        result = await self._make_request("POST", url, json=payload)
        if result.get("success"):
            logger.info("✅ Created Vectorize index: %s", name)
            return name
        else:
            raise ValueError(f"Failed to create Vectorize index: {result}")

    async def create_d1_database(self, name: str = "mcp-memory-db") -> str:
        """Create D1 database and return its ID."""
        logger.info("Creating D1 database: %s", name)

        # List existing databases to check if it exists
        url = f"{self.base_url}/d1/database"
        result = await self._make_request("GET", url)

        if result.get("success"):
            for db in result.get("result", []):
                if db.get("name") == name:
                    db_id = db.get("uuid")
                    logger.info(
                        "D1 database %s already exists with ID: %s", name, db_id
                    )
                    return db_id

        # Create new database
        payload = {"name": name}
        result = await self._make_request("POST", url, json=payload)

        if result.get("success"):
            db_id = result["result"]["uuid"]
            logger.info("✅ Created D1 database: %s (ID: %s)", name, db_id)
            return db_id
        else:
            raise ValueError(f"Failed to create D1 database: {result}")

    async def create_r2_bucket(self, name: str = "mcp-memory-content") -> str:
        """Create R2 bucket and return its name."""
        logger.info("Creating R2 bucket: %s", name)

        # Check if bucket already exists
        try:
            url = f"{self.base_url}/r2/buckets/{name}"
            result = await self._make_request("GET", url)
            if result.get("success"):
                logger.info("R2 bucket %s already exists", name)
                return name
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise

        # Create new bucket
        url = f"{self.base_url}/r2/buckets"
        payload = {"name": name}

        result = await self._make_request("POST", url, json=payload)
        if result.get("success"):
            logger.info("✅ Created R2 bucket: %s", name)
            return name
        else:
            raise ValueError(f"Failed to create R2 bucket: {result}")

    async def verify_workers_ai_access(self) -> bool:
        """Verify Workers AI access and embedding model."""
        logger.info("Verifying Workers AI access...")

        # Test embedding generation
        url = f"{self.base_url}/ai/run/@cf/baai/bge-base-en-v1.5"
        payload = {"text": ["test embedding"]}

        try:
            result = await self._make_request("POST", url, json=payload)
            if result.get("success"):
                logger.info("✅ Workers AI access verified")
                return True
            else:
                logger.warning("Workers AI test failed: %s", result)
                return False
        except (
            Exception  # noqa: BLE001 - optional capability must not abort setup
        ) as e:
            logger.warning("Workers AI verification failed: %s", e)
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()


def _write_env_file(
    api_token: str,
    account_id: str,
    vectorize_index: str,
    d1_database_id: str,
    r2_bucket: str | None = None,
) -> None:
    """Update .env in the project root with Cloudflare resource IDs."""
    env_path = _project_root / ".env"
    if not env_path.exists():
        example = _project_root / ".env.example"
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    updates = {
        "CLOUDFLARE_API_TOKEN": api_token,
        "CLOUDFLARE_ACCOUNT_ID": account_id,
        "CLOUDFLARE_VECTORIZE_INDEX": vectorize_index,
        "CLOUDFLARE_D1_DATABASE_ID": d1_database_id,
        "MCP_MEMORY_STORAGE_BACKEND": "cloudflare",
    }
    if r2_bucket:
        updates["CLOUDFLARE_R2_BUCKET"] = r2_bucket

    seen = set()
    new_lines = []
    for line in lines:
        key = (
            line.split("=", 1)[0]
            if "=" in line and not line.strip().startswith("#")
            else None
        )
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"\n✅ Wrote Cloudflare settings to {_project_root / '.env'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Cloudflare resources for MCP Memory Service."
    )
    parser.add_argument(
        "--no-r2",
        action="store_true",
        help="Skip optional R2 bucket creation",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write resource IDs into the project .env file",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Create resources without confirmation",
    )
    return parser.parse_args(argv)


def _confirm_resource_creation(skip_confirmation: bool) -> bool:
    if skip_confirmation:
        return True

    try:
        answer = input("Create these Cloudflare resources? [y/N]: ")
    except EOFError:
        return False

    return answer.strip().lower() in {"y", "yes"}


async def main(args: argparse.Namespace) -> bool:
    """Main setup routine."""
    print("🚀 Cloudflare Backend Setup for MCP Memory Service")
    print("=" * 55)

    # Check for required environment variables
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    placeholder = "your-cloudflare-api-token-here"

    if not api_token or api_token == placeholder:
        print("❌ CLOUDFLARE_API_TOKEN not set in environment or .env")
        print("1. Sign up: https://dash.cloudflare.com/sign-up")
        print("2. Create token: https://dash.cloudflare.com/profile/api-tokens")
        print("   Permissions: Vectorize:Edit, D1:Edit, Workers AI:Read")
        print("3. Add CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID to .env")
        return False

    if not account_id or account_id == "your-account-id-here":
        print("❌ CLOUDFLARE_ACCOUNT_ID not set in environment or .env")
        print("Find it in the Cloudflare dashboard right sidebar after login.")
        return False

    print("\nTarget configuration:")
    print("  Backend: Cloudflare")
    print(f"  Account ID: {account_id}")
    print("  Vectorize index: mcp-memory-index")
    print("  D1 database: mcp-memory-db")
    print(f"  R2 bucket: {'skipped' if args.no_r2 else 'optional'}")
    print()

    if not _confirm_resource_creation(args.yes):
        print("Setup cancelled by user.")
        return True

    setup = CloudflareSetup(api_token, account_id)

    try:
        # Create resources
        vectorize_index = await setup.create_vectorize_index()
        d1_database_id = await setup.create_d1_database()

        # R2 bucket is optional
        r2_bucket = None
        create_r2 = not args.no_r2 and input(
            "\n🪣 Create R2 bucket for large content storage? (y/N): "
        ).lower().strip() in ["y", "yes"]
        if create_r2:
            try:
                r2_bucket = await setup.create_r2_bucket()
            except Exception as e:  # noqa: BLE001 - R2 is optional
                logger.warning("Failed to create R2 bucket: %s", e)
                logger.warning("Continuing without R2 storage...")

        # Verify Workers AI
        ai_available = await setup.verify_workers_ai_access()

        print("\n🎉 Setup Complete!")
        print("=" * 20)
        print(f"Vectorize Index: {vectorize_index}")
        print(f"D1 Database ID: {d1_database_id}")
        print(f"R2 Bucket: {r2_bucket or 'Not configured'}")
        print(f"Workers AI: {'Available' if ai_available else 'Limited access'}")

        print("\n📝 Environment Variables:")
        print("=" * 25)
        print(f'export CLOUDFLARE_API_TOKEN="{api_token[:10]}..."')
        print(f'export CLOUDFLARE_ACCOUNT_ID="{account_id}"')
        print(f'export CLOUDFLARE_VECTORIZE_INDEX="{vectorize_index}"')
        print(f'export CLOUDFLARE_D1_DATABASE_ID="{d1_database_id}"')
        if r2_bucket:
            print(f'export CLOUDFLARE_R2_BUCKET="{r2_bucket}"')
        print('export MCP_MEMORY_STORAGE_BACKEND="cloudflare"')

        if args.write_env:
            _write_env_file(
                api_token, account_id, vectorize_index, d1_database_id, r2_bucket
            )

        print("\n🧪 Test the setup:")
        print(".venv/bin/python scripts/testing/test_cloudflare_backend.py")

        return True

    except (
        Exception  # noqa: BLE001 - CLI reports API failures without a traceback
    ) as e:
        logger.error("Setup failed: %s", e)
        return False

    finally:
        await setup.close()


if __name__ == "__main__":
    success = asyncio.run(main(parse_args()))
    sys.exit(0 if success else 1)
