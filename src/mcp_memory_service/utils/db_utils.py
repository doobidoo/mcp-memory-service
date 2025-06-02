"""Utilities for database validation and health checks."""
from typing import Dict, Any, Tuple
import logging
import os
# import json # Not used in the provided snippets directly, but often useful
from datetime import datetime
import asyncio # Added for EchoVault validation sleep and async calls in sync context

# Imports for type checking and functionality
from ..storage.chroma import ChromaMemoryStorage
from ..storage.echovault import EchoVaultStorage
from ..models.memory import Memory # For EchoVault validation
import time # For EchoVault validation test_hash and ChromaDB metadata timestamp

logger = logging.getLogger(__name__)

async def validate_database(storage) -> Tuple[bool, str]:
    logger.info(f"Validating database with storage type: {type(storage)}")
    if isinstance(storage, ChromaMemoryStorage):
        try:
            # Check if collection exists and is accessible
            collection_info = storage.collection.count()
            if collection_info == 0:
                logger.info("ChromaDB: Database is empty but accessible")
            
            # Verify embedding function is working
            test_text = "Database validation test"
            embedding = storage.embedding_function([test_text])
            if not embedding or len(embedding) == 0:
                return False, "ChromaDB: Embedding function is not working properly"
            
            # Test basic operations
            test_id = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Test add
            storage.collection.add(
                documents=[test_text],
                metadatas=[{"test": True, "timestamp": time.time()}], # Ensure metadata has timestamp
                ids=[test_id]
            )
            
            # Test query
            query_result = storage.collection.query(
                query_texts=[test_text],
                n_results=1
            )
            if not query_result["ids"]:
                return False, "ChromaDB: Query operation failed"
            
            # Clean up test data
            storage.collection.delete(ids=[test_id])
            
            return True, "ChromaDB: Database validation successful"
        except Exception as e:
            logger.error(f"ChromaDB: Database validation failed: {str(e)}")
            return False, f"ChromaDB: Database validation failed: {str(e)}"

    elif isinstance(storage, EchoVaultStorage):
        try:
            if not storage._is_initialized:
                await storage.initialize()
            
            test_content = "EchoVault validation test"
            test_hash = f"test_echovault_{int(time.time())}"
            test_memory = Memory(
                content=test_content,
                content_hash=test_hash,
                tags=["test_validation"],
                memory_type="test"
            )
            
            # Test store
            logger.info(f"EchoVault: Testing store operation for hash {test_hash}...")
            store_success, store_msg = await storage.store(test_memory)
            if not store_success:
                logger.error(f"EchoVault: Store operation failed: {store_msg}")
                return False, f"EchoVault: Store operation failed: {store_msg}"
            logger.info(f"EchoVault: Store operation successful for hash {test_hash}.")

            # Test retrieve (by query, as retrieve by hash might not be a direct public method)
            # We need to ensure the item is indexed before retrieving. A small delay might help.
            await asyncio.sleep(2) # Allow time for indexing if Qdrant is eventually consistent

            logger.info(f"EchoVault: Testing retrieve operation for content '{test_content}'...")
            retrieved_results = await storage.retrieve(query=test_content, n_results=1)
            
            found_test_item = False
            if retrieved_results:
                for res in retrieved_results:
                    if res.memory.content_hash == test_hash:
                        found_test_item = True
                        logger.info(f"EchoVault: Retrieve operation successful for hash {test_hash}.")
                        break
            
            if not found_test_item:
                logger.error(f"EchoVault: Retrieve operation failed to find test item {test_hash}. Results: {retrieved_results}")
                # Attempt to delete anyway in case it was stored but not retrieved
                await storage.delete(test_hash)
                return False, f"EchoVault: Retrieve operation failed to find test item {test_hash}"

            # Test delete
            logger.info(f"EchoVault: Testing delete operation for hash {test_hash}...")
            delete_success, delete_msg = await storage.delete(test_hash)
            if not delete_success:
                # If delete fails, it might be because the item wasn't found (e.g., if retrieve failed)
                # or due to an actual delete error.
                logger.error(f"EchoVault: Delete operation failed: {delete_msg}")
                return False, f"EchoVault: Delete operation failed for {test_hash}: {delete_msg}"
            logger.info(f"EchoVault: Delete operation successful for hash {test_hash}.")
            
            return True, "EchoVault: Database validation successful"
        except Exception as e:
            logger.error(f"EchoVault: Database validation failed: {str(e)}", exc_info=True)
            return False, f"EchoVault: Database validation failed: {str(e)}"
    else:
        return False, f"Unsupported storage type: {type(storage)}"

def get_database_stats(storage) -> Dict[str, Any]:
    logger.info(f"Getting database stats for storage type: {type(storage)}")
    if isinstance(storage, ChromaMemoryStorage):
        try:
            count = storage.collection.count()
            collection_info = {
                "total_memories": count,
                "embedding_function": storage.embedding_function.__class__.__name__,
                "metadata": storage.collection.metadata
            }
            
            db_path = storage.path
            size = 0
            if db_path and os.path.exists(db_path): # Check if path exists
                for root, dirs, files in os.walk(db_path):
                    size += sum(os.path.getsize(os.path.join(root, name)) for name in files)
            
            storage_info = {
                "path": db_path,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2) if size > 0 else 0
            }
            
            return {
                "collection": collection_info,
                "storage": storage_info,
                "status": "healthy",
                "type": "ChromaDB"
            }
        except Exception as e:
            logger.error(f"ChromaDB: Error getting database stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "type": "ChromaDB"
            }

    elif isinstance(storage, EchoVaultStorage):
        try:
            # For EchoVault, stats are more complex.
            # We'll report basic health and component status.
            # Actual counts would require querying Neon/Qdrant.
            
            # Neon stats (e.g., connection status, basic table count if feasible)
            neon_status = "unknown"
            neon_error = None
            if storage.neon_client and hasattr(storage.neon_client, 'pool') and storage.neon_client.pool:
                # Simplified check: if pool exists, assume basic setup is done.
                # A full check requires an async call, which is problematic from a sync function.
                neon_status = "pool_initialized" 
                # To do a real check, get_database_stats would need to be async
                # or run the check in a separate thread.
                # try:
                #     async def check_neon_sync():
                #         async with storage.neon_client.pool.acquire() as conn:
                #             await conn.fetchval("SELECT 1")
                #     # This is still tricky. For now, we simplify.
                #     # asyncio.run(check_neon_sync()) # This will fail if loop is running
                #     neon_status = "connected_pending_async_verification"
                # except Exception as ne:
                #     neon_status = "error_during_sync_check_attempt"
                #     neon_error = str(ne)
            else:
                neon_status = "not_initialized_or_no_pool"

            # Vector store (Qdrant) stats
            vector_store_status = "unknown"
            vector_store_error = None
            if storage.vector_store and hasattr(storage.vector_store, 'client') and storage.vector_store.client:
                try:
                    # Example: try to get collection info for Qdrant
                    # This depends on the actual methods available in VectorStoreClient
                    # For now, let's assume a simple health check like ping or getting collection info
                    # If VectorStoreClient has a method like `get_collection_info()`
                    # await storage.vector_store.get_collection_info() 
                    # For now, just check if client is there
                    vector_store_status = "client_present" 
                    # Ideally, you'd make a call to Qdrant to check its health
                    # e.g., if `self.vector_store.client.health_check()` exists
                except Exception as qe:
                    vector_store_status = "error"
                    vector_store_error = str(qe)
                    logger.error(f"EchoVault: Vector store client check failed: {qe}")
            else:
                vector_store_status = "not_initialized_or_no_client"
            
            # Blob store (R2) stats
            blob_store_status = "unknown"
            blob_store_error = None
            if storage.blob_store and hasattr(storage.blob_store, 's3_client') and storage.blob_store.s3_client:
                try:
                    # Example: try to list buckets or a specific bucket for R2
                    # This depends on the actual methods available in BlobStoreClient
                    # await storage.blob_store.s3_client.head_bucket(Bucket=storage.blob_store.bucket_name)
                    blob_store_status = "client_present"
                    # Ideally, you'd make a call to R2 to check its health
                except Exception as r2e:
                    blob_store_status = "error"
                    blob_store_error = str(r2e)
                    logger.error(f"EchoVault: Blob store client check failed: {r2e}")
            else:
                blob_store_status = "not_initialized_or_no_client"

            overall_status = "healthy"
            if neon_status == "error" or vector_store_status == "error" or blob_store_status == "error":
                overall_status = "degraded"
            if not storage._is_initialized:
                overall_status = "not_fully_initialized"

            return {
                "status": overall_status,
                "type": "EchoVault",
                "components": {
                    "neon": {"status": neon_status, "error": neon_error},
                    "vector_store": {"status": vector_store_status, "error": vector_store_error},
                    "blob_store": {"status": blob_store_status, "error": blob_store_error},
                },
                "is_initialized": storage._is_initialized
            }
        except Exception as e:
            logger.error(f"EchoVault: Error getting database stats: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "type": "EchoVault"
            }
    else:
        logger.warning(f"Unsupported storage type for stats: {type(storage)}")
        return {
            "status": "error",
            "error": f"Unsupported storage type: {type(storage)}",
            "type": "Unknown"
        }

async def repair_database(storage) -> Tuple[bool, str]:
    logger.info(f"Attempting to repair database with storage type: {type(storage)}")
    if isinstance(storage, ChromaMemoryStorage):
        try:
            # Validate current state
            is_valid, message = await validate_database(storage)
            if is_valid:
                return True, "ChromaDB: Database is already healthy"
            
            logger.warning(f"ChromaDB: Database not healthy, attempting repair. Validation message: {message}")
            
            # Backup current embeddings and metadata
            existing_data = None
            try:
                existing_data = storage.collection.get() # This might fail if collection is corrupted
            except Exception as backup_error:
                logger.error(f"ChromaDB: Could not backup existing data: {str(backup_error)}")
            
            # Recreate collection
            logger.info(f"ChromaDB: Recreating collection '{storage.collection_name}'")
            storage.client.delete_collection(storage.collection_name)
            storage.collection = storage.client.create_collection(
                name=storage.collection_name, # Use the original collection name
                metadata={"hnsw:space": "cosine"}, # Or original metadata
                embedding_function=storage.embedding_function
            )
            logger.info(f"ChromaDB: Collection '{storage.collection_name}' recreated.")
            
            # Restore data if backup was successful
            if existing_data and existing_data.get("ids"): # Check if 'ids' key exists and is not empty
                logger.info(f"ChromaDB: Restoring {len(existing_data['ids'])} items to collection.")
                # ChromaDB expects documents, metadatas, ids to be lists of the same length.
                # Ensure embeddings are not passed if they are auto-generated.
                storage.collection.add(
                    documents=existing_data["documents"],
                    metadatas=existing_data["metadatas"],
                    ids=existing_data["ids"]
                    # embeddings=existing_data.get("embeddings") # Only if embeddings were stored and not auto-generated
                )
                logger.info("ChromaDB: Data restoration complete.")
            elif existing_data:
                logger.info("ChromaDB: Backup existed but contained no items to restore.")
            else:
                logger.info("ChromaDB: No data backed up, starting with a fresh collection.")

            # Validate repair
            is_valid, message = await validate_database(storage)
            if is_valid:
                return True, "ChromaDB: Database successfully repaired"
            else:
                logger.error(f"ChromaDB: Repair failed after recreating collection. Validation message: {message}")
                return False, f"ChromaDB: Repair failed: {message}"
                
        except Exception as e:
            logger.error(f"ChromaDB: Error repairing database: {str(e)}", exc_info=True)
            return False, f"ChromaDB: Error repairing database: {str(e)}"

    elif isinstance(storage, EchoVaultStorage):
        try:
            logger.info("EchoVault: Attempting repair by re-initializing storage.")
            # For EchoVault, "repair" might mean re-initializing clients
            # and ensuring connections are re-established.
            if not storage._is_initialized:
                logger.info("EchoVault: Storage was not initialized. Initializing now.")
            else:
                logger.info("EchoVault: Storage was initialized. Re-initializing.")
            
            # Reset initialization status and re-initialize
            storage._is_initialized = False 
            await storage.initialize()
            
            if not storage._is_initialized:
                logger.error("EchoVault: Failed to re-initialize during repair.")
                return False, "EchoVault: Failed to re-initialize components during repair."

            logger.info("EchoVault: Re-initialization complete. Validating database...")
            is_valid, message = await validate_database(storage)
            if is_valid:
                return True, "EchoVault: Database successfully repaired/re-initialized."
            else:
                logger.error(f"EchoVault: Validation failed after repair attempt: {message}")
                return False, f"EchoVault: Validation failed after repair attempt: {message}"
        except Exception as e:
            logger.error(f"EchoVault: Error repairing database: {str(e)}", exc_info=True)
            return False, f"EchoVault: Error repairing database: {str(e)}"
    else:
        return False, f"Unsupported storage type for repair: {type(storage)}"