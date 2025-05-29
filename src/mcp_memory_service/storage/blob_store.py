"""
Blob Storage Client for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This module provides a client for storing large content blobs in Cloudflare R2,
with functionality for generating presigned URLs.
"""

import os
import logging
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, Union
from io import BytesIO

logger = logging.getLogger(__name__)

class BlobStoreClient:
    """
    Client for storing large content blobs in Cloudflare R2.
    Provides methods for storing and retrieving blobs, with presigned URL generation.
    """
    
    def __init__(self):
        """Initialize the blob store client."""
        self.r2_endpoint = os.environ.get("R2_ENDPOINT")
        self.r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
        self.r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self.r2_bucket = os.environ.get("R2_BUCKET")
        self.blob_threshold = int(os.environ.get("BLOB_THRESHOLD", "32768"))  # 32 KB default
        self.url_expiry = int(os.environ.get("PRESIGN_EXPIRY_SECONDS", "3600"))  # 1 hour default
        self.client = None
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize the blob store client."""
        if self._is_initialized:
            return
            
        if not all([self.r2_endpoint, self.r2_access_key, self.r2_secret_key, self.r2_bucket]):
            logger.warning("R2 credentials not fully configured, blob storage will be disabled")
            return
            
        try:
            import boto3
            from botocore.config import Config
            
            # Configure boto3 client for Cloudflare R2
            self.client = boto3.client(
                service_name='s3',
                endpoint_url=self.r2_endpoint,
                aws_access_key_id=self.r2_access_key,
                aws_secret_access_key=self.r2_secret_key,
                config=Config(signature_version='s3v4')
            )
            
            # Verify bucket exists
            try:
                self.client.head_bucket(Bucket=self.r2_bucket)
                logger.info(f"Connected to R2 bucket: {self.r2_bucket}")
                self._is_initialized = True
            except Exception as e:
                logger.error(f"Failed to connect to R2 bucket: {e}")
                self.client = None
                
        except ImportError:
            logger.warning("boto3 not installed, blob storage will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize R2 client: {e}")
    
    def is_configured(self) -> bool:
        """
        Check if blob storage is properly configured.
        
        Returns:
            True if blob storage is configured and initialized
        """
        return self._is_initialized and self.client is not None
    
    async def save_if_large(self, content: str, content_hash: str) -> Tuple[str, Optional[str]]:
        """
        Save content to blob storage if it exceeds the size threshold.
        
        Args:
            content: Content to potentially store in blob storage
            content_hash: Hash identifier for the content
            
        Returns:
            Tuple of (content, payload_url) where payload_url is None if content is stored inline
        """
        if not self.is_configured():
            await self.initialize()
            if not self.is_configured():
                return content, None
        
        # Check if content exceeds the threshold
        if len(content.encode('utf-8')) <= self.blob_threshold:
            return content, None
        
        try:
            # Generate a key for the blob
            key = f"memories/{content_hash}.txt"
            
            # Upload the blob
            self.client.upload_fileobj(
                BytesIO(content.encode('utf-8')),
                self.r2_bucket,
                key,
                ExtraArgs={'ContentType': 'text/plain; charset=utf-8'}
            )
            
            # Generate a presigned URL for the blob
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.r2_bucket,
                    'Key': key
                },
                ExpiresIn=self.url_expiry
            )
            
            logger.info(f"Stored large content in R2: {key}")
            
            # Return a summary of the content and the URL
            content_preview = content[:100] + "..." if len(content) > 100 else content
            return content_preview, key
            
        except Exception as e:
            logger.error(f"Failed to store content in R2: {e}")
            return content, None
    
    async def generate_presigned_url(self, key: str) -> Optional[str]:
        """
        Generate a presigned URL for a blob.
        
        Args:
            key: Key of the blob in R2
            
        Returns:
            Presigned URL or None if generation failed
        """
        if not self.is_configured():
            await self.initialize()
            if not self.is_configured():
                return None
        
        try:
            # Generate a presigned URL for the blob
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.r2_bucket,
                    'Key': key
                },
                ExpiresIn=self.url_expiry
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    async def retrieve_content(self, key: str) -> Optional[str]:
        """
        Retrieve content from blob storage.
        
        Args:
            key: Key of the blob in R2
            
        Returns:
            Content or None if retrieval failed
        """
        if not self.is_configured():
            await self.initialize()
            if not self.is_configured():
                return None
        
        try:
            # Get the object from R2
            response = self.client.get_object(
                Bucket=self.r2_bucket,
                Key=key
            )
            
            # Read and decode the content
            content = response['Body'].read().decode('utf-8')
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to retrieve content from R2: {e}")
            return None
    
    async def delete_blob(self, key: str) -> bool:
        """
        Delete a blob from storage.
        
        Args:
            key: Key of the blob in R2
            
        Returns:
            True if deletion was successful
        """
        if not self.is_configured():
            await self.initialize()
            if not self.is_configured():
                return False
        
        try:
            # Delete the object from R2
            self.client.delete_object(
                Bucket=self.r2_bucket,
                Key=key
            )
            
            logger.info(f"Deleted blob from R2: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete blob from R2: {e}")
            return False
    
    async def batch_delete_blobs(self, keys: list) -> bool:
        """
        Delete multiple blobs from storage.
        
        Args:
            keys: List of keys to delete
            
        Returns:
            True if all deletions were successful
        """
        if not self.is_configured():
            await self.initialize()
            if not self.is_configured():
                return False
        
        if not keys:
            return True
        
        try:
            # Delete objects in batches of 1000 (S3 limit)
            for i in range(0, len(keys), 1000):
                batch = keys[i:i+1000]
                
                # Format for delete operation
                objects = [{'Key': key} for key in batch]
                
                # Delete the objects
                self.client.delete_objects(
                    Bucket=self.r2_bucket,
                    Delete={
                        'Objects': objects,
                        'Quiet': True
                    }
                )
            
            logger.info(f"Batch deleted {len(keys)} blobs from R2")
            return True
            
        except Exception as e:
            logger.error(f"Failed to batch delete blobs from R2: {e}")
            return False