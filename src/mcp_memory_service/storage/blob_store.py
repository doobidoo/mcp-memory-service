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

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
except ImportError:
    # Allow the code to run even if boto3 is not installed,
    # is_configured() will handle the disabled state.
    boto3 = None 
    ClientError = None 
    NoCredentialsError = None
    PartialCredentialsError = None
    logger.info("boto3 library not found. R2 Blob Storage will be disabled.")

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
        if self._is_initialized:
            return
            
        logger.info("Attempting to initialize R2 BlobStoreClient.")
        logger.info(f"R2_ENDPOINT: {self.r2_endpoint}")
        logger.info(f"R2_BUCKET: {self.r2_bucket}")

        if self.r2_access_key:
            logger.info("R2_ACCESS_KEY_ID is set.")
        else:
            logger.warning("R2_ACCESS_KEY_ID is NOT set.")
        
        if self.r2_secret_key:
            logger.info("R2_SECRET_ACCESS_KEY is set (not logging the key itself).")
        else:
            logger.warning("R2_SECRET_ACCESS_KEY is NOT set.")

        if not all([self.r2_endpoint, self.r2_access_key, self.r2_secret_key, self.r2_bucket]):
            logger.error("R2 credentials/configuration not fully provided. Blob storage will be disabled.")
            return # Stop initialization if config is incomplete
            
        try:
            # Ensure boto3 was imported successfully at the module level
            if not boto3 or not ClientError: # Check if boto3 related imports failed
                logger.warning("boto3 library or its exceptions not available. Blob storage will be disabled.")
                self.client = None
                return

            logger.info("Attempting to create boto3 S3 client for R2...")
            self.client = boto3.client(
                service_name='s3',
                endpoint_url=self.r2_endpoint,
                aws_access_key_id=self.r2_access_key,
                aws_secret_access_key=self.r2_secret_key,
                config=Config(signature_version='s3v4')
            )
            logger.info("boto3 S3 client created. Verifying bucket existence...")
            
            self.client.head_bucket(Bucket=self.r2_bucket)
            logger.info(f"Successfully connected to R2 bucket: {self.r2_bucket}")
            self._is_initialized = True
            
        except (NoCredentialsError, PartialCredentialsError) as cred_err:
            logger.error(f"R2 Boto3 client credentials error: {type(cred_err).__name__} - {cred_err}", exc_info=True)
            logger.error("Please check R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables.")
            self.client = None # Ensure client is None on failure
        except ClientError as ce:
            error_code = ce.response.get('Error', {}).get('Code')
            error_message = ce.response.get('Error', {}).get('Message')
            request_id = ce.response.get('ResponseMetadata', {}).get('RequestId')
            host_id = ce.response.get('ResponseMetadata', {}).get('HostId')
            logger.error(f"R2 Boto3 client error during head_bucket: {type(ce).__name__} - Code: {error_code}, Message: {error_message}, RequestId: {request_id}, HostId: {host_id}", exc_info=True)
            logger.error(f"This could be due to incorrect R2_ENDPOINT, R2_BUCKET name '{self.r2_bucket}', or insufficient permissions for the R2 keys to access the bucket.")
            self.client = None
        except ImportError: # This case should ideally be caught by the module-level try-except for boto3
            logger.warning("boto3 library not installed (caught during client creation). Blob storage will be disabled.")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize R2 client with an unexpected error: {type(e).__name__} - {e}", exc_info=True)
            self.client = None
    
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
            await self.initialize() # Ensure initialization is attempted
            if not self.is_configured(): # Re-check after attempt
                logger.warning("R2 client not configured/initialized in save_if_large. Cannot save to R2.")
                return content, None
        
        if len(content.encode('utf-8')) <= self.blob_threshold:
            return content, None
        
        key = f"memories/{content_hash}.txt"
        logger.info(f"Attempting to store large content in R2. Bucket: '{self.r2_bucket}', Key: '{key}'")
        
        try:
            # Ensure ClientError is available if boto3 was imported
            if not ClientError: # Check if ClientError is None due to import failure
                 logger.error(f"Cannot save to R2: ClientError not defined (boto3 import issue). Bucket: {self.r2_bucket}, Key: {key}")
                 return content, None

            self.client.upload_fileobj(
                BytesIO(content.encode('utf-8')),
                self.r2_bucket,
                key,
                ExtraArgs={'ContentType': 'text/plain; charset=utf-8'}
            )
            # Presigned URL generation was removed from here in the original code, which is fine.
            # The key itself is returned.
            logger.info(f"Successfully stored large content in R2: Bucket '{self.r2_bucket}', Key '{key}'")
            
            content_preview = content[:100] + "..." if len(content) > 100 else content
            return content_preview, key # Return the key, not a presigned URL
            
        except ClientError as ce:
            error_code = ce.response.get('Error', {}).get('Code')
            error_message = ce.response.get('Error', {}).get('Message')
            request_id = ce.response.get('ResponseMetadata', {}).get('RequestId')
            logger.error(f"R2 client error during upload_fileobj (Bucket: {self.r2_bucket}, Key: {key}): {type(ce).__name__} - Code: {error_code}, Message: {error_message}, RequestId: {request_id}", exc_info=True)
            return content, None # Return original content if upload fails
        except Exception as e:
            logger.error(f"Failed to store content in R2 (Bucket: {self.r2_bucket}, Key: {key}) with an unexpected error: {type(e).__name__} - {e}", exc_info=True)
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
                logger.warning(f"R2 client not configured in generate_presigned_url for key '{key}'. Cannot generate URL.")
                return None
        
        try:
            # Ensure ClientError is available if boto3 was imported
            if not ClientError: # Check if ClientError is None due to import failure
                 logger.error(f"Cannot generate presigned URL: ClientError not defined (boto3 import issue). Key: {key}")
                 return None

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
        except ClientError as ce:
            error_code = ce.response.get('Error', {}).get('Code')
            error_message = ce.response.get('Error', {}).get('Message')
            request_id = ce.response.get('ResponseMetadata', {}).get('RequestId')
            logger.error(f"R2 client error during generate_presigned_url (Bucket: {self.r2_bucket}, Key: {key}): {type(ce).__name__} - Code: {error_code}, Message: {error_message}, RequestId: {request_id}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to generate presigned URL (Bucket: {self.r2_bucket}, Key: {key}) with an unexpected error: {type(e).__name__} - {e}", exc_info=True)
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
                logger.warning(f"R2 client not configured in retrieve_content for key '{key}'. Cannot retrieve from R2.")
                return None

        logger.info(f"Attempting to retrieve content from R2. Bucket: '{self.r2_bucket}', Key: '{key}'")
        try:
            # Ensure ClientError is available if boto3 was imported
            if not ClientError: # Check if ClientError is None due to import failure
                 logger.error(f"Cannot retrieve from R2: ClientError not defined (boto3 import issue). Key: {key}")
                 return None

            response = self.client.get_object(
                Bucket=self.r2_bucket,
                Key=key
            )
            content = response['Body'].read().decode('utf-8')
            logger.debug(f"Successfully retrieved content from R2. Bucket: '{self.r2_bucket}', Key: '{key}'")
            return content
        except ClientError as ce:
            error_code = ce.response.get('Error', {}).get('Code')
            error_message = ce.response.get('Error', {}).get('Message')
            request_id = ce.response.get('ResponseMetadata', {}).get('RequestId')
            logger.error(f"R2 client error during get_object (Bucket: {self.r2_bucket}, Key: {key}): {type(ce).__name__} - Code: {error_code}, Message: {error_message}, RequestId: {request_id}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve content from R2 (Bucket: {self.r2_bucket}, Key: {key}) with an unexpected error: {type(e).__name__} - {e}", exc_info=True)
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
                logger.warning(f"R2 client not configured in delete_blob for key '{key}'. Cannot delete from R2.")
                return False
        
        logger.info(f"Attempting to delete blob from R2. Bucket: '{self.r2_bucket}', Key: '{key}'")
        try:
            # Ensure ClientError is available if boto3 was imported
            if not ClientError: # Check if ClientError is None due to import failure
                 logger.error(f"Cannot delete from R2: ClientError not defined (boto3 import issue). Key: {key}")
                 return False

            self.client.delete_object(
                Bucket=self.r2_bucket,
                Key=key
            )
            logger.info(f"Successfully deleted blob from R2: Bucket '{self.r2_bucket}', Key '{key}'")
            return True
        except ClientError as ce:
            error_code = ce.response.get('Error', {}).get('Code')
            error_message = ce.response.get('Error', {}).get('Message')
            request_id = ce.response.get('ResponseMetadata', {}).get('RequestId')
            logger.error(f"R2 client error during delete_object (Bucket: {self.r2_bucket}, Key: {key}): {type(ce).__name__} - Code: {error_code}, Message: {error_message}, RequestId: {request_id}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to delete blob from R2 (Bucket: {self.r2_bucket}, Key: {key}) with an unexpected error: {type(e).__name__} - {e}", exc_info=True)
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
                logger.warning(f"R2 client not configured in batch_delete_blobs. Cannot delete from R2.")
                return False
        
        if not keys:
            return True
        
        logger.info(f"Attempting to batch delete {len(keys)} blobs from R2. Bucket: '{self.r2_bucket}'")
        try:
            # Ensure ClientError is available if boto3 was imported
            if not ClientError: # Check if ClientError is None due to import failure
                 logger.error(f"Cannot batch delete from R2: ClientError not defined (boto3 import issue).")
                 return False

            # Delete objects in batches of 1000 (S3 limit)
            all_successful = True
            for i in range(0, len(keys), 1000):
                batch = keys[i:i+1000]
                objects = [{'Key': key} for key in batch]
                
                logger.debug(f"Batch deleting {len(objects)} objects (Part {i//1000 + 1}) from R2 bucket '{self.r2_bucket}'.")
                response = self.client.delete_objects(
                    Bucket=self.r2_bucket,
                    Delete={
                        'Objects': objects,
                        'Quiet': False # Set to False to get error details if any
                    }
                )
                
                if response.get('Errors'):
                    all_successful = False
                    for error in response['Errors']:
                        logger.error(f"Error deleting object {error.get('Key')} from R2: {error.get('Code')} - {error.get('Message')}")
            
            if all_successful:
                logger.info(f"Successfully batch deleted {len(keys)} blobs from R2: Bucket '{self.r2_bucket}'")
            else:
                logger.warning(f"Batch delete from R2 encountered errors. Some objects may not have been deleted. Bucket: '{self.r2_bucket}'")
            return all_successful
            
        except ClientError as ce:
            error_code = ce.response.get('Error', {}).get('Code')
            error_message = ce.response.get('Error', {}).get('Message')
            request_id = ce.response.get('ResponseMetadata', {}).get('RequestId')
            logger.error(f"R2 client error during batch_delete_objects (Bucket: {self.r2_bucket}): {type(ce).__name__} - Code: {error_code}, Message: {error_message}, RequestId: {request_id}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to batch delete blobs from R2 (Bucket: {self.r2_bucket}) with an unexpected error: {type(e).__name__} - {e}", exc_info=True)
            return False