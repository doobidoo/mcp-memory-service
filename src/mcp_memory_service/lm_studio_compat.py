"""
LM Studio compatibility patch for MCP Memory Service.

This module provides a monkey patch to handle LM Studio's non-standard
'notifications/cancelled' message that isn't part of the standard MCP protocol.
"""

import logging
import sys
import platform
from typing import Any, Union

logger = logging.getLogger(__name__)

def add_windows_timeout_handling():
    """
    Add Windows-specific timeout and error handling.
    """
    if platform.system() != "Windows":
        return
    
    try:
        # Add better timeout handling for Windows
        import signal
        import asyncio
        
        def timeout_handler(signum, frame):
            logger.warning("Server received timeout signal - attempting graceful shutdown")
            print("Server timeout detected - shutting down gracefully", file=sys.stderr, flush=True)
            sys.exit(0)
        
        # Only set up signal handlers if they're available
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, timeout_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, timeout_handler)
            
        logger.info("Added Windows-specific timeout handling")
        
    except Exception as e:
        logger.debug(f"Could not set up Windows timeout handling: {e}")
        # Not critical, continue without it

def create_cancelled_notification_class():
    """Create a proper CancelledNotification class if it doesn't exist."""
    from pydantic import BaseModel, Field

    class CancelledNotificationParams(BaseModel):
        """Parameters for cancelled notification."""
        requestId: Any = Field(default=None, alias="requestId")
        reason: str = Field(default="Operation cancelled")
    
    class CancelledNotification(BaseModel):
        """Cancelled notification that matches MCP expectations."""
        method: str = Field(default="notifications/cancelled")
        params: CancelledNotificationParams = Field(default_factory=CancelledNotificationParams)
        
        @property
        def root(self):
            """Provide root attribute for compatibility."""
            return self
    
    return CancelledNotification

def patch_mcp_for_lm_studio():
    """
    Apply monkey patches to MCP library to handle LM Studio's non-standard notifications.
    This new approach patches at multiple levels to ensure the cancelled notification is handled.
    """
    success = False
    
    try:
        import mcp.shared.session as session_module
        from pydantic_core import ValidationError
        
        # Create or get the CancelledNotification class
        if hasattr(session_module, 'CancelledNotification'):
            CancelledNotification = session_module.CancelledNotification
        else:
            CancelledNotification = create_cancelled_notification_class()
            session_module.CancelledNotification = CancelledNotification
            logger.info("Created CancelledNotification class")
        
        # Patch 1: Override ClientNotification to handle cancelled notifications
        if hasattr(session_module, 'ClientNotification'):
            original_client_notification = session_module.ClientNotification
            
            # Store the original __or__ operator if it exists
            original_or = getattr(original_client_notification, '__or__', None)
            
            # Create a new union type that includes CancelledNotification (unused, kept for documentation)
            if original_or:
                # Add CancelledNotification to the union
                _patched_union = Union[original_client_notification, CancelledNotification]
            else:
                _patched_union = original_client_notification
            del _patched_union  # Variable is only created for type documentation purposes
            
            # Store original model_validate
            original_validate = original_client_notification.model_validate
            
            # Create patched validation function with correct classmethod signature
            @classmethod
            def patched_validate(cls, obj, *args, **kwargs):
                """Enhanced validation that handles cancelled notifications."""
                logger.debug(f"Patched validate called with: {type(obj)} - {obj}")
                
                if isinstance(obj, dict):
                    method = obj.get('method', '')
                    if method == 'notifications/cancelled':
                        params = obj.get('params', {})
                        logger.info(f"[PATCH] PATCH INTERCEPTED cancelled notification: {params.get('reason', 'Unknown')}")
                        # Return a proper CancelledNotification instance with structured params
                        notification = CancelledNotification()
                        if params:
                            notification.params.requestId = params.get('requestId')
                            notification.params.reason = params.get('reason', 'Operation cancelled')
                        return notification
                
                # Try original validation
                try:
                    return original_validate.__func__(cls, obj, *args, **kwargs)
                except ValidationError as e:
                    # If it's a cancelled notification error, handle it
                    if 'notifications/cancelled' in str(e):
                        logger.info("[PATCH] PATCH CAUGHT cancelled notification in validation error")
                        notification = CancelledNotification()
                        if isinstance(obj, dict) and 'params' in obj:
                            params = obj['params']
                            notification.params.requestId = params.get('requestId')
                            notification.params.reason = params.get('reason', 'Operation cancelled')
                        return notification
                    raise
            
            # Apply the patched validation
            original_client_notification.model_validate = patched_validate
            logger.info("[PATCH] Applied NEW LM Studio patch to ClientNotification.model_validate v2.0")
            success = True
        
        # Patch 2: Patch BaseSession to handle errors at the session level
        from mcp.shared.session import BaseSession
        
        if hasattr(BaseSession, '_handle_notification'):
            original_handle = BaseSession._handle_notification
            
            async def patched_handle_notification(self, notification):
                """Handle notifications including cancelled ones."""
                # Check if this is a CancelledNotification
                if hasattr(notification, 'method') and notification.method == 'notifications/cancelled':
                    logger.info("Handling cancelled notification - ignoring")
                    return None  # Just ignore it

                # Otherwise handle normally
                return await original_handle(self, notification)
            
            BaseSession._handle_notification = patched_handle_notification
            logger.info("[PATCH] Applied NEW patch to BaseSession._handle_notification v2.0")
            success = True
        
        # Patch 3: As a last resort, patch the session's _receive_loop
        if hasattr(BaseSession, '_receive_loop'):
            original_loop = BaseSession._receive_loop
            
            async def patched_loop(self):
                """Robust receive loop that continues on cancelled notifications."""
                try:
                    return await original_loop(self)
                except Exception as e:
                    # Check for the specific error pattern
                    error_str = str(e)
                    if ('notifications/cancelled' in error_str or
                        ('ValidationError' in str(type(e).__name__) and 
                         'literal_error' in error_str)):
                        logger.info("Suppressed cancelled notification error in receive loop")
                        # Don't propagate the error - this prevents the TaskGroup from failing
                        return None
                    # Re-raise other errors
                    raise
            
            BaseSession._receive_loop = patched_loop
            logger.info("[PATCH] Applied NEW fallback patch to BaseSession._receive_loop v2.0")
            success = True
        
    except ImportError as e:
        logger.warning(f"Could not import MCP modules: {e}")
        return patch_alternative_approach()
    except Exception as e:
        logger.error(f"Error applying LM Studio compatibility patch: {e}")
    
    if not success:
        logger.warning("Primary patching failed, trying alternative approach")
        return patch_alternative_approach()
    
    return success


def patch_alternative_approach():
    """
    Alternative patching approach that modifies validation at a lower level.
    """
    try:
        # Try to patch pydantic validation directly
        import pydantic_core
        from pydantic_core import ValidationError
        
        original_validation_error = ValidationError.__init__
        
        def patched_validation_error_init(self, *args, **kwargs):
            """Suppress cancelled notification validation errors."""
            # Check if this is about cancelled notifications
            if args and 'notifications/cancelled' in str(args[0]):
                logger.info("Suppressed ValidationError for cancelled notification")
                # Create a minimal error that won't cause issues
                self.errors = []
                return
            
            # Otherwise initialize normally
            original_validation_error(self, *args, **kwargs)
        
        ValidationError.__init__ = patched_validation_error_init
        logger.info("[PATCH] Applied NEW alternative patch to ValidationError v2.0")
        return True
        
    except Exception as e:
        logger.error(f"Alternative patch failed: {e}")
        return False