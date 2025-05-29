"""
Authentication utilities for EchoVault Memory Service.
Provides JWT authentication for API access.
"""

import os
import time
import logging
import json
from typing import Dict, Any, Optional, Callable, Awaitable
import jwt

logger = logging.getLogger(__name__)

class AuthError(Exception):
    """Authentication error."""
    pass

async def verify_jwt(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token and return the payload.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Decoded token payload
        
    Raises:
        AuthError: If token is invalid or expired
    """
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        logger.warning("JWT_SECRET not set, authentication disabled")
        return {"sub": "anonymous", "authenticated": False}
    
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {str(e)}")

def generate_jwt(subject: str, expiry_seconds: int = 3600) -> str:
    """
    Generate a JWT token.
    
    Args:
        subject: Subject of the token (usually user ID)
        expiry_seconds: Token expiry time in seconds
        
    Returns:
        JWT token
        
    Raises:
        ValueError: If JWT_SECRET is not set
    """
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise ValueError("JWT_SECRET not set")
    
    payload = {
        "sub": subject,
        "iat": int(time.time()),
        "exp": int(time.time()) + expiry_seconds
    }
    
    return jwt.encode(payload, secret, algorithm="HS256")

def auth_required(handler: Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Any]]):
    """
    Decorator to require authentication for a handler.
    
    Args:
        handler: Handler function to wrap
        
    Returns:
        Wrapped handler function that checks authentication
    """
    async def wrapped_handler(params: Dict[str, Any], context: Dict[str, Any]) -> Any:
        # Check if authentication is enabled
        if not os.environ.get("JWT_SECRET"):
            # Authentication disabled, proceed
            return await handler(params, context)
        
        # Get token from context
        token = context.get("token")
        if not token:
            raise AuthError("Authentication required")
        
        try:
            # Verify token
            payload = await verify_jwt(token)
            
            # Add user info to context
            context["user"] = payload.get("sub")
            
            # Call handler
            return await handler(params, context)
        except AuthError as e:
            raise AuthError(f"Authentication failed: {str(e)}")
    
    return wrapped_handler

def extract_token_from_header(header: Optional[str]) -> Optional[str]:
    """
    Extract JWT token from Authorization header.
    
    Args:
        header: Authorization header value
        
    Returns:
        JWT token or None if not found
    """
    if not header:
        return None
    
    parts = header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    
    return parts[1]