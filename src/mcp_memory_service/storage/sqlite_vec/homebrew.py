"""
SQLite-vec storage backend with Homebrew PyTorch integration for MCP Memory Service.
"""

import os
import json
import logging
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .sqlite_vec import SqliteVecMemoryStorage
from ..homebrew_integration import get_homebrew_model

logger = logging.getLogger(__name__)

class HomebrewSqliteVecMemoryStorage(SqliteVecMemoryStorage):
    """
    SQLite-vec based memory storage implementation with Homebrew PyTorch integration.
    
    This subclass extends the standard SqliteVecMemoryStorage to use the Homebrew PyTorch
    installation for generating embeddings.
    """
    
    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize storage with Homebrew PyTorch."""
        # Call parent initialization
        super().__init__(db_path, embedding_model)
        self._homebrew_initialized = False
        
        # Set this for compatibility with checks that use it
        self.model = None
        
    async def initialize(self):
        """Initialize the database and embedding model."""
        # First, initialize the database
        await super().initialize()
        
        # Then, initialize the Homebrew PyTorch model
        await self._initialize_homebrew_model()
        
    async def _initialize_homebrew_model(self):
        """Initialize the Homebrew PyTorch model."""
        try:
            if os.environ.get('MCP_MEMORY_USE_HOMEBREW_PYTORCH', '').lower() in ('1', 'true', 'yes'):
                logger.info("Initializing Homebrew PyTorch model")
                
                # Create the Homebrew model
                model = get_homebrew_model(self.embedding_model_name)
                
                if model and model.initialized:
                    # Set model properties
                    self.model = model
                    self.embedding_model = model  # For compatibility
                    self.embedding_dimension = model.dimension
                    self._homebrew_initialized = True
                    logger.info(f"Homebrew PyTorch model initialized: {self.embedding_model_name}")
                else:
                    logger.warning("Failed to initialize Homebrew PyTorch model, falling back to default")
                    await self._initialize_embedding_model()
            else:
                logger.info("Homebrew PyTorch integration disabled")
                # Use standard embedding model initialization
                await self._initialize_embedding_model()
                
        except Exception as e:
            logger.error(f"Error initializing Homebrew model: {e}")
            # Fall back to standard embedding model initialization
            await self._initialize_embedding_model()
            
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Homebrew PyTorch if available."""
        # If Homebrew model is initialized, use it
        if self._homebrew_initialized and self.model:
            try:
                # Generate embedding using Homebrew model
                embedding = self.model.encode(text)
                
                # Convert to list if it's a numpy array
                if isinstance(embedding, np.ndarray):
                    return embedding.tolist()
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding with Homebrew model: {e}")
                return [0.0] * self.embedding_dimension
        
        # Otherwise, fall back to the parent implementation
        return super()._generate_embedding(text)