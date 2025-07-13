"""
Homebrew PyTorch integration for MCP Memory Service.

This module provides seamless integration with Homebrew PyTorch installation
for generating embeddings in the MCP Memory Service.
"""

import os
import sys
import subprocess
import json
import tempfile
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class HomebrewPyTorchModel:
    """A wrapper class for Homebrew PyTorch that mimics the SentenceTransformer interface."""
    
    def __init__(self, model_name: str):
        """Initialize the model with a model name."""
        self.model_name = model_name
        self._model_card_vars = {'modelname': f'homebrew-{model_name}'}
        self.initialized = False
        self.dimension = 384  # Default for common models
        
        # Try to detect Homebrew PyTorch
        try:
            # Check if pytorch is in the brew list first
            list_result = subprocess.run(
                ['brew', 'list'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if 'pytorch' not in list_result.stdout:
                logger.error("PyTorch not found in brew list. Please install PyTorch with: brew install pytorch")
                self.initialized = False
                return
                
            # Get the path to PyTorch
            result = subprocess.run(
                ['brew', '--prefix', 'pytorch'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to get PyTorch prefix: {result.stderr}")
                self.initialized = False
                return
                
            self.pytorch_prefix = result.stdout.strip()
            self.homebrew_python = f"{self.pytorch_prefix}/libexec/bin/python3"
            
            # Check if the Python executable exists
            if not os.path.exists(self.homebrew_python):
                logger.error(f"Homebrew Python not found at {self.homebrew_python}")
                self.initialized = False
                return
                
            # Log success
            logger.info(f"Found Homebrew PyTorch at {self.pytorch_prefix}")
            logger.info(f"Using Python interpreter at {self.homebrew_python}")
            
            # Check if the model is available
            self._check_model()
            
            if self.initialized:
                logger.info(f"Homebrew PyTorch integration initialized successfully for model: {model_name}")
            else:
                logger.warning(f"Failed to initialize Homebrew PyTorch integration for model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Homebrew PyTorch integration: {e}")
            self.initialized = False
    
    def _check_model(self):
        """Check if the model is available in Homebrew Python."""
        # First check if sentence_transformers is installed
        check_import_script = """
try:
    import sentence_transformers
    print("sentence_transformers is installed")
    exit(0)
except ImportError as e:
    print(f"Error: {e}")
    exit(1)
"""
        
        try:
            # Check if sentence_transformers is installed
            import_result = subprocess.run(
                [self.homebrew_python, "-c", check_import_script],
                capture_output=True,
                text=True,
                timeout=30  # Add timeout to prevent hanging
            )
            
            if import_result.returncode != 0:
                logger.error(f"sentence_transformers not installed in Homebrew Python: {import_result.stderr}")
                logger.info("Attempting to install sentence_transformers...")
                
                # Try to install sentence_transformers
                install_result = subprocess.run(
                    [self.homebrew_python, "-m", "pip", "install", "sentence-transformers"],
                    capture_output=True,
                    text=True,
                    timeout=120  # Pip install can take a while
                )
                
                if install_result.returncode != 0:
                    logger.error(f"Failed to install sentence_transformers: {install_result.stderr}")
                    self.initialized = False
                    return
                else:
                    logger.info("Successfully installed sentence_transformers")
            else:
                logger.info("sentence_transformers is already installed")
            
            # Now check if the model can be loaded
            check_script = f"""
import os
import sys
try:
    import torch
    import sentence_transformers
    print("Loading model {self.model_name}...")
    model = sentence_transformers.SentenceTransformer('{self.model_name}')
    # Get embedding dimension
    print("Generating test embedding...")
    test_embedding = model.encode(['test'])
    print(test_embedding.shape[1])
    sys.exit(0)
except Exception as e:
    print(f"Error: {{str(e)}}", file=sys.stderr)
    sys.exit(1)
"""
            
            # Run the model check script
            logger.info(f"Checking if model {self.model_name} is available...")
            result = subprocess.run(
                [self.homebrew_python, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=60  # Model loading can take a while
            )
            
            if result.returncode == 0:
                self.initialized = True
                try:
                    self.dimension = int(result.stdout.strip().split('\n')[-1])
                    logger.info(f"Homebrew model initialized with dimension {self.dimension}")
                except (ValueError, IndexError) as e:
                    # Default to 384 if dimension parsing fails
                    logger.warning(f"Could not parse dimension from output: {result.stdout}. Using default 384.")
                    logger.warning(f"Error: {e}")
                    self.dimension = 384
            else:
                logger.error(f"Failed to initialize model: {result.stderr}")
                logger.error(f"Command output: {result.stdout}")
                self.initialized = False
                
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout while checking model: {e}")
            self.initialized = False
        except Exception as e:
            logger.error(f"Error checking model: {e}")
            self.initialized = False
    
    def encode(self, texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False):
        """Generate embeddings for the input texts."""
        if not self.initialized:
            logger.warning("Model not initialized, returning zero embeddings")
            if isinstance(texts, str):
                return np.zeros(self.dimension)
            else:
                return np.zeros((len(texts), self.dimension))
        
        # Convert a single string to a list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        if not texts:
            return np.zeros((0, self.dimension))
        
        # Write texts to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(texts, f)
            temp_file = f.name
        
        # Embedding script
        embed_script = f"""
import os
import sys
import json
import numpy as np

try:
    import torch
    import sentence_transformers
    
    # Load the model
    model = sentence_transformers.SentenceTransformer('{self.model_name}')
    
    # Load texts from file
    with open('{temp_file}', 'r') as f:
        texts = json.load(f)
    
    # Generate embeddings
    embeddings = model.encode(texts, batch_size={batch_size}, show_progress_bar=False)
    
    # Save embeddings to file
    np.save('{temp_file}.npy', embeddings)
    print("Embeddings generated successfully")
    sys.exit(0)
except Exception as e:
    print(f"Error: {{str(e)}}", file=sys.stderr)
    sys.exit(1)
"""
        
        try:
            # Run the embedding script
            result = subprocess.run(
                [self.homebrew_python, "-c", embed_script],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Load the embeddings
                embeddings = np.load(f"{temp_file}.npy")
                logger.debug(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
                
                # Clean up temporary files
                os.unlink(temp_file)
                os.unlink(f"{temp_file}.npy")
                
                # Return a single vector if a single text was given
                if single_text:
                    return embeddings[0]
                return embeddings
            else:
                logger.error(f"Error generating embeddings: {result.stderr}")
                # Clean up temporary file
                os.unlink(temp_file)
                if single_text:
                    return np.zeros(self.dimension)
                return np.zeros((len(texts), self.dimension))
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if single_text:
                return np.zeros(self.dimension)
            return np.zeros((len(texts), self.dimension))

def get_homebrew_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[HomebrewPyTorchModel]:
    """Get a Homebrew PyTorch model instance if available."""
    try:
        model = HomebrewPyTorchModel(model_name)
        if model.initialized:
            return model
        return None
    except Exception as e:
        logger.error(f"Error creating Homebrew model: {e}")
        return None

def patch_storage(storage):
    """Patch a storage instance to use Homebrew PyTorch for embeddings."""
    try:
        # Check if we should use Homebrew PyTorch
        if os.environ.get('MCP_MEMORY_USE_HOMEBREW_PYTORCH', '').lower() in ('1', 'true', 'yes'):
            logger.info("Patching storage to use Homebrew PyTorch embeddings")
            
            # Get the model name from storage if available
            model_name = getattr(storage, 'embedding_model_name', "all-MiniLM-L6-v2")
            
            # Create the Homebrew model
            model = get_homebrew_model(model_name)
            
            if model and model.initialized:
                # Patch the storage instance
                storage.model = model
                # Keep these for compatibility
                storage.embedding_model = model
                # Set additional properties needed for compatibility
                storage.embedding_dimension = model.dimension
                logger.info(f"Storage patched to use Homebrew PyTorch model: {model_name}")
                return True
            else:
                logger.warning("Failed to initialize Homebrew PyTorch model")
                return False
        return False
    except Exception as e:
        logger.error(f"Error patching storage: {e}")
        return False
