#\!/usr/bin/env python3
"""
A bridge script that provides embeddings from Homebrew PyTorch to MCP Memory Service.
"""
import os
import sys
import subprocess
import json
import tempfile
import time
import numpy as np
from pathlib import Path

class HomebrewPyTorchEmbedder:
    """Embedding provider using Homebrew PyTorch installation."""
    
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2"):
        """Initialize the embedder with the given model name."""
        self.model_name = model_name
        self.initialized = False
        self.dimension = 384  # Default for paraphrase-MiniLM-L3-v2
        
        # Get the Homebrew Python path
        result = subprocess.run(
            ['brew', '--prefix', 'pytorch'],
            capture_output=True,
            text=True,
            check=True
        )
        self.pytorch_prefix = result.stdout.strip()
        self.homebrew_python = f"{self.pytorch_prefix}/libexec/bin/python3"
        
        # Check if the model is available
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if the required model is available in Homebrew Python."""
        check_script = f"""
import os
import sys
try:
    import torch
    import sentence_transformers
    model = sentence_transformers.SentenceTransformer('{self.model_name}')
    # Get embedding dimension
    test_embedding = model.encode(['test'])
    print(test_embedding.shape[1])
    sys.exit(0)
except Exception as e:
    print(f"Error: {{str(e)}}", file=sys.stderr)
    sys.exit(1)
"""
        
        try:
            result = subprocess.run(
                [self.homebrew_python, "-c", check_script],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.initialized = True
                self.dimension = int(result.stdout.strip())
                print(f"Homebrew PyTorch embedding model initialized with dimension {self.dimension}")
            else:
                print(f"Failed to initialize embedding model: {result.stderr}")
        except Exception as e:
            print(f"Error checking model availability: {e}")
    
    def encode(self, texts, batch_size=32):
        """Generate embeddings for the given texts using Homebrew PyTorch."""
        if not self.initialized:
            print("Embedder not initialized")
            return np.zeros((len(texts), self.dimension))
        
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
                print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
                
                # Clean up temporary files
                os.unlink(temp_file)
                os.unlink(f"{temp_file}.npy")
                
                return embeddings
            else:
                print(f"Error generating embeddings: {result.stderr}")
                # Clean up temporary file
                os.unlink(temp_file)
                return np.zeros((len(texts), self.dimension))
        except Exception as e:
            print(f"Error encoding texts: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return np.zeros((len(texts), self.dimension))

# Create a singleton instance
homebrew_embedder = HomebrewPyTorchEmbedder()

# Export the encode function
def encode(texts, batch_size=32):
    """Generate embeddings for the given texts."""
    return homebrew_embedder.encode(texts, batch_size)

if __name__ == "__main__":
    # Test the embedder
    test_texts = [
        "This is a test sentence.",
        "Another test sentence for embedding.",
        "Let's see if this works with Homebrew PyTorch\!"
    ]
    
    embeddings = encode(test_texts)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Print the first embedding vector (truncated)
    print(f"First embedding: {embeddings[0][:5]}...")
