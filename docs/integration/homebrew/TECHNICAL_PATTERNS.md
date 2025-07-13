# Technical Patterns in MCP Memory Service Homebrew PyTorch Integration

This document captures the key technical patterns implemented during the Homebrew PyTorch integration with MCP Memory Service. These patterns represent reusable solutions to common problems in Python module integration, subprocess execution, and cross-environment code execution.

## Module Override Techniques

### Runtime Class Replacement Pattern

This pattern allows replacing a class in an already imported module with a different implementation:

```python
# Original implementation
from .storage.sqlite_vec import SqliteVecMemoryStorage
from .storage.homebrew_sqlite_vec import HomebrewSqliteVecMemoryStorage

# Replace the class in the module
sys.modules['mcp_memory_service.storage.sqlite_vec'].SqliteVecMemoryStorage = HomebrewSqliteVecMemoryStorage

# Verify the replacement worked
import importlib
storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
actual_class = storage_module.SqliteVecMemoryStorage
assert actual_class == HomebrewSqliteVecMemoryStorage
```

This technique allows for "hot-swapping" implementations without changing import statements throughout the codebase.

### Method Patching Pattern

For more granular control, we can patch individual methods while preserving the rest of the class:

```python
# Save the original method
original_method = ClassName.method_name

# Define a replacement method
def patched_method(self, *args, **kwargs):
    # Enhanced implementation that may call the original
    result = original_method(self, *args, **kwargs)
    # Additional functionality
    return modified_result

# Apply the patch
ClassName.method_name = patched_method
```

This approach is useful when only specific behaviors need to be modified while maintaining the overall class structure.

### Import Control Pattern

To ensure patches are applied before dependent modules are imported:

```python
# Apply patches first
apply_module_patches()

# Then import modules that depend on the patched behavior
from .dependent_module import dependent_functions
```

This pattern avoids issues where modules cache references to unpatched classes.

## Subprocess Execution Patterns

### Python Script Injection

For executing Python code in a different Python interpreter:

```python
def execute_in_homebrew_python(code_to_execute, input_data=None):
    # Create the script with proper error handling
    script = f"""
import sys
try:
    {code_to_execute}
    sys.exit(0)
except Exception as e:
    print(f"Error: {{str(e)}}", file=sys.stderr)
    sys.exit(1)
"""
    
    # Execute in the Homebrew Python interpreter
    result = subprocess.run(
        [homebrew_python_path, "-c", script],
        input=input_data,
        capture_output=True,
        text=True
    )
    
    # Handle results
    if result.returncode != 0:
        raise RuntimeError(f"Script execution failed: {result.stderr}")
    
    return result.stdout
```

This pattern allows executing arbitrary Python code in a separate Python environment while maintaining proper error handling.

### File-Based Data Exchange

For transferring data between different Python processes when the data is too complex for command-line arguments:

```python
def process_data_in_homebrew(data):
    # Write input data to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(data, input_file)
        input_path = input_file.name
    
    output_path = f"{input_path}.result"
    
    # Create script that reads input and writes output
    script = f"""
import json
import numpy as np

# Load input data
with open('{input_path}', 'r') as f:
    data = json.load(f)

# Process the data
result = process_function(data)

# Save the result
np.save('{output_path}', result)
"""
    
    # Execute script
    subprocess.run([homebrew_python_path, "-c", script], check=True)
    
    # Load the results
    result = np.load(output_path)
    
    # Clean up temporary files
    os.unlink(input_path)
    os.unlink(output_path)
    
    return result
```

This pattern is especially useful for exchanging numpy arrays or other complex data structures between processes.

### Environment Detection and Configuration

For detecting and configuring the Homebrew environment:

```python
def detect_homebrew_python():
    try:
        # Check if pytorch is installed via Homebrew
        result = subprocess.run(
            ['brew', 'list'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if 'pytorch' not in result.stdout:
            return None
        
        # Get path to Homebrew PyTorch
        prefix_result = subprocess.run(
            ['brew', '--prefix', 'pytorch'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if prefix_result.returncode != 0:
            return None
        
        pytorch_prefix = prefix_result.stdout.strip()
        homebrew_python = f"{pytorch_prefix}/libexec/bin/python3"
        
        # Verify the Python executable exists
        if not os.path.exists(homebrew_python):
            return None
        
        return homebrew_python
    except Exception:
        return None
```

This pattern handles the detection and verification of the required external environment.

## MCP Protocol Compliance

### stderr Redirection Pattern

To maintain MCP protocol compliance by ensuring all diagnostic output goes to stderr, not stdout:

```python
# Configure logging to use stderr
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# In subprocess scripts
print(f"Error: {str(e)}", file=sys.stderr)
```

This pattern keeps stdout clean for JSON protocol responses while allowing diagnostic information to be visible.

### JSON Protocol Wrapper

To ensure all outputs conform to the MCP protocol:

```python
def mcp_json_response(data):
    """Wrap data in a valid MCP JSON response."""
    response = {
        "status": "success",
        "data": data
    }
    # Ensure clean JSON output on a single line
    return json.dumps(response)

def mcp_error_response(error_message):
    """Create an error response that conforms to MCP protocol."""
    response = {
        "status": "error",
        "error": error_message
    }
    # Error details go to stderr
    print(f"Error: {error_message}", file=sys.stderr)
    # Error response goes to stdout
    return json.dumps(response)
```

This pattern ensures all outputs conform to the expected protocol format.

## Feature Flag Implementation

### Environment-Based Feature Flags

For controlling feature availability:

```python
def is_feature_enabled(feature_name):
    """Check if a feature is enabled via environment variables."""
    env_var = f"MCP_MEMORY_{feature_name.upper()}_ENABLED"
    return os.environ.get(env_var, '').lower() in ('1', 'true', 'yes')

# Example usage
if is_feature_enabled('homebrew_pytorch'):
    # Enable homebrew pytorch integration
    apply_homebrew_patches()
else:
    # Use standard implementation
    use_standard_implementation()
```

This pattern allows features to be toggled via environment variables without code changes.

### Conditional Implementation Selection

For dynamically selecting implementations:

```python
def get_storage_implementation():
    """Get the appropriate storage implementation based on configuration."""
    backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chroma')
    
    if backend == 'sqlite_vec':
        if os.environ.get('MCP_MEMORY_USE_HOMEBREW_PYTORCH', '').lower() in ('1', 'true', 'yes'):
            return HomebrewSqliteVecMemoryStorage
        else:
            return SqliteVecMemoryStorage
    elif backend == 'chroma':
        return ChromaMemoryStorage
    else:
        raise ValueError(f"Unknown storage backend: {backend}")

# Use the selected implementation
StorageClass = get_storage_implementation()
storage = StorageClass(**config)
```

This pattern allows for flexible implementation selection based on configuration.

## Error Handling and Fallback Patterns

### Graceful Degradation

For maintaining functionality when optimal implementations are unavailable:

```python
try:
    # Try to use the optimal implementation
    result = optimal_implementation()
except (ImportError, RuntimeError):
    # Fall back to a simpler implementation
    result = fallback_implementation()
except Exception as e:
    # Log the error but still provide basic functionality
    logger.error(f"Unexpected error: {e}")
    result = minimal_implementation()
```

This pattern ensures the system continues to function even when preferred implementations fail.

### Tiered Fallback Chain

For implementing multiple fallback options in order of preference:

```python
def get_embedding_model():
    """Get the best available embedding model."""
    # Try Homebrew PyTorch first
    try:
        if os.environ.get('MCP_MEMORY_USE_HOMEBREW_PYTORCH', '').lower() in ('1', 'true', 'yes'):
            model = get_homebrew_model()
            if model and model.initialized:
                return model, "homebrew"
    except Exception as e:
        logger.warning(f"Homebrew model initialization failed: {e}")
    
    # Try standard PyTorch next
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model, "pytorch"
    except ImportError:
        logger.warning("Standard PyTorch not available")
    
    # Try ONNX runtime next
    try:
        if os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes'):
            model = get_onnx_model()
            if model:
                return model, "onnx"
    except Exception:
        logger.warning("ONNX runtime not available")
    
    # Finally, use a dummy implementation
    return DummyEmbeddingModel(), "dummy"
```

This pattern provides multiple fallback options with clear prioritization.

## Conclusion

These technical patterns represent reusable solutions to common integration challenges. By documenting and formalizing these patterns, we create a foundation for future integration work and a reference for solving similar problems in other contexts.

The key insights from these patterns include:

1. Prefer isolation over direct integration when dealing with complex dependencies
2. Use feature flags to enable gradual rollout and easy rollback
3. Implement robust fallback mechanisms to maintain core functionality
4. Consider protocol compliance from the beginning, not as an afterthought
5. Verify patches and overrides to ensure they're applied correctly

These patterns can be adapted and extended for similar integration challenges in Python applications.