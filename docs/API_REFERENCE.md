# AXL Library - API Reference

## Overview

The AXL (Accelerated Linear Layers) library provides approximate linear layer implementations using key-value storage and vector similarity search. This document provides comprehensive API documentation for the improved version of the library.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Configuration](#configuration)
3. [Backends](#backends)
4. [Utility Functions](#utility-functions)
5. [Examples](#examples)
6. [Best Practices](#best-practices)
7. [Performance Guidelines](#performance-guidelines)
8. [Troubleshooting](#troubleshooting)

## Core Classes

### ApproximateLinear

The main layer class that implements approximate matrix multiplication.

```python
class ApproximateLinear(torch.nn.Module):
    """
    Improved Approximate Linear Layer with better API design and GPU support.
    
    This layer implements approximate matrix multiplication using key-value storage
    and vector similarity search. It provides better tensor handling, GPU support,
    and more intuitive API design.
    """
```

#### Constructor

```python
def __init__(
    self, 
    in_features: int, 
    out_features: int, 
    config: Optional[ApproximateLinearConfig] = None
) -> None:
    """
    Initialize the ApproximateLinear layer.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        config: Configuration object (uses default if None)
        
    Raises:
        ValueError: If in_features or out_features are not positive
    """
```

#### Methods

##### initialize

```python
def initialize(
    self, 
    calibration_data: Optional[torch.Tensor] = None, 
    covariance_matrix: Optional[torch.Tensor] = None
) -> None:
    """
    Initialize the layer for approximate computation.
    
    This method performs the following steps:
    1. Decomposes the weight matrix using SVD
    2. Estimates the input covariance matrix
    3. Whitens and standardizes the keys
    4. Fits the backend with keys and values
    5. Computes the search threshold
    
    Args:
        calibration_data: Calibration data for covariance estimation
        covariance_matrix: Pre-computed covariance matrix (overrides calibration_data)
        
    Raises:
        RuntimeError: If layer is already initialized
        ValueError: If calibration data has wrong dimensions
    """
```

##### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the layer.
    
    Args:
        x: Input tensor of shape (batch_size, in_features) or (in_features,)
        
    Returns:
        Output tensor of shape (batch_size, out_features) or (out_features,)
    """
```

##### to

```python
def to(self, device: Union[str, torch.device]) -> "ApproximateLinear":
    """
    Move the layer to a specified device.
    
    Args:
        device: Target device ('cpu', 'cuda', or torch.device)
        
    Returns:
        Self for method chaining
    """
```

##### save_backend

```python
def save_backend(self, directory: str) -> None:
    """
    Save the backend state to disk.
    
    Args:
        directory: Directory path where to save the backend state
        
    Raises:
        RuntimeError: If layer is not initialized
    """
```

##### load_backend

```python
def load_backend(self, directory: str) -> None:
    """
    Load the backend state from disk.
    
    Args:
        directory: Directory path from where to load the backend state
        
    Raises:
        FileNotFoundError: If backend files are not found
    """
```

##### get_config

```python
def get_config(self) -> Dict[str, Any]:
    """Get the current configuration as a dictionary."""
```

##### set_config

```python
def set_config(self, config: ApproximateLinearConfig) -> None:
    """
    Update the layer configuration.
    
    Args:
        config: New configuration object
        
    Note:
        This will reset the initialization state if the configuration changes.
    """
```

## Configuration

### ApproximateLinearConfig

Configuration class for the ApproximateLinear layer.

```python
@dataclass
class ApproximateLinearConfig:
    """
    Improved configuration class for ApproximateLinear layer.
    
    This class provides a more intuitive and well-organized configuration
    with better naming conventions and validation.
    """
```

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `BackendType` | `FAISS` | Backend type for vector storage |
| `quantization` | `QuantizationType` | `PRODUCT_QUANTIZATION` | Quantization type for values |
| `energy_keep` | `float` | `0.98` | Fraction of energy to keep in SVD decomposition |
| `min_singular_value` | `float` | `0.0` | Minimum singular value threshold |
| `ridge_regularization` | `float` | `1e-6` | Ridge regularization parameter |
| `expected_keys_per_query` | `Optional[int]` | `64` | Expected number of keys to retrieve per query |
| `max_ids_capacity` | `int` | `200000` | Maximum number of IDs to consider |
| `use_straight_through_estimator` | `bool` | `True` | Whether to use STE during training |
| `use_approximate_inference` | `bool` | `True` | Whether to use approximation during inference |
| `use_bias` | `bool` | `True` | Whether to include bias term |
| `device` | `Union[str, torch.device]` | `"cpu"` | Device to use for computation |

#### FAISS-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `faiss_clusters` | `int` | `4096` | Number of clusters for FAISS IVFPQ |
| `faiss_probe_clusters` | `int` | `64` | Number of clusters to probe during search |
| `faiss_key_subvectors` | `int` | `8` | Number of subvectors for key PQ |
| `faiss_key_bits` | `int` | `8` | Number of bits per subvector for key PQ |
| `faiss_value_subvectors` | `int` | `8` | Number of subvectors for value PQ |
| `faiss_value_bits` | `int` | `8` | Number of bits per subvector for value PQ |
| `faiss_use_gpu` | `bool` | `False` | Whether to use GPU acceleration (FAISS only) |

#### NanoPQ-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nanopq_key_subvectors` | `int` | `8` | Number of subvectors for NanoPQ key PQ |
| `nanopq_key_codewords` | `int` | `256` | Number of codewords per subvector for NanoPQ key PQ |
| `nanopq_value_subvectors` | `int` | `8` | Number of subvectors for NanoPQ value PQ |
| `nanopq_value_codewords` | `int` | `256` | Number of codewords per subvector for NanoPQ value PQ |

#### Methods

##### to_dict

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary."""
```

##### from_dict

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> "ApproximateLinearConfig":
    """Create configuration from dictionary."""
```

### BackendType

Enumeration for backend types.

```python
class BackendType(Enum):
    """Enumeration for backend types."""
    FAISS = "faiss"
    NANOPQ = "nanopq"
```

### QuantizationType

Enumeration for quantization types.

```python
class QuantizationType(Enum):
    """Enumeration for quantization types."""
    PRODUCT_QUANTIZATION = "PQ"
    SCALAR_QUANTIZATION = "SQ"
```

## Backends

### KVBackend

Abstract base class for backend implementations.

```python
class KVBackend(abc.ABC):
    """
    Abstract base class for key-value backend implementations.
    
    This class defines the interface that all backend implementations must follow.
    Backends are responsible for storing and retrieving key-value pairs efficiently
    using various vector database techniques.
    """
```

#### Methods

##### fit_keys

```python
@abc.abstractmethod
def fit_keys(self, K_tilde: np.ndarray, ids: np.ndarray, **kwargs: Any) -> None:
    """
    Fit the backend to store key vectors.
    
    Args:
        K_tilde: Whitened and standardized key matrix of shape (n_keys, key_dim)
        ids: Array of key IDs of shape (n_keys,)
        **kwargs: Additional keyword arguments specific to the backend
        
    Raises:
        ValueError: If input dimensions are invalid
    """
```

##### fit_values

```python
@abc.abstractmethod
def fit_values(self, U_cols: np.ndarray, ids: np.ndarray, **kwargs: Any) -> None:
    """
    Fit the backend to store value vectors.
    
    Args:
        U_cols: Value matrix of shape (n_values, value_dim)
        ids: Array of value IDs of shape (n_values,)
        **kwargs: Additional keyword arguments specific to the backend
        
    Raises:
        ValueError: If input dimensions are invalid or keys not fitted
    """
```

##### range_search_ip

```python
@abc.abstractmethod
def range_search_ip(self, z: np.ndarray, threshold: float) -> np.ndarray:
    """
    Perform range search using inner product similarity.
    
    Args:
        z: Query vector of shape (key_dim,)
        threshold: Similarity threshold for range search
        
    Returns:
        Array of key IDs that meet the similarity threshold
        
    Raises:
        RuntimeError: If backend is not fitted
    """
```

##### reconstruct_values

```python
@abc.abstractmethod
def reconstruct_values(self, ids: Sequence[int]) -> np.ndarray:
    """
    Reconstruct value vectors from their IDs.
    
    Args:
        ids: Sequence of value IDs to reconstruct
        
    Returns:
        Reconstructed value matrix of shape (len(ids), value_dim)
        
    Raises:
        RuntimeError: If backend is not fitted
        ValueError: If any ID is invalid
    """
```

##### refine_scores

```python
@abc.abstractmethod
def refine_scores(self, ids: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Refine similarity scores for given key IDs.
    
    Args:
        ids: Array of key IDs
        z: Query vector of shape (key_dim,)
        
    Returns:
        Refined similarity scores for each key ID
        
    Raises:
        RuntimeError: If backend is not fitted
    """
```

##### save

```python
@abc.abstractmethod
def save(self, directory: str) -> None:
    """
    Save the backend state to disk.
    
    Args:
        directory: Directory path where to save the backend state
        
    Raises:
        RuntimeError: If backend is not fitted
        OSError: If directory cannot be created or files cannot be written
    """
```

##### load

```python
@abc.abstractmethod
def load(self, directory: str) -> None:
    """
    Load the backend state from disk.
    
    Args:
        directory: Directory path from where to load the backend state
        
    Raises:
        FileNotFoundError: If backend files are not found
        ValueError: If saved data is corrupted or incompatible
    """
```

##### get_info

```python
def get_info(self) -> Dict[str, Any]:
    """
    Get information about the backend state.
    
    Returns:
        Dictionary containing backend information
    """
```

##### reset

```python
def reset(self) -> None:
    """Reset the backend to initial state."""
```

## Utility Functions

### Mathematical Functions

#### standard_normal_pdf

```python
def standard_normal_pdf(x: float) -> float:
    """
    Compute the probability density function of the standard normal distribution.
    
    Args:
        x: Input value
        
    Returns:
        PDF value at x
    """
```

#### standard_normal_cdf

```python
def standard_normal_cdf(x: float) -> float:
    """
    Compute the cumulative distribution function of the standard normal distribution.
    
    Args:
        x: Input value
        
    Returns:
        CDF value at x
    """
```

#### inverse_standard_normal_cdf

```python
def inverse_standard_normal_cdf(probability: float) -> float:
    """
    Compute the inverse CDF (quantile function) of the standard normal distribution.
    
    Uses binary search to find the value x such that P(X <= x) = probability.
    Results are cached for better performance on repeated calls.
    
    Args:
        probability: Probability value in (0, 1)
        
    Returns:
        The quantile value x
        
    Raises:
        ValueError: If probability is not in (0, 1)
    """
```

### Matrix Operations

#### svd_keys_values

```python
def svd_keys_values(
    weight_matrix: np.ndarray, 
    energy_keep: float = 1.0, 
    min_sigma: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform SVD decomposition and extract key-value components.
    
    Decomposes the weight matrix W = U * S * V^T and constructs key-value pairs
    for approximate computation. The keys are constructed as K = S * V^T.
    
    Args:
        weight_matrix: Input weight matrix of shape (out_features, in_features)
        energy_keep: Fraction of energy to keep in SVD (default: 1.0)
        min_sigma: Minimum singular value threshold (default: 0.0)
        
    Returns:
        Tuple of (U, S, V, K) where:
        - U: Left singular vectors (out_features, rank_kept)
        - S: Singular values (rank_kept,)
        - V: Right singular vectors (in_features, rank_kept)
        - K: Key matrix (rank_kept, in_features)
        
    Raises:
        ValueError: If energy_keep is not in (0, 1] or min_sigma is negative
    """
```

#### estimate_covariance

```python
def estimate_covariance(
    calibration_data: Optional[np.ndarray], 
    feature_dim: int, 
    ridge: float = 1e-6
) -> np.ndarray:
    """
    Estimate the covariance matrix from calibration data.
    
    Args:
        calibration_data: Calibration data of shape (n_samples, feature_dim) or None
        feature_dim: Number of features
        ridge: Ridge regularization parameter for numerical stability
        
    Returns:
        Estimated covariance matrix of shape (feature_dim, feature_dim)
        
    Raises:
        ValueError: If feature_dim is not positive or ridge is negative
    """
```

#### whiten_standardize_keys

```python
def whiten_standardize_keys(
    key_matrix: np.ndarray, 
    covariance_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Whiten and standardize key vectors using Cholesky decomposition.
    
    Args:
        key_matrix: Key matrix of shape (n_keys, key_dim)
        covariance_matrix: Covariance matrix of shape (key_dim, key_dim)
        
    Returns:
        Tuple of (L_inv, K_tilde, scales) where:
        - L_inv: Inverse Cholesky factor (key_dim, key_dim)
        - K_tilde: Whitened and standardized keys (n_keys, key_dim)
        - scales: Scaling factors for each key (n_keys,)
        
    Raises:
        ValueError: If matrices have incompatible dimensions
        np.linalg.LinAlgError: If covariance matrix is not positive definite
    """
```

### Utility Functions

#### compute_search_threshold

```python
def compute_search_threshold(
    expected_keys: int, 
    total_keys: int, 
    method: str = "quantile"
) -> float:
    """
    Compute search threshold based on expected number of keys.
    
    Args:
        expected_keys: Expected number of keys to retrieve
        total_keys: Total number of available keys
        method: Method to compute threshold ("quantile", "fixed", "adaptive")
        
    Returns:
        Search threshold value
        
    Raises:
        ValueError: If expected_keys > total_keys or method is invalid
    """
```

#### validate_tensor

```python
def validate_tensor(
    tensor: np.ndarray, 
    expected_shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    Validate and normalize tensor properties.
    
    Args:
        tensor: Input tensor
        expected_shape: Expected shape (None for no validation)
        dtype: Expected dtype (None for no conversion)
        
    Returns:
        Validated and normalized tensor
        
    Raises:
        ValueError: If tensor doesn't meet requirements
    """
```

#### safe_matrix_inverse

```python
def safe_matrix_inverse(matrix: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Safely compute matrix inverse with regularization.
    
    Args:
        matrix: Input matrix
        ridge: Ridge regularization parameter
        
    Returns:
        Inverse matrix
        
    Raises:
        np.linalg.LinAlgError: If matrix is singular even with regularization
    """
```

## Examples

### Basic Usage

```python
import torch
from axl.layer_improved import ApproximateLinear, ApproximateLinearConfig, BackendType

# Create configuration
config = ApproximateLinearConfig(
    backend=BackendType.FAISS,
    faiss_clusters=1024,
    faiss_probe_clusters=64,
    energy_keep=0.95,
    expected_keys_per_query=32
)

# Create layer
layer = ApproximateLinear(in_features=512, out_features=1024, config=config)

# Generate calibration data
calibration_data = torch.randn(1000, 512)

# Initialize the layer
layer.initialize(calibration_data=calibration_data)

# Use the layer
input_tensor = torch.randn(10, 512)
output = layer(input_tensor)
print(f"Output shape: {output.shape}")
```

### GPU Support

```python
import torch
from axl.layer_improved import ApproximateLinear, ApproximateLinearConfig, BackendType

# Create GPU configuration
config = ApproximateLinearConfig(
    backend=BackendType.FAISS,
    faiss_clusters=1024,
    faiss_probe_clusters=64,
    faiss_use_gpu=True,
    device="cuda",
    energy_keep=0.95
)

# Create layer
layer = ApproximateLinear(in_features=512, out_features=1024, config=config)

# Move to GPU
layer = layer.to("cuda")

# Generate calibration data on GPU
calibration_data = torch.randn(1000, 512, device="cuda")

# Initialize
layer.initialize(calibration_data=calibration_data)

# Use on GPU
input_tensor = torch.randn(10, 512, device="cuda")
output = layer(input_tensor)
```

### Different Backends

```python
import torch
from axl.layer_improved import ApproximateLinear, ApproximateLinearConfig, BackendType

# FAISS backend
faiss_config = ApproximateLinearConfig(
    backend=BackendType.FAISS,
    faiss_clusters=1024,
    faiss_probe_clusters=64,
    faiss_key_subvectors=8,
    faiss_key_bits=8,
    energy_keep=0.95
)

# NanoPQ backend
nanopq_config = ApproximateLinearConfig(
    backend=BackendType.NANOPQ,
    nanopq_key_subvectors=8,
    nanopq_key_codewords=256,
    nanopq_value_subvectors=8,
    nanopq_value_codewords=256,
    energy_keep=0.95
)

# Create layers
faiss_layer = ApproximateLinear(512, 1024, faiss_config)
nanopq_layer = ApproximateLinear(512, 1024, nanopq_config)

# Initialize both
calibration_data = torch.randn(1000, 512)
faiss_layer.initialize(calibration_data=calibration_data)
nanopq_layer.initialize(calibration_data=calibration_data)

# Compare performance
input_tensor = torch.randn(100, 512)

import time
start_time = time.time()
faiss_output = faiss_layer(input_tensor)
faiss_time = time.time() - start_time

start_time = time.time()
nanopq_output = nanopq_layer(input_tensor)
nanopq_time = time.time() - start_time

print(f"FAISS time: {faiss_time:.4f}s")
print(f"NanoPQ time: {nanopq_time:.4f}s")
```

### Save and Load Backends

```python
import torch
import tempfile
import os
from axl.layer_improved import ApproximateLinear, ApproximateLinearConfig, BackendType

# Create and initialize layer
config = ApproximateLinearConfig(
    backend=BackendType.FAISS,
    faiss_clusters=1024,
    energy_keep=0.95
)

layer = ApproximateLinear(512, 1024, config)
calibration_data = torch.randn(1000, 512)
layer.initialize(calibration_data=calibration_data)

# Save backend
with tempfile.TemporaryDirectory() as temp_dir:
    layer.save_backend(temp_dir)
    
    # Create new layer and load backend
    new_layer = ApproximateLinear(512, 1024, config)
    new_layer.load_backend(temp_dir)
    
    # Test that they produce the same output
    input_tensor = torch.randn(10, 512)
    output1 = layer(input_tensor)
    output2 = new_layer(input_tensor)
    
    print(f"Outputs match: {torch.allclose(output1, output2)}")
```

## Best Practices

### Configuration

1. **Choose appropriate backend**: Use FAISS for large-scale applications and NanoPQ for memory-constrained environments.

2. **Set energy_keep carefully**: Higher values (0.95-0.98) provide better accuracy but use more memory.

3. **Tune cluster parameters**: For FAISS, set `faiss_clusters` to be roughly `sqrt(num_keys)` and `faiss_probe_clusters` to 1-10% of total clusters.

4. **Use GPU when available**: Set `faiss_use_gpu=True` and `device="cuda"` for better performance on large models.

### Initialization

1. **Use representative calibration data**: The calibration data should be similar to the data you'll use during inference.

2. **Provide sufficient calibration data**: Use at least 1000 samples for reliable covariance estimation.

3. **Initialize once**: Don't reinitialize unless you change the configuration.

### Performance

1. **Batch processing**: Process multiple inputs together for better efficiency.

2. **Memory management**: Monitor memory usage, especially with large models.

3. **Profile your use case**: Use the benchmarking suite to find optimal parameters.

## Performance Guidelines

### Memory Usage

- **Dense layer**: `in_features × out_features × 4` bytes (float32)
- **Approximate layer**: Varies based on configuration and compression ratio
- **Typical compression**: 2-10x depending on `energy_keep` and backend settings

### Speed vs Accuracy Trade-offs

| Configuration | Speed | Accuracy | Memory |
|---------------|-------|----------|--------|
| High compression | Fast | Lower | Low |
| Medium compression | Medium | Medium | Medium |
| Low compression | Slower | Higher | High |

### Recommended Settings

#### For Speed
```python
config = ApproximateLinearConfig(
    backend=BackendType.FAISS,
    faiss_clusters=256,
    faiss_probe_clusters=16,
    energy_keep=0.8,
    expected_keys_per_query=16
)
```

#### For Accuracy
```python
config = ApproximateLinearConfig(
    backend=BackendType.FAISS,
    faiss_clusters=2048,
    faiss_probe_clusters=128,
    energy_keep=0.98,
    expected_keys_per_query=64
)
```

#### For Memory Efficiency
```python
config = ApproximateLinearConfig(
    backend=BackendType.NANOPQ,
    nanopq_key_subvectors=4,
    nanopq_key_codewords=64,
    nanopq_value_subvectors=4,
    nanopq_value_codewords=64,
    energy_keep=0.9
)
```

## Troubleshooting

### Common Issues

#### Layer not initialized
```
RuntimeError: Layer must be initialized before forward pass
```
**Solution**: Call `layer.initialize()` with calibration data before using the layer.

#### Out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `faiss_clusters`, `energy_keep`, or use CPU backend.

#### Poor accuracy
**Solution**: Increase `energy_keep`, `faiss_clusters`, or `expected_keys_per_query`.

#### Slow performance
**Solution**: Use GPU backend, reduce `faiss_probe_clusters`, or use NanoPQ backend.

### Debugging

1. **Check backend info**:
```python
info = layer.backend.get_info()
print(info)
```

2. **Monitor memory usage**:
```python
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

3. **Profile performance**:
```python
import time
start_time = time.time()
output = layer(input_tensor)
print(f"Forward pass time: {(time.time() - start_time) * 1000:.2f} ms")
```

### Getting Help

1. Check the examples in this documentation
2. Run the test suite: `python -m pytest test_axl_improved.py`
3. Use the benchmarking suite: `python benchmark_axl.py`
4. Review the source code for detailed implementation 