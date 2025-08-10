# AXL: Accelerated Linear Layers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


AXL is a PyTorch library that implements approximate key-value linear layers using vector databases. It provides efficient approximate matrix multiplication through vector quantization and similarity search techniques. The latest API uses `ApproximateLinear` and `ApproximateLinearConfig` for configuration and layer creation.

## Features

- **Approximate Matrix Multiplication**: Efficient computation using key-value storage
- **Multiple Backends**: Support for FAISS and NanoPQ vector databases
- **Straight-Through Estimator**: Gradient flow during training
- **Configurable Quantization**: Adjustable precision and compression parameters
- **GPU Support**: Optional GPU acceleration for FAISS backend
- **Easy Integration**: Drop-in replacement for PyTorch linear layers

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For FAISS backend:
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

For NanoPQ backend:
```bash
pip install nanopq
```


## Quick Start

```python
import torch
from axl.layer import ApproximateLinear, ApproximateLinearConfig, BackendType

# Create configuration
config = ApproximateLinearConfig(
    backend=BackendType.FAISS,
    energy_keep=0.98,
    expected_keys_per_query=32,
    use_bias=True,
    device="cpu"
)

# Create layer
layer = ApproximateLinear(in_features=128, out_features=64, config=config)

# Initialize with calibration data
calibration_data = torch.randn(1000, 128)
layer.initialize(calibration_data=calibration_data)

# Use like a regular PyTorch layer
input_data = torch.randn(32, 128)
output = layer(input_data)
```

## Usage Examples


### Basic Usage

```python
import torch
from axl.layer import ApproximateLinear, ApproximateLinearConfig, BackendType

# Configure the layer
config = ApproximateLinearConfig(
    backend=BackendType.FAISS,           # Use FAISS backend
    faiss_clusters=1024,                 # Number of clusters
    faiss_probe_clusters=32,             # Clusters to probe during search
    energy_keep=0.98,                    # Keep 98% of SVD energy
    expected_keys_per_query=32,          # Expected keys per query
    use_bias=True,                       # Include bias term
    use_straight_through_estimator=True  # Use Straight-Through Estimator
)

# Create layer
layer = ApproximateLinear(in_features=256, out_features=128, config=config)

# Initialize with calibration data
calibration_data = torch.randn(1000, 256)
layer.initialize(calibration_data=calibration_data)

# Training
layer.train()
optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

for epoch in range(10):
    input_data = torch.randn(32, 256)
    target = torch.randn(32, 128)
    output = layer(input_data)
    loss = torch.nn.functional.mse_loss(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
layer.eval()
with torch.no_grad():
    test_input = torch.randn(16, 256)
    result = layer(test_input)
```


### Using NanoPQ Backend

```python
from axl.layer import ApproximateLinear, ApproximateLinearConfig, BackendType

config = ApproximateLinearConfig(
    backend=BackendType.NANOPQ,
    nanopq_key_subvectors=8,     # Number of subvectors for keys
    nanopq_key_codewords=256,    # Codewords per subvector for keys
    nanopq_value_subvectors=8,   # Number of subvectors for values
    nanopq_value_codewords=256,  # Codewords per subvector for values
    energy_keep=0.95
)

layer = ApproximateLinear(in_features=512, out_features=256, config=config)
```


### Comparing Approximate vs Exact

```python
# Approximate inference
layer.config.use_approximate_inference = True
approx_output = layer(input_data)

# Exact inference
layer.config.use_approximate_inference = False
exact_output = layer(input_data)

# Compare accuracy
relative_error = (approx_output - exact_output).norm() / (exact_output.norm() + 1e-12)
print(f"Relative error: {relative_error:.6f}")
```

## Configuration Options


### ApproximateLinearConfig Parameters

#### Backend Selection
- `backend`: Choose between `BackendType.FAISS` or `BackendType.NANOPQ`

#### FAISS-specific Parameters
- `faiss_clusters`: Number of clusters for IVFPQ (default: 4096)
- `faiss_probe_clusters`: Number of clusters to probe during search (default: 64)
- `faiss_key_subvectors`: Number of subvectors for key PQ (default: 8)
- `faiss_key_bits`: Number of bits per subvector for key PQ (default: 8)
- `faiss_value_subvectors`: Number of subvectors for value PQ (default: 8)
- `faiss_value_bits`: Number of bits per subvector for value PQ (default: 8)
- `faiss_use_gpu`: Enable GPU acceleration (default: False)

#### NanoPQ-specific Parameters
- `nanopq_key_subvectors`: Number of subvectors for key PQ (default: 8)
- `nanopq_key_codewords`: Number of codewords per subvector for key PQ (default: 256)
- `nanopq_value_subvectors`: Number of subvectors for value PQ (default: 8)
- `nanopq_value_codewords`: Number of codewords per subvector for value PQ (default: 256)

#### General Parameters
- `energy_keep`: Fraction of energy to keep in SVD (default: 0.98)
- `min_singular_value`: Minimum singular value threshold (default: 0.0)
- `ridge_regularization`: Ridge regularization for covariance estimation (default: 1e-6)
- `expected_keys_per_query`: Expected number of keys to retrieve per query (default: 64)
- `use_straight_through_estimator`: Use Straight-Through Estimator during training (default: True)
- `use_approximate_inference`: Use approximation during inference (default: True)
- `use_bias`: Include bias term (default: True)
- `device`: Device for computation (default: "cpu")

## How It Works

AXL implements approximate matrix multiplication using the following steps:

1. **SVD Decomposition**: Decomposes the weight matrix W = U × S × V^T
2. **Key-Value Construction**: Creates keys K = S × V^T and values U
3. **Whitening**: Applies input whitening using estimated covariance
4. **Vector Quantization**: Compresses keys and values using PQ
5. **Similarity Search**: Uses range search to find relevant key-value pairs
6. **Weighted Sum**: Computes output as weighted sum of retrieved values

## Performance Considerations

- **Memory Usage**: Vector quantization significantly reduces memory footprint
- **Computation Speed**: Approximate computation can be faster for large matrices
- **Accuracy Trade-off**: Lower precision settings reduce accuracy but improve speed
- **Calibration Data**: Quality of calibration data affects approximation accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use AXL in your research, please cite:

```bibtex
@software{axl2024,
  title={AXL: Accelerated Linear Layers},
  author={Soma S Dhavala},
  year={2025},
  url={https://github.com/mlsquare/axl}
}
```
