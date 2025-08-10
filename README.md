# AXL: Accelerated Linear Layers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AXL is a PyTorch library that implements approximate key-value linear layers using vector databases. It provides efficient approximate matrix multiplication through vector quantization and similarity search techniques.

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
from axl import ApproxKVLinear, KVLinearConfig

# Create configuration
config = KVLinearConfig(
    backend="faiss",
    energy_keep=0.98,
    expected_K=32
)

# Create layer
layer = ApproxKVLinear(in_features=128, out_features=64, config=config)

# Prepare with calibration data
calibration_data = torch.randn(1000, 128)
layer.prepare(calibration_data=calibration_data)

# Use like a regular PyTorch layer
input_data = torch.randn(32, 128)
output = layer(input_data)
```

## Usage Examples

### Basic Usage

```python
import torch
from axl import ApproxKVLinear, KVLinearConfig

# Configure the layer
config = KVLinearConfig(
    backend="faiss",           # Use FAISS backend
    nlist=1024,               # Number of clusters
    nprobe=32,                # Clusters to probe during search
    energy_keep=0.98,         # Keep 98% of SVD energy
    expected_K=32,            # Expected keys per query
    use_bias=True,            # Include bias term
    use_ste=True              # Use Straight-Through Estimator
)

# Create layer
layer = ApproxKVLinear(in_features=256, out_features=128, config=config)

# Prepare with calibration data
calibration_data = torch.randn(1000, 256)
layer.prepare(calibration_data=calibration_data)

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
from axl import ApproxKVLinear, KVLinearConfig

config = KVLinearConfig(
    backend="nanopq",
    npq_M_keys=8,     # Number of subvectors for keys
    npq_Ks_keys=256,  # Codewords per subvector for keys
    npq_M_vals=8,     # Number of subvectors for values
    npq_Ks_vals=256,  # Codewords per subvector for values
    energy_keep=0.95
)

layer = ApproxKVLinear(in_features=512, out_features=256, config=config)
```

### Comparing Approximate vs Exact

```python
# Approximate inference
layer.cfg.approx_inference = True
approx_output = layer(input_data)

# Exact inference
layer.cfg.approx_inference = False
exact_output = layer(input_data)

# Compare accuracy
relative_error = (approx_output - exact_output).norm() / exact_output.norm()
print(f"Relative error: {relative_error:.6f}")
```

## Configuration Options

### KVLinearConfig Parameters

#### Backend Selection
- `backend`: Choose between "faiss" or "nanopq"

#### FAISS-specific Parameters
- `nlist`: Number of clusters for IVFPQ (default: 4096)
- `m_pq_keys`: Number of subvectors for key PQ (default: 8)
- `nbits_keys`: Number of bits per subvector for key PQ (default: 8)
- `nprobe`: Number of clusters to probe during search (default: 64)
- `use_gpu`: Enable GPU acceleration (default: False)
- `values_index_type`: Type of index for values ("PQ" or scalar quantization)
- `m_pq_vals`: Number of subvectors for value PQ (default: 8)
- `nbits_vals`: Number of bits per subvector for value PQ (default: 8)
- `sq_type`: Scalar quantization type (default: "QT_8bit")

#### NanoPQ-specific Parameters
- `npq_M_keys`: Number of subvectors for key PQ (default: 8)
- `npq_Ks_keys`: Number of codewords per subvector for key PQ (default: 256)
- `npq_M_vals`: Number of subvectors for value PQ (default: 8)
- `npq_Ks_vals`: Number of codewords per subvector for value PQ (default: 256)

#### General Parameters
- `energy_keep`: Fraction of energy to keep in SVD (default: 1.0)
- `min_sigma`: Minimum singular value threshold (default: 0.0)
- `ridge`: Ridge regularization for covariance estimation (default: 1e-6)
- `expected_K`: Expected number of keys to retrieve per query (default: 64)
- `refine_exact_with_float_keys`: Store float keys for exact refinement (default: False)
- `use_ste`: Use Straight-Through Estimator during training (default: True)
- `approx_inference`: Use approximation during inference (default: True)
- `use_bias`: Include bias term (default: True)

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
  author={Your Name},
  year={2024},
  url={https://github.com/mlsquare/axl}
}
```
