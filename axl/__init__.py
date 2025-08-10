"""
AXL: Accelerated Linear Layers

A PyTorch library for implementing approximate key-value linear layers using vector databases.
This library provides efficient approximate matrix multiplication through vector quantization
and similarity search techniques.

Main Components:
- ApproximateLinear: The main linear layer with approximate computation
- ApproximateLinearConfig: Configuration class for the approximate layer
- Backend implementations: FAISS and NanoPQ backends for vector search

Example:
    >>> from axl import ApproximateLinear, ApproximateLinearConfig, BackendType
    >>> config = ApproximateLinearConfig(backend=BackendType.FAISS, energy_keep=0.98)
    >>> layer = ApproximateLinear(in_features=128, out_features=64, config=config)
"""

from .layer import (
    ApproximateLinear, 
    ApproximateLinearConfig, 
    BackendType, 
    QuantizationType,
    create_backend
)

__version__ = "1.0.0"
__all__ = [
    "ApproximateLinear", 
    "ApproximateLinearConfig", 
    "BackendType", 
    "QuantizationType",
    "create_backend"
]
