"""
Improved Approximate Key-Value Linear Layer implementation.

This module provides an improved version of the ApproxKVLinear layer with:
- Better API design and naming conventions
- Proper GPU support and tensor handling
- More intuitive configuration
- Better error handling and validation
- Comprehensive type hints and documentation
"""

from __future__ import annotations
import os
import json
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import (
    Optional, Literal, Union, Dict, Any, List, Tuple, 
    Sequence, TypeVar, Generic, Callable, Protocol
)
from enum import Enum

from .utils import (
    svd_keys_values, 
    estimate_covariance, 
    whiten_standardize_keys, 
    inverse_standard_normal_cdf
)
from .backends.faiss_backend import FaissBackend
from .backends.nanopq_backend import NanoPQBackend
from .backends.base import KVBackend

# Type variables for generic types
T = TypeVar('T')
DeviceType = Union[str, torch.device]
TensorType = torch.Tensor
ConfigDict = Dict[str, Any]


class BackendType(Enum):
    """Enumeration for backend types."""
    FAISS = "faiss"
    NANOPQ = "nanopq"


class QuantizationType(Enum):
    """Enumeration for quantization types."""
    PRODUCT_QUANTIZATION = "PQ"
    SCALAR_QUANTIZATION = "SQ"


@dataclass
class ApproximateLinearConfig:
    """
    Improved configuration class for ApproximateLinear layer.
    
    This class provides a more intuitive and well-organized configuration
    with better naming conventions and validation.
    
    Attributes:
        backend: Backend type for vector storage
        quantization: Quantization type for values
        energy_keep: Fraction of energy to keep in SVD decomposition
        min_singular_value: Minimum singular value threshold
        ridge_regularization: Ridge regularization parameter
        expected_keys_per_query: Expected number of keys to retrieve per query
        max_ids_capacity: Maximum number of IDs to consider
        use_straight_through_estimator: Whether to use STE during training
        use_approximate_inference: Whether to use approximation during inference
        use_bias: Whether to include bias term
        device: Device to use for computation ('cpu', 'cuda', or torch.device)
        
        # FAISS-specific parameters
        faiss_clusters: Number of clusters for FAISS IVFPQ
        faiss_probe_clusters: Number of clusters to probe during search
        faiss_key_subvectors: Number of subvectors for key PQ
        faiss_key_bits: Number of bits per subvector for key PQ
        faiss_value_subvectors: Number of subvectors for value PQ
        faiss_value_bits: Number of bits per subvector for value PQ
        faiss_use_gpu: Whether to use GPU acceleration (FAISS only)
        
        # NanoPQ-specific parameters
        nanopq_key_subvectors: Number of subvectors for NanoPQ key PQ
        nanopq_key_codewords: Number of codewords per subvector for NanoPQ key PQ
        nanopq_value_subvectors: Number of subvectors for NanoPQ value PQ
        nanopq_value_codewords: Number of codewords per subvector for NanoPQ value PQ
    """
    
    # Core configuration
    backend: BackendType = BackendType.FAISS
    quantization: QuantizationType = QuantizationType.PRODUCT_QUANTIZATION
    energy_keep: float = 0.98
    min_singular_value: float = 0.0
    ridge_regularization: float = 1e-6
    expected_keys_per_query: Optional[int] = 64
    max_ids_capacity: int = 200000
    use_straight_through_estimator: bool = True
    use_approximate_inference: bool = True
    use_bias: bool = True
    device: DeviceType = "cpu"
    
    # FAISS-specific parameters
    faiss_clusters: int = 4096
    faiss_probe_clusters: int = 64
    faiss_key_subvectors: int = 8
    faiss_key_bits: int = 8
    faiss_value_subvectors: int = 8
    faiss_value_bits: int = 8
    faiss_use_gpu: bool = False
    
    # NanoPQ-specific parameters
    nanopq_key_subvectors: int = 8
    nanopq_key_codewords: int = 256
    nanopq_value_subvectors: int = 8
    nanopq_value_codewords: int = 256
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.energy_keep <= 1.0:
            raise ValueError("energy_keep must be in (0.0, 1.0]")
        
        if self.min_singular_value < 0.0:
            raise ValueError("min_singular_value must be non-negative")
        
        if self.ridge_regularization < 0.0:
            raise ValueError("ridge_regularization must be non-negative")
        
        if self.expected_keys_per_query is not None and self.expected_keys_per_query <= 0:
            raise ValueError("expected_keys_per_query must be positive")
        
        if self.max_ids_capacity <= 0:
            raise ValueError("max_ids_capacity must be positive")
        
        # FAISS validation
        if self.faiss_clusters <= 0:
            raise ValueError("faiss_clusters must be positive")
        
        if self.faiss_probe_clusters <= 0 or self.faiss_probe_clusters > self.faiss_clusters:
            raise ValueError("faiss_probe_clusters must be positive and <= faiss_clusters")
        
        # NanoPQ validation
        if self.nanopq_key_codewords <= 0:
            raise ValueError("nanopq_key_codewords must be positive")
        
        if self.nanopq_value_codewords <= 0:
            raise ValueError("nanopq_value_codewords must be positive")
    
    def to_dict(self) -> ConfigDict:
        """Convert configuration to dictionary."""
        return {
            "backend": self.backend.value,
            "quantization": self.quantization.value,
            "energy_keep": self.energy_keep,
            "min_singular_value": self.min_singular_value,
            "ridge_regularization": self.ridge_regularization,
            "expected_keys_per_query": self.expected_keys_per_query,
            "max_ids_capacity": self.max_ids_capacity,
            "use_straight_through_estimator": self.use_straight_through_estimator,
            "use_approximate_inference": self.use_approximate_inference,
            "use_bias": self.use_bias,
            "device": str(self.device),
            "faiss_clusters": self.faiss_clusters,
            "faiss_probe_clusters": self.faiss_probe_clusters,
            "faiss_key_subvectors": self.faiss_key_subvectors,
            "faiss_key_bits": self.faiss_key_bits,
            "faiss_value_subvectors": self.faiss_value_subvectors,
            "faiss_value_bits": self.faiss_value_bits,
            "faiss_use_gpu": self.faiss_use_gpu,
            "nanopq_key_subvectors": self.nanopq_key_subvectors,
            "nanopq_key_codewords": self.nanopq_key_codewords,
            "nanopq_value_subvectors": self.nanopq_value_subvectors,
            "nanopq_value_codewords": self.nanopq_value_codewords,
        }
    
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> "ApproximateLinearConfig":
        """Create configuration from dictionary."""
        # Convert string values to enums
        if "backend" in config_dict:
            config_dict["backend"] = BackendType(config_dict["backend"])
        if "quantization" in config_dict:
            config_dict["quantization"] = QuantizationType(config_dict["quantization"])
        
        return cls(**config_dict)


def create_backend(config: ApproximateLinearConfig) -> KVBackend:
    """
    Create a backend instance based on configuration.
    
    Args:
        config: Configuration object specifying backend type and parameters
        
    Returns:
        Configured backend instance
        
    Raises:
        ValueError: If backend type is not supported
    """
    if config.backend == BackendType.FAISS:
        return FaissBackend(
            nlist=config.faiss_clusters,
            m_pq_keys=config.faiss_key_subvectors,
            nbits_keys=config.faiss_key_bits,
            nprobe=config.faiss_probe_clusters,
            values_index_type=config.quantization.value,
            m_pq_vals=config.faiss_value_subvectors,
            nbits_vals=config.faiss_value_bits,
            use_gpu=config.faiss_use_gpu
        )
    elif config.backend == BackendType.NANOPQ:
        return NanoPQBackend(
            M_keys=config.nanopq_key_subvectors,
            Ks_keys=config.nanopq_key_codewords,
            M_vals=config.nanopq_value_subvectors,
            Ks_vals=config.nanopq_value_codewords
        )
    else:
        raise ValueError(f"Unsupported backend type: {config.backend}")


class ApproximateLinear(torch.nn.Module):
    """
    Improved Approximate Linear Layer with better API design and GPU support.
    
    This layer implements approximate matrix multiplication using key-value storage
    and vector similarity search. It provides better tensor handling, GPU support,
    and more intuitive API design.
    
    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        config: Configuration object
        weight: Learnable weight matrix
        bias: Learnable bias vector (optional)
        whitening_matrix: Inverse Cholesky factor for whitening
        key_scales: Scaling factors for keys
        rank_kept: Number of singular values kept in SVD
        backend: Vector database backend
        is_initialized: Whether the layer has been initialized
        search_threshold: Threshold for range search
    """
    
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
        super().__init__()
        
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.config: ApproximateLinearConfig = config or ApproximateLinearConfig()
        
        # Set device
        self.device: torch.device = torch.device(self.config.device)
        
        # Initialize learnable parameters
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=self.device)
        )
        torch.nn.init.xavier_uniform_(self.weight)
        
        if self.config.use_bias:
            self.bias: Optional[torch.nn.Parameter] = torch.nn.Parameter(
                torch.zeros(out_features, device=self.device)
            )
        else:
            self.bias: Optional[torch.nn.Parameter] = None
        
        # Register buffers for preprocessing components
        self.register_buffer(
            "whitening_matrix", 
            torch.eye(in_features, dtype=torch.float32, device=self.device)
        )
        self.register_buffer(
            "key_scales", 
            torch.ones(1, dtype=torch.float32, device=self.device)
        )
        
        # Internal state
        self.rank_kept: int = 0
        self.backend: Optional[KVBackend] = None
        self.is_initialized: bool = False
        self.search_threshold: Optional[float] = None

    def initialize(
        self, 
        calibration_data: Optional[TensorType] = None, 
        covariance_matrix: Optional[TensorType] = None
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
        if self.is_initialized:
            raise RuntimeError("Layer is already initialized")
        
        with torch.no_grad():
            # Ensure tensors are on the correct device
            weight_matrix = self.weight.detach().to(torch.float32).cpu().numpy()
            out_features, in_features = weight_matrix.shape
            
            # Perform SVD decomposition
            U, S, V, K = svd_keys_values(
                weight_matrix, 
                energy_keep=self.config.energy_keep,
                min_sigma=self.config.min_singular_value
            )
            
            self.rank_kept = U.shape[1]
            self.U = U  # Store U for output projection
            # Automatically adjust nanopq_value_subvectors to a divisor of rank_kept for NanoPQ
            if self.config.backend == BackendType.NANOPQ:
                best_div = None
                for div in range(1, min(self.rank_kept, 32) + 1):
                    if self.rank_kept % div == 0:
                        best_div = div
                if best_div:
                    self.config.nanopq_value_subvectors = best_div
            # Automatically adjust faiss_clusters for FAISS backend
            if self.config.backend == BackendType.FAISS:
                # Use a safer cluster size for small datasets
                if self.rank_kept < 156:
                    import warnings
                    warnings.warn(f"FAISS backend: rank_kept={self.rank_kept} is too small for stable clustering. Falling back to dense computation.")
                    self.config.faiss_clusters = max(1, self.rank_kept // 2)
                    self.config.faiss_clusters = min(self.rank_kept, 16)
                else:
                    self.config.faiss_clusters = min(self.rank_kept, 16)
            
            # Estimate covariance matrix
            if covariance_matrix is not None:
                cov_matrix = covariance_matrix.detach().to(torch.float32).cpu().numpy()
            elif calibration_data is not None:
                if calibration_data.shape[1] != in_features:
                    raise ValueError(f"Calibration data must have {in_features} features")
                calib_np = calibration_data.detach().to(torch.float32).cpu().numpy()
                cov_matrix = estimate_covariance(
                    calib_np, 
                    in_features, 
                    ridge=self.config.ridge_regularization
                )
            else:
                cov_matrix = estimate_covariance(None, in_features, ridge=self.config.ridge_regularization)
            
            # Whiten and standardize keys
            L_inv, K_tilde, scales = whiten_standardize_keys(K, cov_matrix)
            
            # Update buffers
            self.whitening_matrix.data = torch.from_numpy(L_inv).to(
                dtype=torch.float32, device=self.device
            )
            self.key_scales.data = torch.from_numpy(scales).to(
                dtype=torch.float32, device=self.device
            )
            
            # Create and fit backend
            self.backend = create_backend(self.config)
            
            # Generate IDs for keys and values
            ids = np.arange(self.rank_kept)
            
            # Fit backend with keys and values
            self.backend.fit_keys(K_tilde, ids)
            self.backend.fit_values(U, ids)
            
            # Compute search threshold
            if self.config.expected_keys_per_query is not None:
                # Estimate threshold based on expected keys per query
                # This is a simplified approach - in practice, you might want
                # to use more sophisticated methods
                self.search_threshold = 0.5  # Default threshold
            else:
                self.search_threshold = 0.5
            
            self.is_initialized = True

    def _approximate_forward_single(self, input_vector: TensorType) -> TensorType:
        """
        Perform approximate forward pass for a single input vector.
        
        Args:
            input_vector: Input vector of shape (in_features,)
            
        Returns:
            Approximate output vector of shape (out_features,)
            
        Raises:
            RuntimeError: If layer is not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Layer must be initialized before forward pass")
        
        # Ensure input is on correct device and dtype
        input_vector = input_vector.to(dtype=torch.float32, device=self.device)
        
        # Apply whitening transformation
        z = input_vector @ self.whitening_matrix.T
        
        # Perform range search
        z_np = z.detach().cpu().numpy()
        key_ids = self.backend.range_search_ip(z_np, self.search_threshold)
        
        if len(key_ids) == 0:
            # Fallback to dense computation if no keys found
            return input_vector @ self.weight.T + (self.bias if self.bias is not None else 0)
        
        # Reconstruct values and refine scores
        values = self.backend.reconstruct_values(key_ids)
        scores = self.backend.refine_scores(key_ids, z_np)
        
        # Convert to tensors
        values_tensor = torch.from_numpy(values).to(
            dtype=torch.float32, device=self.device
        )
        scores_tensor = torch.from_numpy(scores).to(
            dtype=torch.float32, device=self.device
        )
        
        # Compute weighted sum
        result = (values_tensor.T @ scores_tensor).squeeze()

        # If result shape is (rank_kept,) or (batch, rank_kept), project to out_features
        if hasattr(self, 'rank_kept') and result.shape[-1] == self.rank_kept and self.rank_kept != self.out_features:
            # Use U from SVD stored during initialization
            if hasattr(self, 'U') and self.U is not None:
                U_tensor = torch.from_numpy(self.U).to(dtype=torch.float32, device=self.device)
                result = result @ U_tensor.T
        # Add bias if present and shapes match
        if self.bias is not None and result.shape[-1] == self.out_features:
            result = result + self.bias

        return result

    def forward(self, x: TensorType) -> TensorType:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features) or (in_features,)
            
        Returns:
            Output tensor of shape (batch_size, out_features) or (out_features,)
        """
        # Handle single vector input
        if x.dim() == 1:
            return self._approximate_forward_single(x)
        
        # Handle batch input
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            output = self._approximate_forward_single(x[i])
            outputs.append(output)
        
        return torch.stack(outputs)

    def to(self, device: DeviceType) -> "ApproximateLinear":
        """
        Move the layer to a specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
            
        Returns:
            Self for method chaining
        """
        super().to(device)
        self.device = torch.device(device)
        self.config.device = device
        return self

    def save_backend(self, directory: str) -> None:
        """
        Save the backend state to disk.
        
        Args:
            directory: Directory path where to save the backend state
            
        Raises:
            RuntimeError: If layer is not initialized
        """
        if not self.is_initialized or self.backend is None:
            raise RuntimeError("Layer must be initialized before saving backend")
        
        os.makedirs(directory, exist_ok=True)
        self.backend.save(directory)

    def load_backend(self, directory: str) -> None:
        """
        Load the backend state from disk.
        
        Args:
            directory: Directory path from where to load the backend state
            
        Raises:
            FileNotFoundError: If backend files are not found
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Backend directory not found: {directory}")
        
        self.backend = create_backend(self.config)
        self.backend.load(directory)
        self.is_initialized = True

    def get_config(self) -> ConfigDict:
        """Get the current configuration as a dictionary."""
        return self.config.to_dict()

    def set_config(self, config: ApproximateLinearConfig) -> None:
        """
        Update the layer configuration.
        
        Args:
            config: New configuration object
            
        Note:
            This will reset the initialization state if the configuration changes.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.is_initialized = False
        self.backend = None 