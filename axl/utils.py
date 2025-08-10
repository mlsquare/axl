"""
Utility functions for AXL library.

This module contains mathematical utilities for normal distribution operations,
SVD decomposition, covariance estimation, and key-value preprocessing.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional, Union, List, Callable, TypeVar
from functools import lru_cache

# Type aliases for better readability
FloatArray = np.ndarray
IntArray = np.ndarray
ScalarType = Union[float, int]
T = TypeVar('T')


def standard_normal_pdf(x: float) -> float:
    """
    Compute the probability density function of the standard normal distribution.
    
    Args:
        x: Input value
        
    Returns:
        PDF value at x
        
    Examples:
        >>> standard_normal_pdf(0.0)
        0.3989422804014327
        >>> standard_normal_pdf(1.0)
        0.24197072451914337
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def standard_normal_cdf(x: float) -> float:
    """
    Compute the cumulative distribution function of the standard normal distribution.
    
    Args:
        x: Input value
        
    Returns:
        CDF value at x
        
    Examples:
        >>> standard_normal_cdf(0.0)
        0.5
        >>> standard_normal_cdf(1.0)
        0.8413447460685429
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@lru_cache(maxsize=1024)
def _binary_search_inverse_cdf(probability: float, tolerance: float = 1e-10) -> float:
    """
    Binary search implementation for inverse CDF with caching.
    
    Args:
        probability: Probability value in (0, 1)
        tolerance: Tolerance for convergence
        
    Returns:
        The quantile value x
    """
    # Binary search bounds
    lower_bound, upper_bound = -10.0, 10.0
    
    # Binary search with adaptive iterations
    for _ in range(100):  # Increased iterations for better precision
        midpoint = 0.5 * (lower_bound + upper_bound)
        cdf_value = standard_normal_cdf(midpoint)
        
        if abs(cdf_value - probability) < tolerance:
            return midpoint
        
        if cdf_value < probability:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    
    return 0.5 * (lower_bound + upper_bound)


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
        
    Examples:
        >>> inverse_standard_normal_cdf(0.5)
        0.0
        >>> inverse_standard_normal_cdf(0.8413447460685429)
        1.0
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be in (0, 1)")
    
    return _binary_search_inverse_cdf(probability)


def svd_keys_values(
    weight_matrix: FloatArray, 
    energy_keep: float = 1.0, 
    min_sigma: float = 0.0
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
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
        
    Examples:
        >>> weight_matrix = np.random.randn(10, 8)
        >>> U, S, V, K = svd_keys_values(weight_matrix, energy_keep=0.8)
        >>> print(f"Original rank: {min(10, 8)}, Kept rank: {len(S)}")
    """
    if not 0.0 < energy_keep <= 1.0:
        raise ValueError("energy_keep must be in (0.0, 1.0]")
    
    if min_sigma < 0.0:
        raise ValueError("min_sigma must be non-negative")
    
    # Ensure input is float32 for consistency
    weight_matrix = weight_matrix.astype(np.float32)
    
    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
    
    # Calculate cumulative energy and determine rank to keep
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    rank_keep = np.searchsorted(cumulative_energy, energy_keep, side="right") + 1
    rank_keep = min(rank_keep, len(S))
    
    # Apply minimum singular value threshold
    valid_indices = S >= min_sigma
    rank_keep = min(rank_keep, np.sum(valid_indices))
    
    # Truncate to keep only the top components
    U = U[:, :rank_keep]
    S = S[:rank_keep]
    Vt = Vt[:rank_keep, :]
    
    # Construct key matrix K = S * V^T
    K = S[:, None] * Vt
    
    return U, S, Vt.T, K


def estimate_covariance(
    calibration_data: Optional[FloatArray], 
    feature_dim: int, 
    ridge: float = 1e-6
) -> FloatArray:
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
        
    Examples:
        >>> calib_data = np.random.randn(100, 5)
        >>> cov = estimate_covariance(calib_data, 5)
        >>> print(f"Covariance shape: {cov.shape}")
    """
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    
    if calibration_data is not None:
        # Ensure input is float32
        calibration_data = calibration_data.astype(np.float32)
        
        # Check dimensions
        if calibration_data.shape[1] != feature_dim:
            raise ValueError(f"Calibration data must have {feature_dim} features")
        
        # Compute sample covariance
        n_samples = calibration_data.shape[0]
        if n_samples > 1:
            # Center the data
            centered_data = calibration_data - np.mean(calibration_data, axis=0, keepdims=True)
            covariance = (centered_data.T @ centered_data) / (n_samples - 1)
        else:
            # Single sample case
            covariance = np.zeros((feature_dim, feature_dim), dtype=np.float32)
    else:
        # Default to identity matrix
        covariance = np.eye(feature_dim, dtype=np.float32)
    
    # Add ridge regularization for numerical stability
    covariance += ridge * np.eye(feature_dim, dtype=np.float32)
    
    return covariance


def whiten_standardize_keys(
    key_matrix: FloatArray, 
    covariance_matrix: FloatArray
) -> Tuple[FloatArray, FloatArray, FloatArray]:
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
        
    Examples:
        >>> keys = np.random.randn(10, 5)
        >>> cov = np.eye(5) + 0.1 * np.random.randn(5, 5)
        >>> L_inv, K_tilde, scales = whiten_standardize_keys(keys, cov)
    """
    # Ensure inputs are float32
    key_matrix = key_matrix.astype(np.float32)
    covariance_matrix = covariance_matrix.astype(np.float32)
    
    # Check dimensions
    n_keys, key_dim = key_matrix.shape
    if covariance_matrix.shape != (key_dim, key_dim):
        raise ValueError("Covariance matrix must be square and match key dimension")
    
    # Compute Cholesky decomposition: C = L * L^T
    try:
        L = np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, add regularization
        min_eigenval = np.linalg.eigvalsh(covariance_matrix).min()
        if min_eigenval <= 0:
            regularization = abs(min_eigenval) + 1e-6
            covariance_matrix += regularization * np.eye(key_dim, dtype=np.float32)
            L = np.linalg.cholesky(covariance_matrix)
    
    # Compute inverse of L
    L_inv = np.linalg.inv(L)
    
    # Whiten the keys: K_whitened = K * L_inv^T
    K_whitened = key_matrix @ L_inv.T
    
    # Standardize to unit norm
    norms = np.linalg.norm(K_whitened, axis=1, keepdims=True)
    scales = norms.squeeze()
    
    # Avoid division by zero
    scales = np.where(scales > 1e-12, scales, 1.0)
    K_tilde = K_whitened / norms
    
    return L_inv, K_tilde, scales


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
        
    Examples:
        >>> threshold = compute_search_threshold(10, 100)
        >>> print(f"Search threshold: {threshold}")
    """
    if expected_keys > total_keys:
        raise ValueError("Expected keys cannot exceed total keys")
    
    if method == "quantile":
        # Use quantile-based threshold
        fraction = max(1e-12, min(1.0 - 1e-12, 1.0 - expected_keys / total_keys))
        return float(inverse_standard_normal_cdf(fraction))
    
    elif method == "fixed":
        # Use fixed threshold
        return 0.5
    
    elif method == "adaptive":
        # Adaptive threshold based on data distribution
        return max(0.1, min(0.9, expected_keys / total_keys))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def validate_tensor(
    tensor: FloatArray, 
    expected_shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[np.dtype] = None
) -> FloatArray:
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
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if expected_shape is not None and tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")
    
    if dtype is not None:
        tensor = tensor.astype(dtype)
    
    return tensor


def safe_matrix_inverse(matrix: FloatArray, ridge: float = 1e-6) -> FloatArray:
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
    matrix = matrix.astype(np.float32)
    n = matrix.shape[0]
    
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Add regularization and try again
        regularized = matrix + ridge * np.eye(n, dtype=np.float32)
        return np.linalg.inv(regularized)
