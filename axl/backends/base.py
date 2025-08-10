"""
Base backend interface for AXL key-value storage.

This module defines the abstract base class for backend implementations
that handle key-value storage and retrieval for approximate linear layers.
"""

from __future__ import annotations
import abc
import numpy as np
from typing import (
    Sequence, Optional, Union, Dict, Any, List, Tuple, 
    Protocol, TypeVar, Generic
)

# Type aliases for better readability
ArrayType = np.ndarray
IntArray = np.ndarray  # For ID arrays
FloatArray = np.ndarray  # For float arrays
ConfigDict = Dict[str, Any]

T = TypeVar('T')


class KVBackend(abc.ABC):
    """
    Abstract base class for key-value backend implementations.
    
    This class defines the interface that all backend implementations must follow.
    Backends are responsible for storing and retrieving key-value pairs efficiently
    using various vector database techniques.
    
    Attributes:
        is_fitted: Whether the backend has been fitted with data
        key_dim: Dimension of the key vectors
        value_dim: Dimension of the value vectors
        num_keys: Number of stored keys
    """
    
    def __init__(self) -> None:
        """Initialize the backend."""
        self.is_fitted: bool = False
        self.key_dim: Optional[int] = None
        self.value_dim: Optional[int] = None
        self.num_keys: int = 0
    
    @abc.abstractmethod
    def fit_keys(self, K_tilde: FloatArray, ids: IntArray, **kwargs: Any) -> None:
        """
        Fit the backend to store key vectors.
        
        Args:
            K_tilde: Whitened and standardized key matrix of shape (n_keys, key_dim)
            ids: Array of key IDs of shape (n_keys,)
            **kwargs: Additional keyword arguments specific to the backend
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        ...
    
    @abc.abstractmethod
    def fit_values(self, U_cols: FloatArray, ids: IntArray, **kwargs: Any) -> None:
        """
        Fit the backend to store value vectors.
        
        Args:
            U_cols: Value matrix of shape (n_values, value_dim)
            ids: Array of value IDs of shape (n_values,)
            **kwargs: Additional keyword arguments specific to the backend
            
        Raises:
            ValueError: If input dimensions are invalid or keys not fitted
        """
        ...
    
    @abc.abstractmethod
    def range_search_ip(self, z: FloatArray, threshold: float) -> IntArray:
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
        ...
    
    @abc.abstractmethod
    def reconstruct_values(self, ids: Sequence[int]) -> FloatArray:
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
        ...
    
    @abc.abstractmethod
    def refine_scores(self, ids: IntArray, z: FloatArray) -> FloatArray:
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
        ...
    
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
        ...
    
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
        ...
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the backend state.
        
        Returns:
            Dictionary containing backend information
        """
        return {
            "is_fitted": self.is_fitted,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "num_keys": self.num_keys,
            "backend_type": self.__class__.__name__
        }
    
    def reset(self) -> None:
        """Reset the backend to initial state."""
        self.is_fitted = False
        self.key_dim = None
        self.value_dim = None
        self.num_keys = 0
    
    def _validate_fitted(self) -> None:
        """
        Validate that the backend has been fitted.
        
        Raises:
            RuntimeError: If backend is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Backend must be fitted before use")
    
    def _validate_key_dim(self, key_dim: int) -> None:
        """
        Validate key dimension consistency.
        
        Args:
            key_dim: Key dimension to validate
            
        Raises:
            ValueError: If key dimension is inconsistent
        """
        if self.key_dim is not None and self.key_dim != key_dim:
            raise ValueError(f"Key dimension mismatch: expected {self.key_dim}, got {key_dim}")
    
    def _validate_value_dim(self, value_dim: int) -> None:
        """
        Validate value dimension consistency.
        
        Args:
            value_dim: Value dimension to validate
            
        Raises:
            ValueError: If value dimension is inconsistent
        """
        if self.value_dim is not None and self.value_dim != value_dim:
            raise ValueError(f"Value dimension mismatch: expected {self.value_dim}, got {value_dim}")
    
    def _validate_ids(self, ids: IntArray, expected_count: int) -> None:
        """
        Validate ID array.
        
        Args:
            ids: ID array to validate
            expected_count: Expected number of IDs
            
        Raises:
            ValueError: If ID array is invalid
        """
        if len(ids) != expected_count:
            raise ValueError(f"Expected {expected_count} IDs, got {len(ids)}")
        
        if ids.min() < 0:
            raise ValueError("IDs must be non-negative")
        
        if len(np.unique(ids)) != len(ids):
            raise ValueError("IDs must be unique")


class BackendFactory(Protocol):
    """
    Protocol for backend factory functions.
    
    This protocol defines the interface for functions that create backend instances
    from configuration dictionaries.
    """
    
    def __call__(self, config: ConfigDict) -> KVBackend:
        """
        Create a backend instance from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured backend instance
        """
        ...


def create_backend_factory(backend_type: str) -> BackendFactory:
    """
    Create a backend factory for the specified backend type.
    
    Args:
        backend_type: Type of backend to create factory for
        
    Returns:
        Backend factory function
        
    Raises:
        ValueError: If backend type is not supported
    """
    if backend_type == "faiss":
        from .faiss_backend import FaissBackend
        return lambda config: FaissBackend(**config)
    elif backend_type == "nanopq":
        from .nanopq_backend import NanoPQBackend
        return lambda config: NanoPQBackend(**config)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
