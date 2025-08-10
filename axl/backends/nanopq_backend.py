
"""
NanoPQ backend implementation for AXL key-value storage.

This module provides a NanoPQ-based backend for efficient vector storage and retrieval
using Product Quantization (PQ) with optional fallback to a simple PQ implementation.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Sequence, Optional, List
from .base import KVBackend

try:
    import nanopq as npq
    HAS_NANOPQ = True
except Exception:
    HAS_NANOPQ = False


def _kmeans_lloyd(
    data: np.ndarray, 
    num_clusters: int, 
    max_iterations: int = 25, 
    random_seed: int = 0
) -> np.ndarray:
    """
    Perform K-means clustering using Lloyd's algorithm.
    
    Args:
        data: Input data matrix of shape (n_samples, n_features)
        num_clusters: Number of clusters to find
        max_iterations: Maximum number of iterations (default: 25)
        random_seed: Random seed for reproducibility (default: 0)
        
    Returns:
        Cluster centers of shape (num_clusters, n_features)
    """
    rng = np.random.default_rng(random_seed)
    n_samples, n_features = data.shape
    
    # Initialize cluster centers
    centers = np.empty((num_clusters, n_features), dtype=data.dtype)
    centers[0] = data[rng.integers(0, n_samples)]
    
    # Compute distances to first center
    distances = np.sum((data - centers[0])**2, axis=1)
    
    # K-means++ initialization
    for k in range(1, num_clusters):
        probabilities = distances / (np.sum(distances) + 1e-12)
        center_idx = rng.choice(n_samples, p=probabilities)
        centers[k] = data[center_idx]
        new_distances = np.sum((data - centers[k])**2, axis=1)
        distances = np.minimum(distances, new_distances)
    
    # Lloyd's algorithm iterations
    for _ in range(max_iterations):
        # Compute distances to all centers
        squared_norms_data = np.sum(data**2, axis=1, keepdims=True)
        squared_norms_centers = np.sum(centers**2, axis=1, keepdims=True).T
        distances_matrix = (
            squared_norms_data - 2 * data @ centers.T + squared_norms_centers
        )
        
        # Assign points to nearest centers
        cluster_assignments = np.argmin(distances_matrix, axis=1)
        
        # Update centers
        for k in range(num_clusters):
            cluster_mask = (cluster_assignments == k)
            if np.any(cluster_mask):
                centers[k] = data[cluster_mask].mean(axis=0)
            else:
                # If cluster is empty, randomly select a new center
                centers[k] = data[rng.integers(0, n_samples)]
    
    return centers


class _SimplePQ:
    """
    Simple Product Quantization implementation as fallback when NanoPQ is not available.
    
    This class provides a basic implementation of Product Quantization for vector
    compression and reconstruction.
    """
    
    def __init__(self, M: int = 8, Ks: int = 256, max_iterations: int = 25, random_seed: int = 0):
        """
        Initialize the simple PQ implementation.
        
        Args:
            M: Number of subvectors (default: 8)
            Ks: Number of codewords per subvector (default: 256)
            max_iterations: Maximum iterations for k-means (default: 25)
            random_seed: Random seed for reproducibility (default: 0)
        """
        self.M = M
        self.Ks = Ks
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.codebooks: Optional[List[np.ndarray]] = None

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the PQ codebooks to the data.
        
        Args:
            data: Input data matrix of shape (n_samples, n_features)
        """
        n_samples, n_features = data.shape
        assert n_features % self.M == 0, f"Feature dimension {n_features} must be divisible by M={self.M}"
        
        subvector_dim = n_features // self.M
        self.codebooks = []
        
        # Train codebooks for each subvector
        for j in range(self.M):
            start_idx = j * subvector_dim
            end_idx = (j + 1) * subvector_dim
            subvector_data = data[:, start_idx:end_idx]
            
            codebook = _kmeans_lloyd(
                subvector_data, 
                self.Ks, 
                self.max_iterations, 
                self.random_seed + 7 * j
            )
            self.codebooks.append(codebook.astype(np.float32))

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data using the trained codebooks.
        
        Args:
            data: Input data matrix of shape (n_samples, n_features)
            
        Returns:
            Encoded codes of shape (n_samples, M)
        """
        n_samples, n_features = data.shape
        subvector_dim = n_features // self.M
        codes = np.empty((n_samples, self.M), dtype=np.int32)
        
        # Encode each subvector
        for j, codebook in enumerate(self.codebooks):
            start_idx = j * subvector_dim
            end_idx = (j + 1) * subvector_dim
            subvector_data = data[:, start_idx:end_idx]
            
            # Compute distances to codewords
            squared_norms_data = np.sum(subvector_data**2, axis=1, keepdims=True)
            squared_norms_codebook = np.sum(codebook**2, axis=1, keepdims=True).T
            distances = squared_norms_data - 2 * subvector_data @ codebook.T + squared_norms_codebook
            
            # Assign to nearest codeword
            codes[:, j] = np.argmin(distances, axis=1)
        
        return codes

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode codes back to vectors.
        
        Args:
            codes: Encoded codes of shape (n_samples, M)
            
        Returns:
            Decoded vectors of shape (n_samples, n_features)
        """
        n_samples = codes.shape[0]
        subvector_dim = self.codebooks[0].shape[1]
        n_features = subvector_dim * self.M
        
        decoded_vectors = np.zeros((n_samples, n_features), dtype=np.float32)
        
        # Decode each subvector
        for i in range(n_samples):
            for j, codebook in enumerate(self.codebooks):
                start_idx = j * subvector_dim
                end_idx = (j + 1) * subvector_dim
                decoded_vectors[i, start_idx:end_idx] = codebook[codes[i, j]]
        
        return decoded_vectors


class NanoPQBackend(KVBackend):
    """
    NanoPQ-based backend for key-value storage using Product Quantization.
    
    This backend uses the NanoPQ library (or a simple fallback implementation)
    to provide efficient vector storage and retrieval through Product Quantization.
    
    Attributes:
        M_keys: Number of subvectors for key PQ
        Ks_keys: Number of codewords per subvector for key PQ
        M_vals: Number of subvectors for value PQ
        Ks_vals: Number of codewords per subvector for value PQ
        max_iterations: Maximum iterations for k-means clustering
    """
    
    def __init__(
        self, 
        M_keys: int = 8, 
        Ks_keys: int = 256, 
        M_vals: int = 8, 
        Ks_vals: int = 256, 
        max_iterations: int = 25
    ):
        """
        Initialize the NanoPQ backend.
        
        Args:
            M_keys: Number of subvectors for key PQ (default: 8)
            Ks_keys: Number of codewords per subvector for key PQ (default: 256)
            M_vals: Number of subvectors for value PQ (default: 8)
            Ks_vals: Number of codewords per subvector for value PQ (default: 256)
            max_iterations: Maximum iterations for k-means (default: 25)
        """
        super().__init__()  # Call parent constructor
        
        self.M_keys = M_keys
        self.Ks_keys = Ks_keys
        self.M_vals = M_vals
        self.Ks_vals = Ks_vals
        self.max_iterations = max_iterations
        
        # Internal state
        self.pq_keys: Optional[object] = None
        self.pq_vals: Optional[object] = None
        self.codes_keys: Optional[np.ndarray] = None
        self.codes_vals: Optional[np.ndarray] = None
        self.Kt_shape: Optional[tuple] = None
        self.V_shape: Optional[tuple] = None
        self._store_float_keys: Optional[np.ndarray] = None

    def fit_keys(
        self, 
        K_tilde: np.ndarray, 
        ids: np.ndarray, 
        store_float_keys: bool = False, 
        **kwargs
    ) -> None:
        """
        Fit the PQ codebooks for key storage.
        
        Args:
            K_tilde: Whitened and standardized key matrix
            ids: Array of key IDs
            store_float_keys: Whether to store original float keys for refinement
            **kwargs: Additional keyword arguments (ignored)
        """
        self.Kt_shape = K_tilde.shape
        
        if HAS_NANOPQ:
            # Use NanoPQ library
            pq = npq.PQ(M=self.M_keys, Ks=self.Ks_keys)
            pq.fit(K_tilde)
            codes = pq.encode(K_tilde)
            self.pq_keys = pq
        else:
            # Use simple PQ implementation
            pq = _SimplePQ(
                M=self.M_keys, 
                Ks=self.Ks_keys, 
                max_iterations=self.max_iterations
            )
            pq.fit(K_tilde)
            codes = pq.encode(K_tilde)
            self.pq_keys = pq
        
        # Store codes with the original dtype from NanoPQ
        self.codes_keys = codes
        self._store_float_keys = (
            K_tilde.astype(np.float32) if store_float_keys else None
        )

    def fit_values(
        self, 
        U_cols: np.ndarray, 
        ids: np.ndarray, 
        **kwargs
    ) -> None:
        """
        Fit the PQ codebooks for value storage.
        
        Args:
            U_cols: Value matrix
            ids: Array of value IDs
            **kwargs: Additional keyword arguments (ignored)
        """
        self.V_shape = U_cols.shape
        
        if HAS_NANOPQ:
            # Use NanoPQ library
            pq = npq.PQ(M=self.M_vals, Ks=self.Ks_vals)
            pq.fit(U_cols)
            codes = pq.encode(U_cols)
            self.pq_vals = pq
        else:
            # Use simple PQ implementation
            pq = _SimplePQ(
                M=self.M_vals, 
                Ks=self.Ks_vals, 
                max_iterations=self.max_iterations
            )
            pq.fit(U_cols)
            codes = pq.encode(U_cols)
            self.pq_vals = pq
        
        # Store codes with the original dtype from NanoPQ
        self.codes_vals = codes

    def range_search_ip(self, z: np.ndarray, threshold: float) -> np.ndarray:
        """
        Perform range search using inner product similarity.
        
        Args:
            z: Query vector
            threshold: Similarity threshold
            
        Returns:
            Array of key IDs that meet the similarity threshold
        """
        rank, key_dim = self.Kt_shape
        M = self.pq_keys.M if HAS_NANOPQ else self.pq_keys.M
        subvector_dim = key_dim // M
        
        # Compute approximate inner products
        scores = np.zeros(rank, dtype=np.float32)
        codebooks = (
            self.pq_keys.codewords if HAS_NANOPQ else self.pq_keys.codebooks
        )
        
        # Compute scores for each subvector
        for j in range(M):
            codebook = codebooks[j]
            z_subvector = z[j * subvector_dim:(j + 1) * subvector_dim]
            inner_products = codebook @ z_subvector
            scores += inner_products[self.codes_keys[:, j]]
        
        # Return indices where scores meet threshold
        return np.where(scores >= float(threshold))[0].astype(np.int64)

    def reconstruct_values(self, ids: Sequence[int]) -> np.ndarray:
        """
        Reconstruct value vectors from their IDs.
        
        Args:
            ids: Sequence of value IDs to reconstruct
            
        Returns:
            Reconstructed value matrix
        """
        codes = self.codes_vals[np.array(ids, dtype=np.int64)]
        
        if HAS_NANOPQ:
            return self.pq_vals.decode(codes)
        else:
            return self.pq_vals.decode_codes(codes)

    def refine_scores(self, ids: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Refine similarity scores for given key IDs.
        
        Args:
            ids: Array of key IDs
            z: Query vector
            
        Returns:
            Refined similarity scores for each key ID
        """
        if self._store_float_keys is not None:
            # Use stored float keys for exact computation
            return (self._store_float_keys[ids] @ z.astype(np.float32))
        
        # Use approximate computation with PQ
        rank, key_dim = self.Kt_shape
        M = self.pq_keys.M if HAS_NANOPQ else self.pq_keys.M
        subvector_dim = key_dim // M
        
        scores = np.zeros(ids.shape[0], dtype=np.float32)
        codebooks = (
            self.pq_keys.codewords if HAS_NANOPQ else self.pq_keys.codebooks
        )
        
        # Compute scores for each subvector
        for j in range(M):
            codebook = codebooks[j]
            z_subvector = z[j * subvector_dim:(j + 1) * subvector_dim]
            inner_products = codebook @ z_subvector
            scores += inner_products[self.codes_keys[ids, j]]
        
        return scores

    def save(self, directory: str) -> None:
        """
        Save the NanoPQ backend state to disk.
        
        Args:
            directory: Directory path where to save the backend state
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save codes
        np.save(os.path.join(directory, "codes_keys.npy"), self.codes_keys)
        np.save(os.path.join(directory, "codes_vals.npy"), self.codes_vals)
        
        # Save metadata
        metadata = {
            "M_keys": self.M_keys,
            "Ks_keys": self.Ks_keys,
            "M_vals": self.M_vals,
            "Ks_vals": self.Ks_vals,
            "Kt_shape": self.Kt_shape,
            "V_shape": self.V_shape,
            "has_float_keys": self._store_float_keys is not None
        }
        np.save(os.path.join(directory, "meta.npy"), metadata, allow_pickle=True)
        
        # Save codebooks
        if HAS_NANOPQ:
            key_codebooks = [cw.astype(np.float32) for cw in self.pq_keys.codewords]
            val_codebooks = [cw.astype(np.float32) for cw in self.pq_vals.codewords]
        else:
            key_codebooks = [C.astype(np.float32) for C in self.pq_keys.codebooks]
            val_codebooks = [C.astype(np.float32) for C in self.pq_vals.codebooks]
        
        np.save(
            os.path.join(directory, "key_codebooks.npy"), 
            np.array(key_codebooks, dtype=object), 
            allow_pickle=True
        )
        np.save(
            os.path.join(directory, "val_codebooks.npy"), 
            np.array(val_codebooks, dtype=object), 
            allow_pickle=True
        )
        
        # Save float keys if available
        if self._store_float_keys is not None:
            np.save(os.path.join(directory, "K_tilde.npy"), self._store_float_keys)

    def load(self, directory: str) -> None:
        """
        Load the NanoPQ backend state from disk.
        
        Args:
            directory: Directory path from where to load the backend state
        """
        # Load codes
        self.codes_keys = np.load(os.path.join(directory, "codes_keys.npy"))
        self.codes_vals = np.load(os.path.join(directory, "codes_vals.npy"))
        
        # Load metadata
        metadata = np.load(os.path.join(directory, "meta.npy"), allow_pickle=True).item()
        self.M_keys = metadata["M_keys"]
        self.Ks_keys = metadata["Ks_keys"]
        self.M_vals = metadata["M_vals"]
        self.Ks_vals = metadata["Ks_vals"]
        self.Kt_shape = tuple(metadata["Kt_shape"])
        self.V_shape = tuple(metadata["V_shape"])
        has_float_keys = bool(metadata["has_float_keys"])
        
        # Load codebooks
        key_codebooks = np.load(
            os.path.join(directory, "key_codebooks.npy"), 
            allow_pickle=True
        )
        val_codebooks = np.load(
            os.path.join(directory, "val_codebooks.npy"), 
            allow_pickle=True
        )
        
        if HAS_NANOPQ:
            # Reconstruct NanoPQ objects
            import nanopq as npq
            self.pq_keys = npq.PQ(M=self.M_keys, Ks=self.Ks_keys)
            self.pq_vals = npq.PQ(M=self.M_vals, Ks=self.Ks_vals)
            self.pq_keys.codewords = [
                np.array(cw, dtype=np.float32) for cw in key_codebooks
            ]
            self.pq_vals.codewords = [
                np.array(cw, dtype=np.float32) for cw in val_codebooks
            ]
        else:
            # Reconstruct simple PQ objects
            self.pq_keys = _SimplePQ(M=self.M_keys, Ks=self.Ks_keys)
            self.pq_vals = _SimplePQ(M=self.M_vals, Ks=self.Ks_vals)
            self.pq_keys.codebooks = [
                np.array(cw, dtype=np.float32) for cw in key_codebooks
            ]
            self.pq_vals.codebooks = [
                np.array(cw, dtype=np.float32) for cw in val_codebooks
            ]
        
        # Load float keys if available
        if has_float_keys:
            self._store_float_keys = np.load(
                os.path.join(directory, "K_tilde.npy")
            ).astype(np.float32)
        else:
            self._store_float_keys = None
