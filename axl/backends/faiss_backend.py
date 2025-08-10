"""
FAISS backend implementation for AXL key-value storage.

This module provides a FAISS-based backend for efficient vector storage and retrieval
using Product Quantization (PQ) and Inverted File with Product Quantization (IVFPQ).
"""

from __future__ import annotations
import os
import numpy as np
from typing import Sequence, Optional
from .base import KVBackend

try:
    import faiss
    HAS_FAISS = True
except ImportError as e:
    HAS_FAISS = False
    raise ImportError(
        "FAISS backend requested but faiss is not installed. "
        "Install with: pip install faiss-cpu"
    ) from e


class FaissBackend(KVBackend):
    """
    FAISS-based backend for key-value storage using vector quantization.
    
    This backend uses FAISS library to implement efficient vector storage and retrieval
    through Product Quantization (PQ) and Inverted File with Product Quantization (IVFPQ).
    
    Attributes:
        nlist: Number of clusters for IVFPQ
        m_pq_keys: Number of subvectors for key PQ
        nbits_keys: Number of bits per subvector for key PQ
        nprobe: Number of clusters to probe during search
        values_index_type: Type of index for values ("PQ" or scalar quantization)
        m_pq_vals: Number of subvectors for value PQ
        nbits_vals: Number of bits per subvector for value PQ
        sq_type: Scalar quantization type for values
        use_gpu: Whether to use GPU acceleration
    """
    
    def __init__(
        self,
        nlist: int = 4096,
        m_pq_keys: int = 8,
        nbits_keys: int = 8,
        nprobe: int = 64,
        values_index_type: str = "PQ",
        m_pq_vals: int = 8,
        nbits_vals: int = 8,
        sq_type: str = "QT_8bit",
        use_gpu: bool = False
    ):
        """
        Initialize the FAISS backend.
        
        Args:
            nlist: Number of clusters for IVFPQ (default: 4096)
            m_pq_keys: Number of subvectors for key PQ (default: 8)
            nbits_keys: Number of bits per subvector for key PQ (default: 8)
            nprobe: Number of clusters to probe during search (default: 64)
            values_index_type: Type of index for values (default: "PQ")
            m_pq_vals: Number of subvectors for value PQ (default: 8)
            nbits_vals: Number of bits per subvector for value PQ (default: 8)
            sq_type: Scalar quantization type for values (default: "QT_8bit")
            use_gpu: Whether to use GPU acceleration (default: False)
        """
        super().__init__()  # Call parent constructor
        
        self.nlist = nlist
        self.m_pq_keys = m_pq_keys
        self.nbits_keys = nbits_keys
        self.nprobe = nprobe
        self.values_index_type = values_index_type
        self.m_pq_vals = m_pq_vals
        self.nbits_vals = nbits_vals
        self.sq_type = sq_type
        self.use_gpu = use_gpu
        
        # Internal state
        self.key_index: Optional[faiss.Index] = None
        self.val_index: Optional[faiss.Index] = None
        self._K_tilde_float: Optional[np.ndarray] = None

    def fit_keys(
        self, 
        K_tilde: np.ndarray, 
        ids: np.ndarray, 
        store_float_keys: bool = False, 
        **kwargs
    ) -> None:
        """
        Fit the FAISS index for key storage using IVFPQ.
        
        Args:
            K_tilde: Whitened and standardized key matrix
            ids: Array of key IDs
            store_float_keys: Whether to store original float keys for refinement
            **kwargs: Additional keyword arguments (ignored)
        """
        key_dim = K_tilde.shape[1]
        
        # Create quantizer (flat index for inner product)
        quantizer = faiss.IndexFlatIP(key_dim)
        
        # Create IVFPQ index
        index = faiss.IndexIVFPQ(
            quantizer, 
            key_dim, 
            self.nlist, 
            self.m_pq_keys, 
            self.nbits_keys
        )
        
        # Train and add keys
        index.train(K_tilde.astype(np.float32))
        index.add_with_ids(K_tilde.astype(np.float32), ids.astype(np.int64))
        
        # Configure search parameters
        index.nprobe = self.nprobe
        index.make_direct_map()
        
        # Move to GPU if requested
        if self.use_gpu:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
            index.nprobe = self.nprobe
        
        self.key_index = index
        
        # Store original float keys if requested
        self._K_tilde_float = (
            K_tilde.astype(np.float32) if store_float_keys else None
        )

    def fit_values(
        self, 
        U_cols: np.ndarray, 
        ids: np.ndarray, 
        **kwargs
    ) -> None:
        """
        Fit the FAISS index for value storage.
        
        Args:
            U_cols: Value matrix
            ids: Array of value IDs
            **kwargs: Additional keyword arguments (ignored)
        """
        rank, value_dim = U_cols.shape
        values = U_cols.astype(np.float32)
        
        if self.values_index_type.upper() == "PQ":
            # Use Product Quantization for values
            index = faiss.IndexPQ(
                value_dim,
                kwargs.get("m_pq_vals", self.m_pq_vals),
                kwargs.get("nbits_vals", self.nbits_vals)
            )
            index.train(values)
            index.add_with_ids(values, ids.astype(np.int64))
        else:
            # Use Scalar Quantization for values
            qt_type = getattr(
                faiss, 
                kwargs.get("sq_type", self.sq_type), 
                faiss.ScalarQuantizer.QT_8bit
            )
            index = faiss.IndexScalarQuantizer(
                value_dim, 
                qt_type, 
                faiss.METRIC_L2
            )
            index.train(values)
            index.add_with_ids(values, ids.astype(np.int64))
        
        self.val_index = index

    def range_search_ip(self, z: np.ndarray, threshold: float) -> np.ndarray:
        """
        Perform range search using inner product similarity.
        
        Args:
            z: Query vector
            threshold: Similarity threshold
            
        Returns:
            Array of key IDs that meet the similarity threshold
        """
        # Reshape query for FAISS (add batch dimension)
        query = z[None, :].astype(np.float32)
        
        # Perform range search
        limits, distances, indices = self.key_index.range_search(
            query, 
            float(threshold)
        )
        
        # Extract results
        start_idx, end_idx = limits[0], limits[1]
        return indices[start_idx:end_idx].astype(np.int64)

    def reconstruct_values(self, ids: Sequence[int]) -> np.ndarray:
        """
        Reconstruct value vectors from their IDs.
        
        Args:
            ids: Sequence of value IDs to reconstruct
            
        Returns:
            Reconstructed value matrix
        """
        reconstructed_values = []
        for value_id in ids:
            reconstructed_values.append(self.val_index.reconstruct(int(value_id)))
        return np.array(reconstructed_values, dtype=np.float32)

    def refine_scores(self, ids: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Refine similarity scores for given key IDs.
        
        Args:
            ids: Array of key IDs
            z: Query vector
            
        Returns:
            Refined similarity scores for each key ID
        """
        query_vector = z.astype(np.float32)
        scores = np.zeros(ids.shape[0], dtype=np.float32)
        
        if self._K_tilde_float is not None:
            # Use stored float keys for exact computation
            for i, key_id in enumerate(ids):
                scores[i] = float(np.dot(self._K_tilde_float[int(key_id)], query_vector))
        else:
            # Reconstruct keys from index for computation
            # Convert to CPU index if on GPU
            cpu_index = (
                faiss.index_gpu_to_cpu(self.key_index) 
                if isinstance(self.key_index, faiss.GpuIndex) 
                else self.key_index
            )
            
            for i, key_id in enumerate(ids):
                reconstructed_key = cpu_index.reconstruct(int(key_id))
                scores[i] = float(np.dot(reconstructed_key, query_vector))
        
        return scores

    def save(self, directory: str) -> None:
        """
        Save the FAISS backend state to disk.
        
        Args:
            directory: Directory path where to save the backend state
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save key index (convert to CPU if on GPU)
        cpu_key_index = (
            faiss.index_gpu_to_cpu(self.key_index) 
            if isinstance(self.key_index, faiss.GpuIndex) 
            else self.key_index
        )
        faiss.write_index(cpu_key_index, os.path.join(directory, "keys.faiss"))
        
        # Save value index
        faiss.write_index(self.val_index, os.path.join(directory, "values.faiss"))
        
        # Save float keys if available
        if self._K_tilde_float is not None:
            np.save(os.path.join(directory, "K_tilde.npy"), self._K_tilde_float)

    def load(self, directory: str) -> None:
        """
        Load the FAISS backend state from disk.
        
        Args:
            directory: Directory path from where to load the backend state
        """
        # Load key index
        self.key_index = faiss.read_index(os.path.join(directory, "keys.faiss"))
        
        # Load value index
        self.val_index = faiss.read_index(os.path.join(directory, "values.faiss"))
        
        # Load float keys if available
        float_keys_path = os.path.join(directory, "K_tilde.npy")
        if os.path.exists(float_keys_path):
            self._K_tilde_float = np.load(float_keys_path).astype(np.float32)
        else:
            self._K_tilde_float = None
