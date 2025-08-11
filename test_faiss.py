#!/usr/bin/env python3
"""
Independent FAISS Testing Script

This script tests FAISS functionality independently of the AXL framework.
It creates a small 100x100 matrix and tests various FAISS configurations
to ensure FAISS works correctly with different settings.
"""

import numpy as np
import faiss
import time
from typing import Dict, Any, Tuple


def create_test_data(matrix_size: int = 100, calibration_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test data for FAISS testing.
    
    Args:
        matrix_size: Size of the main matrix (matrix_size x matrix_size)
        calibration_size: Number of calibration vectors
        
    Returns:
        Tuple of (weight_matrix, calibration_data)
    """
    print(f"Creating test data: {matrix_size}x{matrix_size} matrix, {calibration_size} calibration vectors")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create weight matrix (simulating a layer weight matrix)
    weight_matrix = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    
    # Create calibration data (simulating input data for initialization)
    calibration_data = np.random.randn(calibration_size, matrix_size).astype(np.float32)
    
    print(f"Weight matrix shape: {weight_matrix.shape}")
    print(f"Calibration data shape: {calibration_data.shape}")
    
    return weight_matrix, calibration_data


def get_divisors(n: int, max_divisor: int = 16) -> list:
    """
    Get all divisors of n up to max_divisor.
    
    Args:
        n: Number to find divisors for
        max_divisor: Maximum divisor to consider
        
    Returns:
        List of divisors
    """
    divisors = []
    for i in range(1, min(n + 1, max_divisor + 1)):
        if n % i == 0:
            divisors.append(i)
    return divisors


def test_faiss_basic_functionality(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """
    Test basic FAISS functionality with simple settings.
    
    Args:
        weight_matrix: Weight matrix to test
        calibration_data: Calibration data for initialization
        
    Returns:
        Dictionary with test results
    """
    print("\n=== Testing Basic FAISS Functionality ===")
    
    results = {}
    
    try:
        # Test 1: Simple IndexFlatIP (exact inner product)
        print("Test 1: IndexFlatIP (exact inner product)")
        start_time = time.time()
        
        # Create flat index for inner product
        index_flat = faiss.IndexFlatIP(weight_matrix.shape[1])
        index_flat.add(weight_matrix.astype(np.float32))
        
        # Test search
        query = calibration_data[0:1]  # Single query vector
        distances, indices = index_flat.search(query, k=5)
        
        flat_time = time.time() - start_time
        results['flat_ip'] = {
            'success': True,
            'time': flat_time,
            'distances': distances[0],
            'indices': indices[0]
        }
        print(f"✓ IndexFlatIP: Found {len(indices[0])} results in {flat_time:.4f}s")
        
    except Exception as e:
        results['flat_ip'] = {'success': False, 'error': str(e)}
        print(f"✗ IndexFlatIP failed: {e}")
    
    # Test 2: IndexPQ (Product Quantization) with proper dimension handling
    print("Test 2: IndexPQ (Product Quantization)")
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    
    if valid_m_values:
        m = valid_m_values[-1]  # Use the largest valid divisor
        # Use fewer bits to avoid requiring too many training points
        nbits = 4  # bits per subvector (2^4 = 16 clusters, needs at least 16 training points)
        
        try:
            start_time = time.time()
            
            # Create PQ index
            index_pq = faiss.IndexPQ(dimension, m, nbits)
            
            # Train and add vectors
            index_pq.train(weight_matrix.astype(np.float32))
            index_pq.add(weight_matrix.astype(np.float32))
            
            # Test search
            query = calibration_data[0:1]
            distances, indices = index_pq.search(query, k=5)
            
            pq_time = time.time() - start_time
            results['pq'] = {
                'success': True,
                'time': pq_time,
                'distances': distances[0],
                'indices': indices[0],
                'm': m,
                'nbits': nbits
            }
            print(f"✓ IndexPQ: Found {len(indices[0])} results in {pq_time:.4f}s (m={m}, nbits={nbits})")
            
        except Exception as e:
            results['pq'] = {'success': False, 'error': str(e)}
            print(f"✗ IndexPQ failed: {e}")
    else:
        results['pq'] = {'success': False, 'error': f'No valid divisors for dimension {dimension}'}
        print(f"✗ IndexPQ: No valid divisors for dimension {dimension}")
    
    return results


def test_faiss_ivfpq(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """
    Test FAISS IVFPQ (Inverted File with Product Quantization).
    
    Args:
        weight_matrix: Weight matrix to test
        calibration_data: Calibration data for initialization
        
    Returns:
        Dictionary with test results
    """
    print("\n=== Testing FAISS IVFPQ ===")
    
    results = {}
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    
    # Test different IVFPQ configurations with valid parameters
    configs = [
        {'nlist': 4, 'm': valid_m_values[0] if valid_m_values else 4, 'nbits': 4, 'nprobe': 2, 'name': 'Small'},
        {'nlist': 8, 'm': valid_m_values[1] if len(valid_m_values) > 1 else valid_m_values[0] if valid_m_values else 4, 'nbits': 4, 'nprobe': 4, 'name': 'Medium'},
        {'nlist': 16, 'm': valid_m_values[-1] if valid_m_values else 4, 'nbits': 4, 'nprobe': 8, 'name': 'Large'}
    ]
    
    for config in configs:
        try:
            print(f"Testing IVFPQ {config['name']} config: nlist={config['nlist']}, m={config['m']}, nbits={config['nbits']}, nprobe={config['nprobe']}")
            start_time = time.time()
            
            # Create quantizer
            quantizer = faiss.IndexFlatIP(dimension)
            
            # Create IVFPQ index
            index_ivfpq = faiss.IndexIVFPQ(
                quantizer, 
                dimension, 
                config['nlist'], 
                config['m'], 
                config['nbits']
            )
            
            # Train and add vectors
            index_ivfpq.train(weight_matrix.astype(np.float32))
            index_ivfpq.add(weight_matrix.astype(np.float32))
            
            # Set nprobe
            index_ivfpq.nprobe = config['nprobe']
            
            # Test search
            query = calibration_data[0:1]
            distances, indices = index_ivfpq.search(query, k=5)
            
            ivfpq_time = time.time() - start_time
            results[f'ivfpq_{config["name"].lower()}'] = {
                'success': True,
                'time': ivfpq_time,
                'distances': distances[0],
                'indices': indices[0],
                'config': config
            }
            print(f"✓ IVFPQ {config['name']}: Found {len(indices[0])} results in {ivfpq_time:.4f}s")
            
        except Exception as e:
            results[f'ivfpq_{config["name"].lower()}'] = {'success': False, 'error': str(e), 'config': config}
            print(f"✗ IVFPQ {config['name']} failed: {e}")
    
    return results


def test_faiss_range_search(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """
    Test FAISS range search functionality.
    
    Args:
        weight_matrix: Weight matrix to test
        calibration_data: Calibration data for initialization
        
    Returns:
        Dictionary with test results
    """
    print("\n=== Testing FAISS Range Search ===")
    
    results = {}
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    m = valid_m_values[0] if valid_m_values else 4
    
    try:
        # Create IVFPQ index for range search
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, 8, m, 4)
        
        # Train and add vectors
        index.train(weight_matrix.astype(np.float32))
        index.add(weight_matrix.astype(np.float32))
        index.nprobe = 4
        index.make_direct_map()  # Required for range search
        
        # Test range search with different thresholds
        query = calibration_data[0:1]
        thresholds = [0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            try:
                start_time = time.time()
                limits, distances, indices = index.range_search(query, threshold)
                range_time = time.time() - start_time
                
                # Extract results
                start_idx, end_idx = limits[0], limits[1]
                result_indices = indices[start_idx:end_idx]
                result_distances = distances[start_idx:end_idx]
                
                results[f'range_search_{threshold}'] = {
                    'success': True,
                    'time': range_time,
                    'threshold': threshold,
                    'num_results': len(result_indices),
                    'indices': result_indices,
                    'distances': result_distances
                }
                print(f"✓ Range search (threshold={threshold}): Found {len(result_indices)} results in {range_time:.4f}s")
                
            except Exception as e:
                results[f'range_search_{threshold}'] = {'success': False, 'error': str(e), 'threshold': threshold}
                print(f"✗ Range search (threshold={threshold}) failed: {e}")
                
    except Exception as e:
        results['range_search'] = {'success': False, 'error': str(e)}
        print(f"✗ Range search setup failed: {e}")
    
    return results


def test_faiss_reconstruction(weight_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Test FAISS vector reconstruction functionality.
    
    Args:
        weight_matrix: Weight matrix to test
        
    Returns:
        Dictionary with test results
    """
    print("\n=== Testing FAISS Reconstruction ===")
    
    results = {}
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    m = valid_m_values[0] if valid_m_values else 4
    
    try:
        # Create PQ index
        index = faiss.IndexPQ(dimension, m, 4)  # Use 4 bits instead of 8
        index.train(weight_matrix.astype(np.float32))
        index.add(weight_matrix.astype(np.float32))
        
        # Test reconstruction of a few vectors
        test_indices = [0, 10, 50]
        original_vectors = weight_matrix[test_indices]
        reconstructed_vectors = []
        
        start_time = time.time()
        for idx in test_indices:
            reconstructed = index.reconstruct(idx)
            reconstructed_vectors.append(reconstructed)
        recon_time = time.time() - start_time
        
        reconstructed_vectors = np.array(reconstructed_vectors)
        
        # Calculate reconstruction error
        reconstruction_errors = np.linalg.norm(original_vectors - reconstructed_vectors, axis=1)
        mean_error = np.mean(reconstruction_errors)
        
        results['reconstruction'] = {
            'success': True,
            'time': recon_time,
            'mean_error': mean_error,
            'errors': reconstruction_errors.tolist(),
            'original_shape': original_vectors.shape,
            'reconstructed_shape': reconstructed_vectors.shape,
            'm': m
        }
        print(f"✓ Reconstruction: Mean error {mean_error:.6f} in {recon_time:.4f}s (m={m})")
        
    except Exception as e:
        results['reconstruction'] = {'success': False, 'error': str(e)}
        print(f"✗ Reconstruction failed: {e}")
    
    return results


def test_faiss_gpu_support() -> Dict[str, Any]:
    """
    Test FAISS GPU support if available.
    
    Returns:
        Dictionary with test results
    """
    print("\n=== Testing FAISS GPU Support ===")
    
    results = {}
    
    try:
        # Check if GPU is available
        if not faiss.get_num_gpus():
            results['gpu'] = {'success': False, 'error': 'No GPU available'}
            print("✗ No GPU available for FAISS")
            return results
        
        print(f"Found {faiss.get_num_gpus()} GPU(s)")
        
        # Create small test data
        test_data = np.random.randn(50, 32).astype(np.float32)
        query = np.random.randn(1, 32).astype(np.float32)
        
        # Create CPU index
        cpu_index = faiss.IndexFlatIP(32)
        cpu_index.add(test_data)
        
        # Create GPU index
        gpu_resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
        
        # Test search on both CPU and GPU
        start_time = time.time()
        cpu_distances, cpu_indices = cpu_index.search(query, k=5)
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        gpu_distances, gpu_indices = gpu_index.search(query, k=5)
        gpu_time = time.time() - start_time
        
        # Check if results are the same
        results_match = np.array_equal(cpu_indices, gpu_indices)
        
        results['gpu'] = {
            'success': True,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time if gpu_time > 0 else float('inf'),
            'results_match': results_match,
            'cpu_indices': cpu_indices[0].tolist(),
            'gpu_indices': gpu_indices[0].tolist()
        }
        print(f"✓ GPU test: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={cpu_time/gpu_time:.2f}x, Results match: {results_match}")
        
    except Exception as e:
        results['gpu'] = {'success': False, 'error': str(e)}
        print(f"✗ GPU test failed: {e}")
    
    return results


def test_faiss_scalar_quantization(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """
    Test FAISS Scalar Quantization functionality.
    
    Args:
        weight_matrix: Weight matrix to test
        calibration_data: Calibration data for initialization
        
    Returns:
        Dictionary with test results
    """
    print("\n=== Testing FAISS Scalar Quantization ===")
    
    results = {}
    
    try:
        # Create Scalar Quantization index
        dimension = weight_matrix.shape[1]
        index_sq = faiss.IndexScalarQuantizer(
            dimension, 
            faiss.ScalarQuantizer.QT_8bit, 
            faiss.METRIC_L2
        )
        
        # Train and add vectors
        index_sq.train(weight_matrix.astype(np.float32))
        index_sq.add(weight_matrix.astype(np.float32))
        
        # Test search
        query = calibration_data[0:1]
        start_time = time.time()
        distances, indices = index_sq.search(query, k=5)
        sq_time = time.time() - start_time
        
        results['scalar_quantization'] = {
            'success': True,
            'time': sq_time,
            'distances': distances[0],
            'indices': indices[0]
        }
        print(f"✓ Scalar Quantization: Found {len(indices[0])} results in {sq_time:.4f}s")
        
    except Exception as e:
        results['scalar_quantization'] = {'success': False, 'error': str(e)}
        print(f"✗ Scalar Quantization failed: {e}")
    
    return results


def print_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a summary of all test results.
    
    Args:
        all_results: Dictionary containing all test results
    """
    print("\n" + "="*60)
    print("FAISS TEST SUMMARY")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in all_results.items():
        total_tests += 1
        if result.get('success', False):
            passed_tests += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            if 'error' in result:
                print(f"{test_name:30} {status} - {result['error']}")
            else:
                print(f"{test_name:30} {status}")
            continue
        
        # Print additional info for successful tests
        info_parts = []
        if 'time' in result:
            info_parts.append(f"time={result['time']:.4f}s")
        if 'num_results' in result:
            info_parts.append(f"results={result['num_results']}")
        if 'mean_error' in result:
            info_parts.append(f"error={result['mean_error']:.6f}")
        if 'speedup' in result:
            info_parts.append(f"speedup={result['speedup']:.2f}x")
        if 'm' in result:
            info_parts.append(f"m={result['m']}")
        
        info_str = ", ".join(info_parts) if info_parts else ""
        print(f"{test_name:30} {status} - {info_str}")
    
    print("-" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")


def main():
    """Main function to run all FAISS tests."""
    print("FAISS Independent Testing Script")
    print("=" * 50)
    
    # Create test data
    weight_matrix, calibration_data = create_test_data(matrix_size=100, calibration_size=1000)
    
    # Run all tests
    all_results = {}
    
    # Basic functionality tests
    basic_results = test_faiss_basic_functionality(weight_matrix, calibration_data)
    all_results.update(basic_results)
    
    # IVFPQ tests
    ivfpq_results = test_faiss_ivfpq(weight_matrix, calibration_data)
    all_results.update(ivfpq_results)
    
    # Range search tests
    range_results = test_faiss_range_search(weight_matrix, calibration_data)
    all_results.update(range_results)
    
    # Reconstruction tests
    recon_results = test_faiss_reconstruction(weight_matrix)
    all_results.update(recon_results)
    
    # Scalar quantization tests
    sq_results = test_faiss_scalar_quantization(weight_matrix, calibration_data)
    all_results.update(sq_results)
    
    # GPU support tests
    gpu_results = test_faiss_gpu_support()
    all_results.update(gpu_results)
    
    # Print summary
    print_summary(all_results)
    
    print("\nFAISS testing completed!")


if __name__ == "__main__":
    main() 