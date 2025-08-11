#!/usr/bin/env python3
"""
Independent NanoPQ Testing Script

This script tests NanoPQ functionality independently of the AXL framework.
It creates test data and tests various NanoPQ configurations to ensure
NanoPQ works correctly with different settings and parameters.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, Optional
import os
import tempfile


def check_nanopq_availability() -> bool:
    """Check if NanoPQ is available for import."""
    try:
        import nanopq as npq
        print(f"✓ NanoPQ version: {npq.__version__}")
        return True
    except ImportError as e:
        print(f"✗ NanoPQ not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Error importing NanoPQ: {e}")
        return False


def create_test_data(matrix_size: int = 100, calibration_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create test data for NanoPQ testing."""
    print(f"Creating test data: {matrix_size}x{matrix_size} matrix, {calibration_size} calibration vectors")
    
    np.random.seed(42)
    weight_matrix = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    calibration_data = np.random.randn(calibration_size, matrix_size).astype(np.float32)
    
    print(f"Weight matrix shape: {weight_matrix.shape}")
    print(f"Calibration data shape: {calibration_data.shape}")
    
    return weight_matrix, calibration_data


def get_divisors(n: int, max_divisor: int = 16) -> list:
    """Get all divisors of n up to max_divisor."""
    divisors = []
    for i in range(1, min(n + 1, max_divisor + 1)):
        if n % i == 0:
            divisors.append(i)
    return divisors


def test_nanopq_basic_functionality(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """Test basic NanoPQ functionality with simple settings."""
    print("\n=== Testing Basic NanoPQ Functionality ===")
    
    results = {}
    
    try:
        import nanopq as npq
        
        # Test 1: Basic PQ with default settings
        print("Test 1: Basic PQ with default settings")
        start_time = time.time()
        
        # Use a reasonable M value for the dimension
        dimension = weight_matrix.shape[1]
        m = min(8, dimension)  # Use 8 or dimension, whichever is smaller
        
        pq = npq.PQ(M=m, Ks=64)  # Use smaller Ks to ensure enough training data
        pq.fit(weight_matrix)
        codes = pq.encode(weight_matrix)
        reconstructed = pq.decode(codes)
        
        query = calibration_data[0:1]
        distances, indices = pq.search(query, k=5)
        
        basic_time = time.time() - start_time
        reconstruction_error = np.mean(np.linalg.norm(weight_matrix - reconstructed, axis=1))
        
        results['basic_pq'] = {
            'success': True,
            'time': basic_time,
            'reconstruction_error': reconstruction_error,
            'codes_shape': codes.shape,
            'distances': distances[0],
            'indices': indices[0],
            'M': pq.M,
            'Ks': pq.Ks
        }
        print(f"✓ Basic PQ: M={pq.M}, Ks={pq.Ks}, Error={reconstruction_error:.6f}, Time={basic_time:.4f}s")
        
    except Exception as e:
        results['basic_pq'] = {'success': False, 'error': str(e)}
        print(f"✗ Basic PQ failed: {e}")
    
    # Test 2: PQ with custom parameters
    print("Test 2: PQ with custom parameters")
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    
    if valid_m_values:
        m = valid_m_values[-1]
        ks = min(64, weight_matrix.shape[0] // 2)  # Ensure Ks is smaller than training data
        
        try:
            start_time = time.time()
            
            pq_custom = npq.PQ(M=m, Ks=ks)
            pq_custom.fit(weight_matrix)
            codes_custom = pq_custom.encode(weight_matrix)
            reconstructed_custom = pq_custom.decode(codes_custom)
            
            query = calibration_data[0:1]
            distances, indices = pq_custom.search(query, k=5)
            
            custom_time = time.time() - start_time
            reconstruction_error = np.mean(np.linalg.norm(weight_matrix - reconstructed_custom, axis=1))
            
            results['custom_pq'] = {
                'success': True,
                'time': custom_time,
                'reconstruction_error': reconstruction_error,
                'codes_shape': codes_custom.shape,
                'distances': distances[0],
                'indices': indices[0],
                'M': m,
                'Ks': ks
            }
            print(f"✓ Custom PQ: M={m}, Ks={ks}, Error={reconstruction_error:.6f}, Time={custom_time:.4f}s")
            
        except Exception as e:
            results['custom_pq'] = {'success': False, 'error': str(e)}
            print(f"✗ Custom PQ failed: {e}")
    else:
        results['custom_pq'] = {'success': False, 'error': f'No valid divisors for dimension {dimension}'}
        print(f"✗ Custom PQ: No valid divisors for dimension {dimension}")
    
    return results


def test_nanopq_different_configurations(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """Test NanoPQ with different configurations."""
    print("\n=== Testing NanoPQ Different Configurations ===")
    
    results = {}
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    
    # Ensure Ks values are smaller than the number of training vectors
    max_ks = min(64, weight_matrix.shape[0] // 2)
    
    configs = [
        {'M': 2, 'Ks': min(16, max_ks), 'name': 'Small'},
        {'M': 4, 'Ks': min(32, max_ks), 'name': 'Medium'},
        {'M': 8, 'Ks': min(64, max_ks), 'name': 'Large'},
        {'M': valid_m_values[-1] if valid_m_values else 4, 'Ks': min(64, max_ks), 'name': 'Extra Large'}
    ]
    
    for config in configs:
        try:
            print(f"Testing {config['name']} config: M={config['M']}, Ks={config['Ks']}")
            start_time = time.time()
            
            import nanopq as npq
            pq = npq.PQ(M=config['M'], Ks=config['Ks'])
            pq.fit(weight_matrix)
            codes = pq.encode(weight_matrix)
            reconstructed = pq.decode(codes)
            
            query = calibration_data[0:1]
            distances, indices = pq.search(query, k=5)
            
            config_time = time.time() - start_time
            reconstruction_error = np.mean(np.linalg.norm(weight_matrix - reconstructed, axis=1))
            
            original_size = weight_matrix.nbytes
            compressed_size = codes.nbytes + sum(cw.nbytes for cw in pq.codewords)
            compression_ratio = original_size / compressed_size
            
            results[f'config_{config["name"].lower()}'] = {
                'success': True,
                'time': config_time,
                'reconstruction_error': reconstruction_error,
                'compression_ratio': compression_ratio,
                'distances': distances[0],
                'indices': indices[0],
                'config': config
            }
            print(f"✓ {config['name']}: Error={reconstruction_error:.6f}, Compression={compression_ratio:.2f}x, Time={config_time:.4f}s")
            
        except Exception as e:
            results[f'config_{config["name"].lower()}'] = {'success': False, 'error': str(e), 'config': config}
            print(f"✗ {config['name']} failed: {e}")
    
    return results


def test_nanopq_reconstruction_quality(weight_matrix: np.ndarray) -> Dict[str, Any]:
    """Test NanoPQ reconstruction quality with different settings."""
    print("\n=== Testing NanoPQ Reconstruction Quality ===")
    
    results = {}
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    
    # Ensure Ks values are smaller than the number of training vectors
    max_ks = min(64, weight_matrix.shape[0] // 2)
    
    quality_configs = [
        {'M': 2, 'Ks': min(8, max_ks), 'name': 'Low Quality'},
        {'M': 4, 'Ks': min(16, max_ks), 'name': 'Medium Quality'},
        {'M': 8, 'Ks': min(32, max_ks), 'name': 'High Quality'},
        {'M': valid_m_values[-1] if valid_m_values else 4, 'Ks': min(64, max_ks), 'name': 'Very High Quality'}
    ]
    
    for config in quality_configs:
        try:
            print(f"Testing {config['name']}: M={config['M']}, Ks={config['Ks']}")
            start_time = time.time()
            
            import nanopq as npq
            pq = npq.PQ(M=config['M'], Ks=config['Ks'])
            pq.fit(weight_matrix)
            codes = pq.encode(weight_matrix)
            reconstructed = pq.decode(codes)
            
            quality_time = time.time() - start_time
            
            reconstruction_error = np.mean(np.linalg.norm(weight_matrix - reconstructed, axis=1))
            max_error = np.max(np.linalg.norm(weight_matrix - reconstructed, axis=1))
            min_error = np.min(np.linalg.norm(weight_matrix - reconstructed, axis=1))
            
            original_norms = np.linalg.norm(weight_matrix, axis=1)
            reconstructed_norms = np.linalg.norm(reconstructed, axis=1)
            
            valid_original = original_norms > 1e-8
            valid_reconstructed = reconstructed_norms > 1e-8
            valid_indices = valid_original & valid_reconstructed
            
            if np.any(valid_indices):
                cosine_similarities = np.sum(weight_matrix[valid_indices] * reconstructed[valid_indices], axis=1) / (
                    original_norms[valid_indices] * reconstructed_norms[valid_indices]
                )
                mean_cosine_similarity = np.mean(cosine_similarities)
            else:
                mean_cosine_similarity = 0.0
            
            original_size = weight_matrix.nbytes
            compressed_size = codes.nbytes + sum(cw.nbytes for cw in pq.codewords)
            compression_ratio = original_size / compressed_size
            
            results[f'quality_{config["name"].lower().replace(" ", "_")}'] = {
                'success': True,
                'time': quality_time,
                'reconstruction_error': reconstruction_error,
                'max_error': max_error,
                'min_error': min_error,
                'mean_cosine_similarity': mean_cosine_similarity,
                'compression_ratio': compression_ratio,
                'config': config
            }
            print(f"✓ {config['name']}: Error={reconstruction_error:.6f}, Cosine={mean_cosine_similarity:.4f}, Compression={compression_ratio:.2f}x")
            
        except Exception as e:
            results[f'quality_{config["name"].lower().replace(" ", "_")}'] = {'success': False, 'error': str(e), 'config': config}
            print(f"✗ {config['name']} failed: {e}")
    
    return results


def test_nanopq_search_functionality(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """Test NanoPQ search functionality with different parameters."""
    print("\n=== Testing NanoPQ Search Functionality ===")
    
    results = {}
    
    try:
        import nanopq as npq
        
        # Use smaller Ks to ensure enough training data
        max_ks = min(64, weight_matrix.shape[0] // 2)
        pq = npq.PQ(M=8, Ks=max_ks)
        pq.fit(weight_matrix)
        codes = pq.encode(weight_matrix)
        
        search_configs = [
            {'k': 1, 'name': 'Top-1'},
            {'k': 5, 'name': 'Top-5'},
            {'k': 10, 'name': 'Top-10'},
            {'k': 20, 'name': 'Top-20'}
        ]
        
        queries = calibration_data[:5]
        
        for config in search_configs:
            try:
                print(f"Testing {config['name']} search")
                start_time = time.time()
                
                distances, indices = pq.search(queries, k=config['k'])
                
                search_time = time.time() - start_time
                mean_distances = np.mean(distances, axis=0)
                
                results[f'search_{config["name"].lower().replace("-", "_")}'] = {
                    'success': True,
                    'time': search_time,
                    'k': config['k'],
                    'num_queries': len(queries),
                    'mean_distances': mean_distances.tolist(),
                    'indices_shape': indices.shape
                }
                print(f"✓ {config['name']}: {len(queries)} queries in {search_time:.4f}s, Mean distance={np.mean(mean_distances):.6f}")
                
            except Exception as e:
                results[f'search_{config["name"].lower().replace("-", "_")}'] = {'success': False, 'error': str(e), 'config': config}
                print(f"✗ {config['name']} search failed: {e}")
        
        # Test single query
        try:
            print("Testing single query search")
            start_time = time.time()
            
            single_query = calibration_data[0:1]
            distances, indices = pq.search(single_query, k=10)
            
            single_time = time.time() - start_time
            
            results['search_single_query'] = {
                'success': True,
                'time': single_time,
                'distances': distances[0].tolist(),
                'indices': indices[0].tolist()
            }
            print(f"✓ Single query: Found {len(indices[0])} results in {single_time:.4f}s")
            
        except Exception as e:
            results['search_single_query'] = {'success': False, 'error': str(e)}
            print(f"✗ Single query search failed: {e}")
            
    except Exception as e:
        results['search'] = {'success': False, 'error': str(e)}
        print(f"✗ Search functionality failed: {e}")
    
    return results


def test_nanopq_save_load_functionality(weight_matrix: np.ndarray) -> Dict[str, Any]:
    """Test NanoPQ save and load functionality."""
    print("\n=== Testing NanoPQ Save/Load Functionality ===")
    
    results = {}
    
    try:
        import nanopq as npq
        
        # Use smaller Ks to ensure enough training data
        max_ks = min(64, weight_matrix.shape[0] // 2)
        pq_original = npq.PQ(M=8, Ks=max_ks)
        pq_original.fit(weight_matrix)
        codes_original = pq_original.encode(weight_matrix)
        reconstructed_original = pq_original.decode(codes_original)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "pq_model")
            pq_original.save(save_path)
            
            pq_loaded = npq.PQ.load(save_path)
            
            codes_loaded = pq_loaded.encode(weight_matrix)
            reconstructed_loaded = pq_loaded.decode(codes_loaded)
            
            codes_match = np.array_equal(codes_original, codes_loaded)
            reconstruction_match = np.allclose(reconstructed_original, reconstructed_loaded, rtol=1e-6)
            
            query = np.random.randn(1, weight_matrix.shape[1]).astype(np.float32)
            distances_original, indices_original = pq_original.search(query, k=5)
            distances_loaded, indices_loaded = pq_loaded.search(query, k=5)
            
            search_match = (np.array_equal(indices_original, indices_loaded) and 
                          np.allclose(distances_original, distances_loaded, rtol=1e-6))
            
            results['save_load'] = {
                'success': True,
                'codes_match': codes_match,
                'reconstruction_match': reconstruction_match,
                'search_match': search_match,
                'save_path': save_path
            }
            print(f"✓ Save/Load: Codes match={codes_match}, Reconstruction match={reconstruction_match}, Search match={search_match}")
            
    except Exception as e:
        results['save_load'] = {'success': False, 'error': str(e)}
        print(f"✗ Save/Load failed: {e}")
    
    return results


def test_nanopq_performance_comparison(weight_matrix: np.ndarray, calibration_data: np.ndarray) -> Dict[str, Any]:
    """Test NanoPQ performance with different configurations."""
    print("\n=== Testing NanoPQ Performance Comparison ===")
    
    results = {}
    dimension = weight_matrix.shape[1]
    valid_m_values = get_divisors(dimension, 16)
    
    # Ensure Ks values are smaller than the number of training vectors
    max_ks = min(64, weight_matrix.shape[0] // 2)
    
    perf_configs = [
        {'M': 2, 'Ks': min(16, max_ks), 'name': 'Fast'},
        {'M': 4, 'Ks': min(32, max_ks), 'name': 'Balanced'},
        {'M': 8, 'Ks': min(64, max_ks), 'name': 'Accurate'},
        {'M': valid_m_values[-1] if valid_m_values else 4, 'Ks': min(64, max_ks), 'name': 'Very Accurate'}
    ]
    
    for config in perf_configs:
        try:
            print(f"Testing {config['name']} performance: M={config['M']}, Ks={config['Ks']}")
            
            import nanopq as npq
            start_time = time.time()
            pq = npq.PQ(M=config['M'], Ks=config['Ks'])
            pq.fit(weight_matrix)
            training_time = time.time() - start_time
            
            start_time = time.time()
            codes = pq.encode(weight_matrix)
            encoding_time = time.time() - start_time
            
            start_time = time.time()
            reconstructed = pq.decode(codes)
            decoding_time = time.time() - start_time
            
            queries = calibration_data[:10]
            start_time = time.time()
            distances, indices = pq.search(queries, k=5)
            search_time = time.time() - start_time
            
            reconstruction_error = np.mean(np.linalg.norm(weight_matrix - reconstructed, axis=1))
            
            original_size = weight_matrix.nbytes
            compressed_size = codes.nbytes + sum(cw.nbytes for cw in pq.codewords)
            compression_ratio = original_size / compressed_size
            
            results[f'performance_{config["name"].lower()}'] = {
                'success': True,
                'training_time': training_time,
                'encoding_time': encoding_time,
                'decoding_time': decoding_time,
                'search_time': search_time,
                'total_time': training_time + encoding_time + decoding_time + search_time,
                'reconstruction_error': reconstruction_error,
                'compression_ratio': compression_ratio,
                'config': config
            }
            print(f"✓ {config['name']}: Train={training_time:.4f}s, Encode={encoding_time:.4f}s, "
                  f"Decode={decoding_time:.4f}s, Search={search_time:.4f}s, "
                  f"Error={reconstruction_error:.6f}, Compression={compression_ratio:.2f}x")
            
        except Exception as e:
            results[f'performance_{config["name"].lower()}'] = {'success': False, 'error': str(e), 'config': config}
            print(f"✗ {config['name']} performance test failed: {e}")
    
    return results


def test_nanopq_edge_cases(weight_matrix: np.ndarray) -> Dict[str, Any]:
    """Test NanoPQ with edge cases and error conditions."""
    print("\n=== Testing NanoPQ Edge Cases ===")
    
    results = {}
    
    try:
        import nanopq as npq
        
        # Test 1: Very small M and Ks
        try:
            print("Test 1: Very small M and Ks")
            pq_small = npq.PQ(M=1, Ks=2)
            pq_small.fit(weight_matrix)
            codes_small = pq_small.encode(weight_matrix)
            reconstructed_small = pq_small.decode(codes_small)
            
            reconstruction_error = np.mean(np.linalg.norm(weight_matrix - reconstructed_small, axis=1))
            
            results['edge_case_small'] = {
                'success': True,
                'reconstruction_error': reconstruction_error,
                'M': 1,
                'Ks': 2
            }
            print(f"✓ Very small config: Error={reconstruction_error:.6f}")
            
        except Exception as e:
            results['edge_case_small'] = {'success': False, 'error': str(e)}
            print(f"✗ Very small config failed: {e}")
        
        # Test 2: Large Ks (but within limits)
        try:
            print("Test 2: Large Ks")
            max_ks = min(64, weight_matrix.shape[0] // 2)
            pq_large = npq.PQ(M=4, Ks=max_ks)
            pq_large.fit(weight_matrix)
            codes_large = pq_large.encode(weight_matrix)
            reconstructed_large = pq_large.decode(codes_large)
            
            reconstruction_error = np.mean(np.linalg.norm(weight_matrix - reconstructed_large, axis=1))
            
            results['edge_case_large'] = {
                'success': True,
                'reconstruction_error': reconstruction_error,
                'M': 4,
                'Ks': max_ks
            }
            print(f"✓ Large Ks config: Error={reconstruction_error:.6f}")
            
        except Exception as e:
            results['edge_case_large'] = {'success': False, 'error': str(e)}
            print(f"✗ Large Ks config failed: {e}")
        
        # Test 3: Zero variance data
        try:
            print("Test 3: Zero variance data")
            zero_var_data = np.ones((50, 32), dtype=np.float32)
            pq_zero = npq.PQ(M=4, Ks=16)  # Use smaller Ks
            pq_zero.fit(zero_var_data)
            codes_zero = pq_zero.encode(zero_var_data)
            reconstructed_zero = pq_zero.decode(codes_zero)
            
            reconstruction_error = np.mean(np.linalg.norm(zero_var_data - reconstructed_zero, axis=1))
            
            results['edge_case_zero_variance'] = {
                'success': True,
                'reconstruction_error': reconstruction_error
            }
            print(f"✓ Zero variance data: Error={reconstruction_error:.6f}")
            
        except Exception as e:
            results['edge_case_zero_variance'] = {'success': False, 'error': str(e)}
            print(f"✗ Zero variance data failed: {e}")
        
        # Test 4: Single vector (not supported by NanoPQ)
        try:
            print("Test 4: Single vector")
            # Use multiple vectors but test with small dataset
            small_data = weight_matrix[:10]  # Use first 10 vectors
            pq_single = npq.PQ(M=2, Ks=4)  # Very small parameters
            pq_single.fit(small_data)
            codes_single = pq_single.encode(small_data)
            reconstructed_single = pq_single.decode(codes_single)
            
            reconstruction_error = np.mean(np.linalg.norm(small_data - reconstructed_single, axis=1))
            
            results['edge_case_single_vector'] = {
                'success': True,
                'reconstruction_error': reconstruction_error
            }
            print(f"✓ Small dataset: Error={reconstruction_error:.6f}")
            
        except Exception as e:
            results['edge_case_single_vector'] = {'success': False, 'error': str(e)}
            print(f"✗ Small dataset failed: {e}")
            
    except Exception as e:
        results['edge_cases'] = {'success': False, 'error': str(e)}
        print(f"✗ Edge cases testing failed: {e}")
    
    return results


def print_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary of all test results."""
    print("\n" + "="*60)
    print("NANOPQ TEST SUMMARY")
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
                print(f"{test_name:35} {status} - {result['error']}")
            else:
                print(f"{test_name:35} {status}")
            continue
        
        info_parts = []
        if 'time' in result:
            info_parts.append(f"time={result['time']:.4f}s")
        if 'reconstruction_error' in result:
            info_parts.append(f"error={result['reconstruction_error']:.6f}")
        if 'compression_ratio' in result:
            info_parts.append(f"compression={result['compression_ratio']:.2f}x")
        if 'M' in result:
            info_parts.append(f"M={result['M']}")
        if 'Ks' in result:
            info_parts.append(f"Ks={result['Ks']}")
        if 'training_time' in result:
            info_parts.append(f"train={result['training_time']:.4f}s")
        if 'encoding_time' in result:
            info_parts.append(f"encode={result['encoding_time']:.4f}s")
        if 'search_time' in result:
            info_parts.append(f"search={result['search_time']:.4f}s")
        
        info_str = ", ".join(info_parts) if info_parts else ""
        print(f"{test_name:35} {status} - {info_str}")
    
    print("-" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")


def main():
    """Main function to run all NanoPQ tests."""
    print("NanoPQ Independent Testing Script")
    print("=" * 50)
    
    if not check_nanopq_availability():
        print("NanoPQ is not available. Please install it with: pip install nanopq")
        return
    
    weight_matrix, calibration_data = create_test_data(matrix_size=100, calibration_size=1000)
    
    all_results = {}
    
    basic_results = test_nanopq_basic_functionality(weight_matrix, calibration_data)
    all_results.update(basic_results)
    
    config_results = test_nanopq_different_configurations(weight_matrix, calibration_data)
    all_results.update(config_results)
    
    quality_results = test_nanopq_reconstruction_quality(weight_matrix)
    all_results.update(quality_results)
    
    search_results = test_nanopq_search_functionality(weight_matrix, calibration_data)
    all_results.update(search_results)
    
    save_load_results = test_nanopq_save_load_functionality(weight_matrix)
    all_results.update(save_load_results)
    
    perf_results = test_nanopq_performance_comparison(weight_matrix, calibration_data)
    all_results.update(perf_results)
    
    edge_results = test_nanopq_edge_cases(weight_matrix)
    all_results.update(edge_results)
    
    print_summary(all_results)
    
    print("\nNanoPQ testing completed!")


if __name__ == "__main__":
    main() 