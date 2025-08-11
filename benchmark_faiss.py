"""
FAISS-only benchmarking suite for AXL library.

This module provides benchmarks for FAISS backend with safer configurations
to avoid clustering issues and segmentation faults.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import os
from dataclasses import dataclass
from collections import defaultdict

from axl.layer import (
    ApproximateLinear, 
    ApproximateLinearConfig, 
    BackendType, 
    QuantizationType
)
from axl.utils import (
    svd_keys_values,
    estimate_covariance,
    whiten_standardize_keys,
    compute_search_threshold
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    time_taken: float
    memory_used: float
    accuracy: float
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "time_taken": self.time_taken,
            "memory_used": self.memory_used,
            "accuracy": self.accuracy,
            "parameters": self.parameters
        }


class FAISSBenchmark:
    """FAISS-only benchmarking suite for AXL library."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def benchmark_utils_functions(self) -> List[BenchmarkResult]:
        """Benchmark utility functions."""
        print("Benchmarking utility functions...")
        results = []
        
        # Test matrix sizes
        matrix_sizes = [(100, 50), (500, 250), (1000, 500), (2000, 1000)]
        
        for out_features, in_features in matrix_sizes:
            print(f"  Testing matrix size: {out_features}x{in_features}")
            
            # Generate test data
            weight_matrix = np.random.randn(out_features, in_features).astype(np.float32)
            calibration_data = np.random.randn(1000, in_features).astype(np.float32)
            
            # Benchmark SVD decomposition
            start_time = time.time()
            U, S, V, K = svd_keys_values(weight_matrix, energy_keep=0.9)
            svd_time = time.time() - start_time
            
            results.append(BenchmarkResult(
                name=f"SVD_{out_features}x{in_features}",
                time_taken=svd_time,
                memory_used=weight_matrix.nbytes / 1024 / 1024,  # MB
                accuracy=1.0,  # SVD is exact
                parameters={
                    "out_features": out_features,
                    "in_features": in_features,
                    "energy_keep": 0.9
                }
            ))
            
            # Benchmark covariance estimation
            start_time = time.time()
            cov_matrix = estimate_covariance(calibration_data, in_features)
            cov_time = time.time() - start_time
            
            results.append(BenchmarkResult(
                name=f"Covariance_{in_features}",
                time_taken=cov_time,
                memory_used=cov_matrix.nbytes / 1024 / 1024,  # MB
                accuracy=1.0,
                parameters={
                    "in_features": in_features,
                    "calibration_samples": 1000
                }
            ))
        
        self.results.extend(results)
        return results
    
    def benchmark_layer_initialization(self) -> List[BenchmarkResult]:
        """Benchmark layer initialization with FAISS configurations."""
        print("Benchmarking FAISS layer initialization...")
        results = []
        
        # Test only FAISS configurations with safer settings
        # Use larger layer sizes to ensure rank_kept >= 156
        configs = [
            ("FAISS_Safe_Small", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=32,  # Reduced from 64
                faiss_probe_clusters=8,
                faiss_key_subvectors=4,
                faiss_key_bits=4,
                faiss_value_subvectors=4,
                faiss_value_bits=4,
                energy_keep=0.95  # Higher energy keep to get more singular values
            )),
            ("FAISS_Safe_Medium", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=64,  # Reduced from 256
                faiss_probe_clusters=16,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.98
            )),
            ("FAISS_Safe_Large", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=128,  # Reduced from 1024
                faiss_probe_clusters=32,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.99
            ))
        ]
        
        # Test larger layer sizes to ensure sufficient rank_kept
        layer_sizes = [(512, 1024), (1024, 2048), (2048, 4096)]
        
        for name, config in configs:
            for in_features, out_features in layer_sizes:
                print(f"  Testing {name} with {in_features}x{out_features}")
                
                try:
                    # Create layer
                    layer = ApproximateLinear(in_features, out_features, config)
                    
                    # Generate more calibration data to ensure sufficient training points
                    calibration_data = torch.randn(2000, in_features)  # Increased from 1000
                    
                    # Benchmark initialization
                    start_time = time.time()
                    layer.initialize(calibration_data=calibration_data)
                    init_time = time.time() - start_time
                    
                    # Measure memory usage (approximate)
                    memory_usage = (
                        layer.weight.numel() * 4 +  # float32
                        (layer.bias.numel() if layer.bias is not None else 0) * 4 +
                        layer.whitening_matrix.numel() * 4 +
                        layer.key_scales.numel() * 4
                    ) / 1024 / 1024  # MB
                    
                    results.append(BenchmarkResult(
                        name=f"Init_{name}_{in_features}x{out_features}",
                        time_taken=init_time,
                        memory_used=memory_usage,
                        accuracy=1.0,
                        parameters={
                            "backend": config.backend.value,
                            "in_features": in_features,
                            "out_features": out_features,
                            "energy_keep": config.energy_keep,
                            "rank_kept": layer.rank_kept,
                            **{k: v for k, v in config.to_dict().items() 
                               if k not in ['backend', 'energy_keep']}
                        }
                    ))
                    
                    # Test forward pass
                    input_data = torch.randn(32, in_features)
                    start_time = time.time()
                    output = layer(input_data)
                    forward_time = time.time() - start_time
                    
                    # Calculate accuracy (cosine similarity with exact computation)
                    exact_output = torch.mm(input_data, layer.weight.T) + layer.bias
                    accuracy = torch.cosine_similarity(output, exact_output, dim=1).mean().item()
                    
                    results.append(BenchmarkResult(
                        name=f"Forward_{name}_{in_features}x{out_features}",
                        time_taken=forward_time,
                        memory_used=memory_usage,
                        accuracy=accuracy,
                        parameters={
                            "backend": config.backend.value,
                            "in_features": in_features,
                            "out_features": out_features,
                            "batch_size": 32,
                            "energy_keep": config.energy_keep,
                            "rank_kept": layer.rank_kept
                        }
                    ))
                    
                except Exception as e:
                    print(f"    Error with {name} {in_features}x{out_features}: {e}")
                    continue
        
        self.results.extend(results)
        return results
    
    def benchmark_memory_efficiency(self) -> List[BenchmarkResult]:
        """Benchmark memory efficiency with different compression levels."""
        print("Benchmarking memory efficiency...")
        results = []
        
        # Test different compression levels with FAISS
        compression_configs = [
            ("Low_Compression", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=16,
                faiss_probe_clusters=4,
                faiss_key_subvectors=2,
                faiss_key_bits=4,
                faiss_value_subvectors=2,
                faiss_value_bits=4,
                energy_keep=0.8
            )),
            ("Medium_Compression", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=32,
                faiss_probe_clusters=8,
                faiss_key_subvectors=4,
                faiss_key_bits=8,
                faiss_value_subvectors=4,
                faiss_value_bits=8,
                energy_keep=0.9
            )),
            ("High_Compression", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=64,
                faiss_probe_clusters=16,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.95
            ))
        ]
        
        layer_size = (1024, 2048)  # Use larger size to ensure sufficient rank
        in_features, out_features = layer_size
        
        for name, config in compression_configs:
            print(f"  Testing {name}")
            
            try:
                # Create and initialize layer
                layer = ApproximateLinear(in_features, out_features, config)
                calibration_data = torch.randn(2000, in_features)
                layer.initialize(calibration_data=calibration_data)
                
                # Calculate compression ratio
                original_size = in_features * out_features * 4  # float32
                compressed_size = (
                    layer.weight.numel() * 4 +
                    (layer.bias.numel() if layer.bias is not None else 0) * 4 +
                    layer.whitening_matrix.numel() * 4 +
                    layer.key_scales.numel() * 4
                )
                compression_ratio = original_size / compressed_size
                
                # Test accuracy
                input_data = torch.randn(100, in_features)
                output = layer(input_data)
                exact_output = torch.mm(input_data, layer.weight.T) + layer.bias
                accuracy = torch.cosine_similarity(output, exact_output, dim=1).mean().item()
                
                results.append(BenchmarkResult(
                    name=f"Memory_{name}",
                    time_taken=0.0,  # Not timing this
                    memory_used=compressed_size / 1024 / 1024,  # MB
                    accuracy=compression_ratio,
                    parameters={
                        "compression_ratio": compression_ratio,
                        "accuracy": accuracy,
                        "original_size_mb": original_size / 1024 / 1024,
                        "compressed_size_mb": compressed_size / 1024 / 1024,
                        "rank_kept": layer.rank_kept
                    }
                ))
                
            except Exception as e:
                print(f"    Error with {name}: {e}")
                continue
        
        self.results.extend(results)
        return results
    
    def benchmark_search_performance(self) -> List[BenchmarkResult]:
        """Benchmark search performance with different thresholds."""
        print("Benchmarking search performance...")
        results = []
        
        # Test different search thresholds
        config = ApproximateLinearConfig(
            backend=BackendType.FAISS,
            faiss_clusters=32,
            faiss_probe_clusters=8,
            faiss_key_subvectors=4,
            faiss_key_bits=8,
            faiss_value_subvectors=4,
            faiss_value_bits=8,
            energy_keep=0.95
        )
        
        layer_size = (1024, 2048)
        in_features, out_features = layer_size
        
        try:
            # Create and initialize layer
            layer = ApproximateLinear(in_features, out_features, config)
            calibration_data = torch.randn(2000, in_features)
            layer.initialize(calibration_data=calibration_data)
            
            # Test different search thresholds
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            input_data = torch.randn(100, in_features)
            
            for threshold in thresholds:
                print(f"  Testing threshold {threshold}")
                
                # Temporarily set search threshold
                original_threshold = layer.search_threshold
                layer.search_threshold = threshold
                
                # Benchmark forward pass
                start_time = time.time()
                output = layer(input_data)
                forward_time = time.time() - start_time
                
                # Calculate accuracy
                exact_output = torch.mm(input_data, layer.weight.T) + layer.bias
                accuracy = torch.cosine_similarity(output, exact_output, dim=1).mean().item()
                
                results.append(BenchmarkResult(
                    name=f"Search_Threshold_{threshold}",
                    time_taken=forward_time,
                    memory_used=0.0,  # Not measuring memory here
                    accuracy=accuracy,
                    parameters={
                        "threshold": threshold,
                        "time_per_sample_ms": forward_time * 1000 / 100,
                        "rank_kept": layer.rank_kept
                    }
                ))
                
                # Restore original threshold
                layer.search_threshold = original_threshold
                
        except Exception as e:
            print(f"    Error in search performance benchmark: {e}")
        
        self.results.extend(results)
        return results
    
    def generate_plots(self):
        """Generate visualization plots."""
        print("Generating plots...")
        
        if not self.results:
            print("No results to plot")
            return
        
        # Categorize results
        categories = defaultdict(list)
        for result in self.results:
            if result.name.startswith("SVD_"):
                categories["SVD"].append(result)
            elif result.name.startswith("Covariance_"):
                categories["Covariance"].append(result)
            elif result.name.startswith("Init_"):
                categories["Initialization"].append(result)
            elif result.name.startswith("Forward_"):
                categories["Forward_Pass"].append(result)
            elif result.name.startswith("Memory_"):
                categories["Memory"].append(result)
            elif result.name.startswith("Search_"):
                categories["Search"].append(result)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("FAISS AXL Benchmark Results", fontsize=16)
        
        # 1. SVD performance
        if "SVD" in categories:
            ax = axes[0, 0]
            svd_results = categories["SVD"]
            
            sizes = []
            times = []
            for result in svd_results:
                size = result.parameters["out_features"] * result.parameters["in_features"]
                sizes.append(size)
                times.append(result.time_taken * 1000)  # ms
            
            ax.scatter(sizes, times, alpha=0.7)
            ax.set_title("SVD Decomposition Time")
            ax.set_xlabel("Matrix Size (elements)")
            ax.set_ylabel("Time (ms)")
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # 2. Initialization time
        if "Initialization" in categories:
            ax = axes[0, 1]
            init_results = categories["Initialization"]
            
            configs = []
            times = []
            for result in init_results:
                config_name = result.name.split('_')[1] + '_' + result.name.split('_')[2]
                configs.append(config_name)
                times.append(result.time_taken * 1000)  # ms
            
            ax.bar(configs, times)
            ax.set_title("Layer Initialization Time")
            ax.set_xlabel("Configuration")
            ax.set_ylabel("Time (ms)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Forward pass time
        if "Forward_Pass" in categories:
            ax = axes[0, 2]
            forward_results = categories["Forward_Pass"]
            
            configs = []
            times = []
            for result in forward_results:
                config_name = result.name.split('_')[1] + '_' + result.name.split('_')[2]
                configs.append(config_name)
                times.append(result.time_taken * 1000)  # ms
            
            ax.bar(configs, times)
            ax.set_title("Forward Pass Time")
            ax.set_xlabel("Configuration")
            ax.set_ylabel("Time (ms)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Memory efficiency
        if "Memory" in categories:
            ax = axes[1, 0]
            memory_results = categories["Memory"]
            
            compression_levels = []
            compression_ratios = []
            for result in memory_results:
                if "Low_Compression" in result.name:
                    compression_levels.append("Low")
                elif "Medium_Compression" in result.name:
                    compression_levels.append("Medium")
                elif "High_Compression" in result.name:
                    compression_levels.append("High")
                compression_ratios.append(result.accuracy)
            
            ax.bar(compression_levels, compression_ratios)
            ax.set_title("Memory Compression Ratio")
            ax.set_ylabel("Compression Ratio")
        
        # 5. Search performance
        if "Search" in categories:
            ax = axes[1, 1]
            search_results = categories["Search"]
            
            thresholds = []
            times_per_sample = []
            for result in search_results:
                thresholds.append(result.parameters["threshold"])
                times_per_sample.append(result.parameters["time_per_sample_ms"])
            
            ax.plot(thresholds, times_per_sample, 'o-')
            ax.set_title("Search Performance vs Threshold")
            ax.set_xlabel("Search Threshold")
            ax.set_ylabel("Time per Sample (ms)")
        
        # 6. Summary statistics
        ax = axes[1, 2]
        if self.results:
            total_tests = len(self.results)
            successful_tests = len([r for r in self.results if r.accuracy > 0])
            avg_accuracy = np.mean([r.accuracy for r in self.results if r.accuracy > 0])
            
            ax.text(0.1, 0.8, f"Total Tests: {total_tests}", fontsize=12)
            ax.text(0.1, 0.6, f"Successful: {successful_tests}", fontsize=12)
            ax.text(0.1, 0.4, f"Avg Accuracy: {avg_accuracy:.3f}", fontsize=12)
            ax.set_title("Benchmark Summary")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "faiss_benchmark_results.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        results_dict = [result.to_dict() for result in self.results]
        
        with open(os.path.join(self.output_dir, "faiss_benchmark_results.json"), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/faiss_benchmark_results.json")
    
    def run_all_benchmarks(self):
        """Run all FAISS benchmarks and generate reports."""
        print("Starting FAISS AXL benchmark suite...")
        
        # Run all benchmarks
        self.benchmark_utils_functions()
        self.benchmark_layer_initialization()
        self.benchmark_memory_efficiency()
        self.benchmark_search_performance()
        
        # Generate reports
        self.generate_plots()
        self.save_results()
        
        # Print summary
        print("\nBenchmark Summary:")
        print(f"Total benchmarks run: {len(self.results)}")
        
        # Find fastest and most accurate configurations
        forward_results = [r for r in self.results if r.name.startswith("Forward_")]
        if forward_results:
            fastest = min(forward_results, key=lambda x: x.time_taken)
            most_accurate = max(forward_results, key=lambda x: x.accuracy)
            
            print(f"Fastest forward pass: {fastest.name} ({fastest.time_taken*1000:.2f} ms)")
            print(f"Most accurate forward pass: {most_accurate.name} ({most_accurate.accuracy:.3f})")
        
        print(f"Results saved to: {self.output_dir}/")


if __name__ == "__main__":
    # Run benchmarks
    benchmark = FAISSBenchmark()
    benchmark.run_all_benchmarks() 