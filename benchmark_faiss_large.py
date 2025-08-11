"""
FAISS Large-scale benchmarking suite for AXL library.

This module provides benchmarks for FAISS backend with very large configurations
to ensure sufficient data for clustering and avoid segmentation faults.
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


class FAISSLargeBenchmark:
    """FAISS Large-scale benchmarking suite for AXL library."""
    
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
        """Benchmark layer initialization with very large FAISS configurations."""
        print("Benchmarking FAISS layer initialization with large configurations...")
        results = []
        
        # Test only FAISS configurations with very large settings
        # Use much larger layer sizes to ensure rank_kept >= 156
        configs = [
            ("FAISS_Large_Config", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=64,
                faiss_probe_clusters=16,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.99  # Very high energy keep to get maximum singular values
            ))
        ]
        
        # Test very large layer sizes to ensure sufficient rank_kept
        layer_sizes = [(2048, 4096), (4096, 8192)]  # Much larger sizes
        
        for name, config in configs:
            for in_features, out_features in layer_sizes:
                print(f"  Testing {name} with {in_features}x{out_features}")
                
                try:
                    # Create layer
                    layer = ApproximateLinear(in_features, out_features, config)
                    
                    # Generate much more calibration data to ensure sufficient training points
                    calibration_data = torch.randn(5000, in_features)  # Much more data
                    
                    # Benchmark initialization
                    start_time = time.time()
                    layer.initialize(calibration_data=calibration_data)
                    init_time = time.time() - start_time
                    
                    print(f"    Rank kept: {layer.rank_kept}")
                    
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
                    input_data = torch.randn(16, in_features)  # Smaller batch for large layers
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
                            "batch_size": 16,
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
        """Benchmark memory efficiency with large configurations."""
        print("Benchmarking memory efficiency...")
        results = []
        
        # Test different compression levels with FAISS
        compression_configs = [
            ("Low_Compression", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=32,
                faiss_probe_clusters=8,
                faiss_key_subvectors=4,
                faiss_key_bits=8,
                faiss_value_subvectors=4,
                faiss_value_bits=8,
                energy_keep=0.9
            )),
            ("Medium_Compression", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=64,
                faiss_probe_clusters=16,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.95
            )),
            ("High_Compression", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=128,
                faiss_probe_clusters=32,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.98
            ))
        ]
        
        layer_size = (2048, 4096)  # Use large size to ensure sufficient rank
        in_features, out_features = layer_size
        
        for name, config in compression_configs:
            print(f"  Testing {name}")
            
            try:
                # Create and initialize layer
                layer = ApproximateLinear(in_features, out_features, config)
                calibration_data = torch.randn(5000, in_features)
                layer.initialize(calibration_data=calibration_data)
                
                print(f"    Rank kept: {layer.rank_kept}")
                
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
                input_data = torch.randn(50, in_features)
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
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("FAISS Large-Scale AXL Benchmark Results", fontsize=16)
        
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
            ax = axes[1, 0]
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
            ax = axes[1, 1]
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "faiss_large_benchmark_results.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        results_dict = [result.to_dict() for result in self.results]
        
        with open(os.path.join(self.output_dir, "faiss_large_benchmark_results.json"), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/faiss_large_benchmark_results.json")
    
    def run_all_benchmarks(self):
        """Run all FAISS large-scale benchmarks and generate reports."""
        print("Starting FAISS Large-Scale AXL benchmark suite...")
        
        # Run all benchmarks
        self.benchmark_utils_functions()
        self.benchmark_layer_initialization()
        self.benchmark_memory_efficiency()
        
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
    benchmark = FAISSLargeBenchmark()
    benchmark.run_all_benchmarks() 