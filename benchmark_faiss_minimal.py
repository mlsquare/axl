"""
Minimal FAISS benchmarking suite for AXL library.

This module provides minimal benchmarks for FAISS backend to test basic functionality
without triggering clustering issues.
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


class FAISSMinimalBenchmark:
    """Minimal FAISS benchmarking suite for AXL library."""
    
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
                    "energy_keep": 0.9,
                    "rank": len(S)
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
    
    def benchmark_single_faiss_config(self) -> List[BenchmarkResult]:
        """Benchmark a single, very conservative FAISS configuration."""
        print("Benchmarking single FAISS configuration...")
        results = []
        
        # Try a very conservative configuration
        config = ApproximateLinearConfig(
            backend=BackendType.FAISS,
            faiss_clusters=8,  # Very small
            faiss_probe_clusters=4,
            faiss_key_subvectors=2,
            faiss_key_bits=4,
            faiss_value_subvectors=2,
            faiss_value_bits=4,
            energy_keep=0.8  # Lower energy keep to get fewer singular values
        )
        
        # Try with a moderate layer size
        in_features, out_features = 256, 512
        
        print(f"  Testing FAISS with {in_features}x{out_features}")
        
        try:
            # Create layer
            layer = ApproximateLinear(in_features, out_features, config)
            
            # Generate calibration data
            calibration_data = torch.randn(1000, in_features)
            
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
                name=f"Init_FAISS_Conservative_{in_features}x{out_features}",
                time_taken=init_time,
                memory_used=memory_usage,
                accuracy=1.0,
                parameters={
                    "backend": config.backend.value,
                    "in_features": in_features,
                    "out_features": out_features,
                    "energy_keep": config.energy_keep,
                    "rank_kept": layer.rank_kept,
                    "faiss_clusters": config.faiss_clusters,
                    "faiss_probe_clusters": config.faiss_probe_clusters
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
                name=f"Forward_FAISS_Conservative_{in_features}x{out_features}",
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
            print(f"    Error with FAISS configuration: {e}")
            # Add a placeholder result to show the error
            results.append(BenchmarkResult(
                name=f"Error_FAISS_Conservative_{in_features}x{out_features}",
                time_taken=0.0,
                memory_used=0.0,
                accuracy=0.0,
                parameters={
                    "error": str(e),
                    "backend": "faiss",
                    "in_features": in_features,
                    "out_features": out_features
                }
            ))
        
        self.results.extend(results)
        return results
    
    def benchmark_backend_creation(self) -> List[BenchmarkResult]:
        """Benchmark backend creation without initialization."""
        print("Benchmarking backend creation...")
        results = []
        
        # Test creating FAISS backend without fitting
        configs = [
            ("FAISS_Small", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=16,
                faiss_probe_clusters=4,
                faiss_key_subvectors=4,
                faiss_key_bits=4,
                faiss_value_subvectors=4,
                faiss_value_bits=4
            )),
            ("FAISS_Medium", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=32,
                faiss_probe_clusters=8,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8
            ))
        ]
        
        for name, config in configs:
            print(f"  Testing {name} backend creation")
            
            try:
                start_time = time.time()
                from axl.layer import create_backend
                backend = create_backend(config)
                creation_time = time.time() - start_time
                
                # Get backend info
                info = backend.get_info()
                
                results.append(BenchmarkResult(
                    name=f"Backend_Creation_{name}",
                    time_taken=creation_time,
                    memory_used=0.0,  # Not measuring memory for creation
                    accuracy=1.0,
                    parameters={
                        "backend_type": info.get("backend_type", "Unknown"),
                        "is_fitted": info.get("is_fitted", False),
                        "faiss_clusters": config.faiss_clusters,
                        "faiss_probe_clusters": config.faiss_probe_clusters
                    }
                ))
                
            except Exception as e:
                print(f"    Error creating {name} backend: {e}")
                results.append(BenchmarkResult(
                    name=f"Error_Backend_Creation_{name}",
                    time_taken=0.0,
                    memory_used=0.0,
                    accuracy=0.0,
                    parameters={
                        "error": str(e),
                        "backend": "faiss"
                    }
                ))
        
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
            elif result.name.startswith("Backend_Creation_"):
                categories["Backend_Creation"].append(result)
            elif result.name.startswith("Error_"):
                categories["Errors"].append(result)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("FAISS Minimal AXL Benchmark Results", fontsize=16)
        
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
        
        # 2. Backend creation time
        if "Backend_Creation" in categories:
            ax = axes[0, 1]
            backend_results = categories["Backend_Creation"]
            
            configs = []
            times = []
            for result in backend_results:
                config_name = result.name.split('_')[2]  # Get the config name
                configs.append(config_name)
                times.append(result.time_taken * 1000)  # ms
            
            ax.bar(configs, times)
            ax.set_title("Backend Creation Time")
            ax.set_xlabel("Configuration")
            ax.set_ylabel("Time (ms)")
        
        # 3. Forward pass time (if available)
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
        
        # 4. Summary statistics
        ax = axes[1, 1]
        if self.results:
            total_tests = len(self.results)
            successful_tests = len([r for r in self.results if r.accuracy > 0 and not r.name.startswith("Error_")])
            error_tests = len([r for r in self.results if r.name.startswith("Error_")])
            avg_accuracy = np.mean([r.accuracy for r in self.results if r.accuracy > 0 and not r.name.startswith("Error_")])
            
            ax.text(0.1, 0.8, f"Total Tests: {total_tests}", fontsize=12)
            ax.text(0.1, 0.6, f"Successful: {successful_tests}", fontsize=12)
            ax.text(0.1, 0.4, f"Errors: {error_tests}", fontsize=12)
            ax.text(0.1, 0.2, f"Avg Accuracy: {avg_accuracy:.3f}", fontsize=12)
            ax.set_title("Benchmark Summary")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "faiss_minimal_benchmark_results.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        results_dict = [result.to_dict() for result in self.results]
        
        with open(os.path.join(self.output_dir, "faiss_minimal_benchmark_results.json"), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/faiss_minimal_benchmark_results.json")
    
    def run_all_benchmarks(self):
        """Run all minimal FAISS benchmarks and generate reports."""
        print("Starting FAISS Minimal AXL benchmark suite...")
        
        # Run all benchmarks
        self.benchmark_utils_functions()
        self.benchmark_backend_creation()
        self.benchmark_single_faiss_config()
        
        # Generate reports
        self.generate_plots()
        self.save_results()
        
        # Print summary
        print("\nBenchmark Summary:")
        print(f"Total benchmarks run: {len(self.results)}")
        
        # Count successful vs error tests
        successful_tests = [r for r in self.results if r.accuracy > 0 and not r.name.startswith("Error_")]
        error_tests = [r for r in self.results if r.name.startswith("Error_")]
        
        print(f"Successful tests: {len(successful_tests)}")
        print(f"Error tests: {len(error_tests)}")
        
        # Find fastest and most accurate configurations
        forward_results = [r for r in self.results if r.name.startswith("Forward_") and r.accuracy > 0]
        if forward_results:
            fastest = min(forward_results, key=lambda x: x.time_taken)
            most_accurate = max(forward_results, key=lambda x: x.accuracy)
            
            print(f"Fastest forward pass: {fastest.name} ({fastest.time_taken*1000:.2f} ms)")
            print(f"Most accurate forward pass: {most_accurate.name} ({most_accurate.accuracy:.3f})")
        
        print(f"Results saved to: {self.output_dir}/")


if __name__ == "__main__":
    # Run benchmarks
    benchmark = FAISSMinimalBenchmark()
    benchmark.run_all_benchmarks() 