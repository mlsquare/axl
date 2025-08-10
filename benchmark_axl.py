"""
Comprehensive benchmarking suite for AXL library.

This module provides benchmarks to measure performance improvements,
compare different configurations, and identify optimization opportunities.
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


class AXLBenchmark:
    """Comprehensive benchmarking suite for AXL library."""
    
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
                name=f"Covariance_{out_features}x{in_features}",
                time_taken=cov_time,
                memory_used=cov_matrix.nbytes / 1024 / 1024,
                accuracy=1.0,
                parameters={
                    "feature_dim": in_features,
                    "n_samples": 1000
                }
            ))
            
            # Benchmark key whitening
            start_time = time.time()
            L_inv, K_tilde, scales = whiten_standardize_keys(K, cov_matrix)
            whiten_time = time.time() - start_time
            
            results.append(BenchmarkResult(
                name=f"Whitening_{out_features}x{in_features}",
                time_taken=whiten_time,
                memory_used=K.nbytes / 1024 / 1024,
                accuracy=1.0,
                parameters={
                    "n_keys": K.shape[0],
                    "key_dim": K.shape[1]
                }
            ))
        
        self.results.extend(results)
        return results
    
    def benchmark_layer_initialization(self) -> List[BenchmarkResult]:
        """Benchmark layer initialization with different configurations."""
        print("Benchmarking layer initialization...")
        results = []
        
        # Test configurations
        configs = [
            ("FAISS_Small", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=64,
                faiss_probe_clusters=8,
                faiss_key_subvectors=4,
                faiss_key_bits=4,
                faiss_value_subvectors=4,
                faiss_value_bits=4,
                energy_keep=0.9
            )),
            ("FAISS_Medium", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=256,
                faiss_probe_clusters=32,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.95
            )),
            ("FAISS_Large", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=1024,
                faiss_probe_clusters=128,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.98
            )),
            ("NanoPQ_Small", ApproximateLinearConfig(
                backend=BackendType.NANOPQ,
                nanopq_key_subvectors=4,
                nanopq_key_codewords=64,
                nanopq_value_subvectors=4,
                nanopq_value_codewords=64,
                energy_keep=0.9
            )),
            ("NanoPQ_Medium", ApproximateLinearConfig(
                backend=BackendType.NANOPQ,
                nanopq_key_subvectors=8,
                nanopq_key_codewords=256,
                nanopq_value_subvectors=8,
                nanopq_value_codewords=256,
                energy_keep=0.95
            )),
            ("NanoPQ_Large", ApproximateLinearConfig(
                backend=BackendType.NANOPQ,
                nanopq_key_subvectors=8,
                nanopq_key_codewords=512,
                nanopq_value_subvectors=8,
                nanopq_value_codewords=512,
                energy_keep=0.98
            ))
        ]
        
        # Test layer sizes
        layer_sizes = [(128, 256), (512, 1024), (1024, 2048)]
        
        for name, config in configs:
            for in_features, out_features in layer_sizes:
                print(f"  Testing {name} with {in_features}x{out_features}")
                
                # Create layer
                layer = ApproximateLinear(in_features, out_features, config)
                
                # Generate calibration data
                calibration_data = torch.randn(1000, in_features)
                
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
                        **{k: v for k, v in config.to_dict().items() 
                           if k.startswith(('faiss_', 'nanopq_'))}
                    }
                ))
        
        self.results.extend(results)
        return results
    
    def benchmark_forward_pass(self) -> List[BenchmarkResult]:
        """Benchmark forward pass performance."""
        print("Benchmarking forward pass...")
        results = []
        
        # Test configurations
        configs = [
            ("FAISS_Fast", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=64,
                faiss_probe_clusters=8,
                faiss_key_subvectors=4,
                faiss_key_bits=4,
                faiss_value_subvectors=4,
                faiss_value_bits=4,
                energy_keep=0.8
            )),
            ("FAISS_Accurate", ApproximateLinearConfig(
                backend=BackendType.FAISS,
                faiss_clusters=512,
                faiss_probe_clusters=64,
                faiss_key_subvectors=8,
                faiss_key_bits=8,
                faiss_value_subvectors=8,
                faiss_value_bits=8,
                energy_keep=0.95
            )),
            ("NanoPQ_Fast", ApproximateLinearConfig(
                backend=BackendType.NANOPQ,
                nanopq_key_subvectors=4,
                nanopq_key_codewords=64,
                nanopq_value_subvectors=4,
                nanopq_value_codewords=64,
                energy_keep=0.8
            )),
            ("NanoPQ_Accurate", ApproximateLinearConfig(
                backend=BackendType.NANOPQ,
                nanopq_key_subvectors=8,
                nanopq_key_codewords=256,
                nanopq_value_subvectors=8,
                nanopq_value_codewords=256,
                energy_keep=0.95
            ))
        ]
        
        # Test input sizes
        input_sizes = [1, 10, 100, 1000]
        layer_size = (512, 1024)
        
        for name, config in configs:
            print(f"  Testing {name}")
            
            # Create and initialize layer
            layer = ApproximateLinear(layer_size[0], layer_size[1], config)
            calibration_data = torch.randn(1000, layer_size[0])
            layer.initialize(calibration_data=calibration_data)
            
            for batch_size in input_sizes:
                # Generate input
                input_tensor = torch.randn(batch_size, layer_size[0])
                
                # Warm up
                for _ in range(10):
                    _ = layer(input_tensor)
                
                # Benchmark
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                for _ in range(100):
                    output = layer(input_tensor)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                forward_time = time.time() - start_time
                
                # Compute accuracy (approximate vs exact)
                with torch.no_grad():
                    exact_output = input_tensor @ layer.weight.T
                    if layer.bias is not None:
                        exact_output += layer.bias
                    
                    # Compute relative error
                    relative_error = torch.norm(output - exact_output) / torch.norm(exact_output)
                    accuracy = 1.0 - relative_error.item()
                
                results.append(BenchmarkResult(
                    name=f"Forward_{name}_batch{batch_size}",
                    time_taken=forward_time / 100,  # Average time per forward pass
                    memory_used=input_tensor.numel() * 4 / 1024 / 1024,
                    accuracy=accuracy,
                    parameters={
                        "backend": config.backend.value,
                        "batch_size": batch_size,
                        "in_features": layer_size[0],
                        "out_features": layer_size[1],
                        "energy_keep": config.energy_keep
                    }
                ))
        
        self.results.extend(results)
        return results
    
    def benchmark_memory_efficiency(self) -> List[BenchmarkResult]:
        """Benchmark memory efficiency improvements."""
        print("Benchmarking memory efficiency...")
        results = []
        
        # Test different layer sizes
        layer_sizes = [(256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
        
        for in_features, out_features in layer_sizes:
            print(f"  Testing memory for {in_features}x{out_features}")
            
            # Dense layer memory
            dense_memory = in_features * out_features * 4 / 1024 / 1024  # MB
            
            # Approximate layer configurations
            configs = [
                ("Low_Compression", ApproximateLinearConfig(
                    backend=BackendType.FAISS,
                    faiss_clusters=256,
                    energy_keep=0.95
                )),
                ("Medium_Compression", ApproximateLinearConfig(
                    backend=BackendType.FAISS,
                    faiss_clusters=128,
                    energy_keep=0.9
                )),
                ("High_Compression", ApproximateLinearConfig(
                    backend=BackendType.FAISS,
                    faiss_clusters=64,
                    energy_keep=0.8
                ))
            ]
            
            for name, config in configs:
                # Create layer
                layer = ApproximateLinear(in_features, out_features, config)
                
                # Initialize
                calibration_data = torch.randn(1000, in_features)
                layer.initialize(calibration_data=calibration_data)
                
                # Estimate memory usage
                approx_memory = (
                    layer.weight.numel() * 4 +  # Parameters
                    (layer.bias.numel() if layer.bias is not None else 0) * 4 +
                    layer.whitening_matrix.numel() * 4 +  # Buffers
                    layer.key_scales.numel() * 4 +
                    layer.rank_kept * in_features * 4 +  # Keys (approximate)
                    layer.rank_kept * out_features * 4   # Values (approximate)
                ) / 1024 / 1024  # MB
                
                compression_ratio = dense_memory / approx_memory
                
                results.append(BenchmarkResult(
                    name=f"Memory_{name}_{in_features}x{out_features}",
                    time_taken=0.0,  # Not timing memory
                    memory_used=approx_memory,
                    accuracy=compression_ratio,
                    parameters={
                        "dense_memory_mb": dense_memory,
                        "approx_memory_mb": approx_memory,
                        "compression_ratio": compression_ratio,
                        "in_features": in_features,
                        "out_features": out_features,
                        "rank_kept": layer.rank_kept,
                        "energy_keep": config.energy_keep
                    }
                ))
        
        self.results.extend(results)
        return results
    
    def benchmark_gpu_support(self) -> List[BenchmarkResult]:
        """Benchmark GPU support if available."""
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU benchmarks")
            return []
        
        print("Benchmarking GPU support...")
        results = []
        
        # Test configurations
        config = ApproximateLinearConfig(
            backend=BackendType.FAISS,
            faiss_clusters=256,
            faiss_probe_clusters=32,
            faiss_key_subvectors=8,
            faiss_key_bits=8,
            faiss_value_subvectors=8,
            faiss_value_bits=8,
            energy_keep=0.9
        )
        
        layer_size = (512, 1024)
        batch_size = 100
        
        # CPU benchmark
        print("  Testing CPU")
        layer_cpu = ApproximateLinear(layer_size[0], layer_size[1], config)
        calibration_data_cpu = torch.randn(1000, layer_size[0])
        layer_cpu.initialize(calibration_data=calibration_data_cpu)
        
        input_cpu = torch.randn(batch_size, layer_size[0])
        
        # Warm up
        for _ in range(10):
            _ = layer_cpu(input_cpu)
        
        # CPU timing
        start_time = time.time()
        for _ in range(100):
            output_cpu = layer_cpu(input_cpu)
        cpu_time = time.time() - start_time
        
        # GPU benchmark
        print("  Testing GPU")
        config_gpu = ApproximateLinearConfig(
            backend=BackendType.FAISS,
            faiss_clusters=256,
            faiss_probe_clusters=32,
            faiss_key_subvectors=8,
            faiss_key_bits=8,
            faiss_value_subvectors=8,
            faiss_value_bits=8,
            energy_keep=0.9,
            faiss_use_gpu=True,
            device="cuda"
        )
        
        layer_gpu = ApproximateLinear(layer_size[0], layer_size[1], config_gpu)
        calibration_data_gpu = torch.randn(1000, layer_size[0], device="cuda")
        layer_gpu.initialize(calibration_data=calibration_data_gpu)
        
        input_gpu = torch.randn(batch_size, layer_size[0], device="cuda")
        
        # Warm up
        for _ in range(10):
            _ = layer_gpu(input_gpu)
        
        # GPU timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            output_gpu = layer_gpu(input_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # Compare accuracy
        output_cpu_gpu = output_cpu.to("cuda")
        accuracy = 1.0 - torch.norm(output_gpu - output_cpu_gpu) / torch.norm(output_cpu_gpu)
        
        results.append(BenchmarkResult(
            name="GPU_vs_CPU",
            time_taken=gpu_time / cpu_time,  # Speedup ratio
            memory_used=0.0,
            accuracy=accuracy.item(),
            parameters={
                "cpu_time": cpu_time / 100,
                "gpu_time": gpu_time / 100,
                "speedup": cpu_time / gpu_time,
                "batch_size": batch_size,
                "layer_size": layer_size
            }
        ))
        
        self.results.extend(results)
        return results
    
    def generate_plots(self):
        """Generate comprehensive plots from benchmark results."""
        print("Generating plots...")
        
        # Group results by category
        categories = defaultdict(list)
        for result in self.results:
            if result.name.startswith("SVD_"):
                categories["SVD"].append(result)
            elif result.name.startswith("Covariance_"):
                categories["Covariance"].append(result)
            elif result.name.startswith("Whitening_"):
                categories["Whitening"].append(result)
            elif result.name.startswith("Init_"):
                categories["Initialization"].append(result)
            elif result.name.startswith("Forward_"):
                categories["Forward_Pass"].append(result)
            elif result.name.startswith("Memory_"):
                categories["Memory"].append(result)
            elif result.name.startswith("GPU_"):
                categories["GPU"].append(result)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("AXL Library Benchmark Results", fontsize=16)
        
        # 1. Utility functions performance
        if "SVD" in categories:
            ax = axes[0, 0]
            svd_results = categories["SVD"]
            sizes = [f"{r.parameters['out_features']}x{r.parameters['in_features']}" 
                    for r in svd_results]
            times = [r.time_taken * 1000 for r in svd_results]  # Convert to ms
            
            ax.bar(sizes, times)
            ax.set_title("SVD Decomposition Time")
            ax.set_ylabel("Time (ms)")
            ax.tick_params(axis='x', rotation=45)
        
        # 2. Initialization performance by backend
        if "Initialization" in categories:
            ax = axes[0, 1]
            init_results = categories["Initialization"]
            
            backends = defaultdict(list)
            for result in init_results:
                backend = result.parameters.get("backend", "unknown")
                backends[backend].append(result.time_taken)
            
            backend_names = list(backends.keys())
            backend_times = [np.mean(times) for times in backends.values()]
            
            ax.bar(backend_names, backend_times)
            ax.set_title("Average Initialization Time by Backend")
            ax.set_ylabel("Time (s)")
        
        # 3. Forward pass performance
        if "Forward_Pass" in categories:
            ax = axes[0, 2]
            forward_results = categories["Forward_Pass"]
            
            # Group by batch size
            batch_sizes = defaultdict(list)
            for result in forward_results:
                batch_size = result.parameters.get("batch_size", 0)
                batch_sizes[batch_size].append(result.time_taken * 1000)  # ms
            
            batch_names = sorted(batch_sizes.keys())
            batch_times = [np.mean(times) for times in [batch_sizes[b] for b in batch_names]]
            
            ax.bar([str(b) for b in batch_names], batch_times)
            ax.set_title("Forward Pass Time by Batch Size")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Time (ms)")
        
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
        
        # 5. Accuracy vs Speed trade-off
        if "Forward_Pass" in categories:
            ax = axes[1, 1]
            forward_results = categories["Forward_Pass"]
            
            times = [r.time_taken * 1000 for r in forward_results]  # ms
            accuracies = [r.accuracy for r in forward_results]
            
            ax.scatter(times, accuracies, alpha=0.7)
            ax.set_title("Accuracy vs Speed Trade-off")
            ax.set_xlabel("Time per Forward Pass (ms)")
            ax.set_ylabel("Accuracy")
        
        # 6. GPU vs CPU performance
        if "GPU" in categories:
            ax = axes[1, 2]
            gpu_results = categories["GPU"]
            
            if gpu_results:
                result = gpu_results[0]
                speedup = result.parameters.get("speedup", 1.0)
                accuracy = result.accuracy
                
                ax.bar(["GPU Speedup"], [speedup])
                ax.set_title(f"GPU vs CPU Performance\nAccuracy: {accuracy:.3f}")
                ax.set_ylabel("Speedup Factor")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "benchmark_results.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        results_dict = [result.to_dict() for result in self.results]
        
        with open(os.path.join(self.output_dir, "benchmark_results.json"), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/benchmark_results.json")
    
    def run_all_benchmarks(self):
        """Run all benchmarks and generate reports."""
        print("Starting comprehensive AXL benchmark suite...")
        
        # Run all benchmarks
        self.benchmark_utils_functions()
        self.benchmark_layer_initialization()
        self.benchmark_forward_pass()
        self.benchmark_memory_efficiency()
        self.benchmark_gpu_support()
        
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
    benchmark = AXLBenchmark()
    benchmark.run_all_benchmarks() 