"""
Improved example demonstrating the enhanced ApproximateLinear API.

This script showcases the improved API design with:
- Better naming conventions
- Proper GPU support
- Enhanced tensor handling
- More intuitive configuration
- Better error handling
"""

import torch
import numpy as np
from axl.layer import ApproximateLinear, ApproximateLinearConfig, BackendType, QuantizationType


def demonstrate_basic_usage():
    """Demonstrate basic usage of the improved API."""
    print("=== Basic Usage Demo ===")
    
    # Create initial config
    config = ApproximateLinearConfig(
        backend=BackendType.NANOPQ,
        energy_keep=0.95,
        expected_keys_per_query=32,
        nanopq_key_codewords=32,  # Reduced for small dataset
        nanopq_value_codewords=32,  # Reduced for small dataset
        use_bias=True,
        device="cpu"
    )

    # Generate weight and perform SVD to get rank_kept
    out_features, in_features = 64, 128
    weight_matrix = np.random.randn(out_features, in_features).astype(np.float32)
    from axl.utils import svd_keys_values
    U, S, V, K = svd_keys_values(weight_matrix, energy_keep=config.energy_keep)
    rank_kept = U.shape[1]
    print(f"SVD rank_kept: {rank_kept}")
    # Find a divisor of rank_kept for nanopq_value_subvectors
    best_div = None
    for div in range(1, min(rank_kept, 32) + 1):
        if rank_kept % div == 0:
            best_div = div
    if best_div:
        config.nanopq_value_subvectors = best_div
    print(f"Adjusted nanopq_value_subvectors: {config.nanopq_value_subvectors}")

    # Create layer with adjusted config
    layer = ApproximateLinear(in_features=in_features, out_features=out_features, config=config)

    # Initialize with calibration data
    calibration_data = torch.randn(1000, in_features)
    layer.initialize(calibration_data=calibration_data)

    # Forward pass
    input_data = torch.randn(16, in_features)
    output = layer(input_data)

    # If output shape is (batch, rank_kept), project back to (batch, out_features)
    if output.shape[1] == rank_kept and output.shape[1] != out_features:
        # Use U from SVD to project back
        output = output @ torch.from_numpy(U.T).float()
        print(f"Output projected back to out_features: {output.shape}")

    # Add bias if present and shapes match
    if hasattr(layer, 'bias') and layer.bias is not None and output.shape[1] == out_features:
        output = output + layer.bias

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Layer initialized: {layer.is_initialized}")
    print()


def demonstrate_gpu_support():
    """Demonstrate GPU support if available."""
    print("=== GPU Support Demo ===")
    
    if torch.cuda.is_available():
        print("CUDA is available!")
        
        # Create configuration for GPU
        config = ApproximateLinearConfig(
            backend=BackendType.FAISS,
            faiss_use_gpu=True,
            faiss_clusters=128,  # Reduce clusters for small dataset
            device="cuda"
        )
        
        # Create layer on GPU
        layer = ApproximateLinear(in_features=256, out_features=128, config=config)
        layer = layer.to("cuda")
        
        # Initialize with GPU data
        calibration_data = torch.randn(2000, 256, device="cuda")
        layer.initialize(calibration_data=calibration_data)
        
        # Forward pass on GPU
        input_data = torch.randn(32, 256, device="cuda")
        output = layer(input_data)
        
        print(f"Layer device: {layer.device}")
        print(f"Input device: {input_data.device}")
        print(f"Output device: {output.device}")
        print(f"Layer initialized: {layer.is_initialized}")
    else:
        print("CUDA not available, skipping GPU demo")
    print()


def demonstrate_configuration_management():
    """Demonstrate configuration management features."""
    print("=== Configuration Management Demo ===")
    
    # Create initial configuration
    config = ApproximateLinearConfig(
        backend=BackendType.NANOPQ,
        energy_keep=0.9,
        expected_keys_per_query=16,
        nanopq_key_codewords=32,  # Reduced for small dataset
        nanopq_value_codewords=32  # Reduced for small dataset
    )
    
    # Create layer
    layer = ApproximateLinear(in_features=64, out_features=32, config=config)
    
    # Get current configuration
    current_config = layer.get_config()
    print("Current configuration:")
    for key, value in current_config.items():
        print(f"  {key}: {value}")
    
    # Update configuration
    new_config = ApproximateLinearConfig(
        backend=BackendType.NANOPQ,
        energy_keep=0.95,
        expected_keys_per_query=32,
        nanopq_key_codewords=64,
        nanopq_value_codewords=64
    )
    
    layer.set_config(new_config)
    print("\nConfiguration updated!")
    print()


def demonstrate_error_handling():
    """Demonstrate improved error handling."""
    print("=== Error Handling Demo ===")
    
    # Test invalid configuration
    try:
        config = ApproximateLinearConfig(energy_keep=1.5)  # Invalid value
        print("This should not print")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test invalid layer dimensions
    try:
        layer = ApproximateLinear(in_features=0, out_features=32)  # Invalid dimensions
        print("This should not print")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test uninitialized layer
    try:
        layer = ApproximateLinear(in_features=64, out_features=32)
        input_data = torch.randn(16, 64)
        output = layer(input_data)  # Should work in training mode
        print("Training mode works without initialization")
        
        layer.eval()
        output = layer(input_data)  # Should work in eval mode without approx
        print("Eval mode works without initialization (dense computation)")
        
        # Enable approximate inference
        layer.config.use_approximate_inference = True
        output = layer(input_data)  # Should fail
        print("This should not print")
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    print()


def demonstrate_training_vs_inference():
    """Demonstrate training vs inference behavior."""
    print("=== Training vs Inference Demo ===")
    
    # Create configuration
    config = ApproximateLinearConfig(
        backend=BackendType.NANOPQ,
        energy_keep=0.9,
        expected_keys_per_query=16,
        nanopq_key_codewords=32,  # Reduced for small dataset
        nanopq_value_codewords=32,  # Reduced for small dataset
        use_straight_through_estimator=True,
        use_approximate_inference=True
    )
    
    # Create layer
    layer = ApproximateLinear(in_features=128, out_features=64, config=config)
    
    # Initialize
    calibration_data = torch.randn(1000, 128)
    layer.initialize(calibration_data=calibration_data)
    
    # Test input
    input_data = torch.randn(8, 128)
    
    # Training mode
    layer.train()
    training_output = layer(input_data)
    print(f"Training mode output shape: {training_output.shape}")
    print(f"Training mode requires grad: {training_output.requires_grad}")
    
    # Inference mode
    layer.eval()
    inference_output = layer(input_data)
    print(f"Inference mode output shape: {inference_output.shape}")
    print(f"Inference mode requires grad: {inference_output.requires_grad}")
    
    # Compare with dense computation
    layer.config.use_approximate_inference = False
    dense_output = layer(input_data)
    
    # Compute relative error
    relative_error = float(
        (inference_output - dense_output).norm() / 
        (dense_output.norm() + 1e-12)
    )
    print(f"Relative error: {relative_error:.6f}")
    print()


def demonstrate_backend_comparison():
    """Demonstrate different backend configurations."""
    print("=== Backend Comparison Demo ===")
    
    in_features_nano = 256
    out_features_nano = 256
    
    out_features_faiss = 5128
    in_features_faiss = 5128
    
    

    backends = [
        (BackendType.NANOPQ, "NanoPQ"),
        (BackendType.FAISS, "FAISS")
    ]
    
    for backend_type, name in backends:
        print(f"\nTesting {name} backend:")
        
        try:
            # Create configuration
            if backend_type == BackendType.FAISS:
                config = ApproximateLinearConfig(
                    backend=backend_type,
                    energy_keep=0.9,
                    expected_keys_per_query=16,
                    faiss_clusters = 16,
                    faiss_probe_clusters = 8,
                    faiss_key_subvectors = 8,
                    faiss_key_bits = 8,
                    faiss_value_subvectors = 8,
                    faiss_value_bits = 8,
                    faiss_use_gpu= False,
                    
                )
                layer = ApproximateLinear(in_features=in_features_faiss, out_features=out_features_faiss, config=config)
                input_data = torch.randn(32, in_features_faiss)
                calibration_data = torch.randn(5000, in_features_faiss)
    
            else:
                config = ApproximateLinearConfig(
                    backend=backend_type,
                    energy_keep=0.9,
                    expected_keys_per_query=16,
                    nanopq_key_codewords=32,
                    nanopq_value_codewords=32
                )
                layer = ApproximateLinear(in_features=in_features_nano, out_features=out_features_nano, config=config)
                input_data = torch.randn(32, in_features_nano)
                calibration_data = torch.randn(5000, in_features_nano)
    
            print('init eval of FAISS layer')
            layer.initialize(calibration_data=calibration_data)
            # Test forward pass
            print('Before eval of FAISS layer')
            layer.eval()
            print('After eval of FAISS layer')
            output = layer(input_data)
            print(f"  Success! Output shape: {output.shape}")
            print(f"  Rank kept: {layer.rank_kept}")
        except Exception as e:
            print(f"  Error: {e}")
    print()


def main():
    """Main demonstration function."""
    print("AXL Improved API Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_basic_usage()
    demonstrate_gpu_support()
    demonstrate_configuration_management()
    demonstrate_error_handling()
    demonstrate_training_vs_inference()
    demonstrate_backend_comparison()
    
    print("Demonstration completed successfully!")


if __name__ == "__main__":
    main()