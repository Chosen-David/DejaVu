"""
Sparse-Aware FFN Tensor Parallel Optimization System

Main entry point for the sparse-aware FFN tensor parallel optimization system.

This system provides:
1. Offline neuron partitioning and load balancing
2. Runtime sparse computation with hybrid TC/CC execution
3. Pipeline scheduling for compute-communication overlap
4. Sparse communication optimization

Usage:
    # Basic usage
    from runtime import create_sparse_tp_ffn
    
    ffn = create_sparse_tp_ffn(
        hidden_dim=4096,
        intermediate_dim=16384,
        num_gpus=8,
        rank=0
    )
    
    # Run offline analysis
    analysis_results = ffn.offline_analysis(sample_masks)
    
    # Run inference
    output = ffn(input_tensor)

Modules:
    - solver: Three-layer optimization solvers
    - kernels: High-performance CUDA/Triton kernels
    - communication: Sparse-aware communication primitives
    - runtime: End-to-end system integration
    - examples: Usage examples and tutorials
"""

__version__ = "0.1.0"
__author__ = "Sparse FFN Research Team"

# Import main components for easy access
from .runtime import SparseTPFFN, SparseTPFFNConfig, create_sparse_tp_ffn
from .solver import (
    CoactivationAnalyzer,
    NeuronPartitioner,
    GPUBalancer,
    TCCCBalancer,
    PipelineScheduler,
)
from .kernels import FlashFFN
from .communication import create_sparse_communicator

__all__ = [
    # Main entry point
    'SparseTPFFN',
    'SparseTPFFNConfig',
    'create_sparse_tp_ffn',
    
    # Solver components
    'CoactivationAnalyzer',
    'NeuronPartitioner',
    'GPUBalancer',
    'TCCCBalancer',
    'PipelineScheduler',
    
    # Kernel
    'FlashFFN',
    
    # Communication
    'create_sparse_communicator',
]


def get_system_info():
    """Get system information and capabilities."""
    import torch
    
    info = {
        'version': __version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    
    return info


if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("Sparse-Aware FFN Tensor Parallel Optimization System")
    print("=" * 80)
    
    info = get_system_info()
    print(f"\nSystem Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nUsage:")
    print("  python -m tests.test_sparse_tp_ffn  # Run tests")
    print("  python -m examples.example_usage     # Run examples")
    print("\nFor more information, see the documentation in each module.")
