"""
Example: Using Sparse-Aware FFN Tensor Parallel System

This example demonstrates how to use the complete system for
sparse FFN inference with tensor parallelism.
"""

import torch
import torch.nn as nn
import numpy as np
from runtime import create_sparse_tp_ffn, SparseTPFFNConfig
from solver import NeuronPartitioner, GPUBalancer, TCCCBalancer, PipelineScheduler


def example_basic_usage():
    """Basic usage example."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Step 1: Create SparseTPFFN instance
    print("\n1. Creating SparseTPFFN instance...")
    ffn = create_sparse_tp_ffn(
        hidden_dim=4096,
        intermediate_dim=16384,
        num_gpus=8,
        rank=0,  # Current GPU rank
        activation='gelu',
        activation_threshold=0.1,
        target_sparsity=0.5,
        num_chunks=2,
    )
    
    # Step 2: Generate or load sample activation masks
    print("\n2. Generating sample activation masks...")
    # In practice, these would come from running the predictor on representative data
    num_samples = 1000
    num_neurons = 16384
    sample_masks = np.random.random((num_samples, num_neurons)) > 0.5
    
    # Add structure: first 30% neurons are more active (HANS)
    sample_masks[:, :int(num_neurons * 0.3)] = np.random.random(
        (num_samples, int(num_neurons * 0.3))
    ) > 0.3
    
    # Step 3: Run offline analysis
    print("\n3. Running offline analysis...")
    analysis_results = ffn.offline_analysis(
        sample_masks=sample_masks,
        balance_strategy='hybrid'
    )
    
    print(f"\nAnalysis completed:")
    print(f"  - TC neurons: {analysis_results['tc_cc_info']['tc_count']}")
    print(f"  - CC neurons: {analysis_results['tc_cc_info']['cc_count']}")
    print(f"  - Overlap efficiency: {analysis_results['tc_cc_info']['overlap_efficiency']:.4f}")
    
    # Step 4: Run inference
    print("\n4. Running inference...")
    if torch.cuda.is_available():
        ffn = ffn.cuda()
        x = torch.randn(2048, 4096, device='cuda', dtype=torch.float16)
        
        # Run forward pass
        output = ffn(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Get statistics
        stats = ffn.get_statistics()
        print(f"  Average time: {stats.get('avg_time_ms', 0):.4f} ms")
    else:
        print("  CUDA not available, skipping inference")
    
    print("\n✓ Basic usage example completed")


def example_custom_predictor():
    """Example with custom predictor."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Predictor Integration")
    print("=" * 80)
    
    # Define a simple predictor
    class SimplePredictor(nn.Module):
        """Simple MLP-based predictor for neuron activation."""
        
        def __init__(self, hidden_dim, intermediate_dim, low_rank_dim=256):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, low_rank_dim, bias=False)
            self.fc2 = nn.Linear(low_rank_dim, intermediate_dim, bias=False)
        
        def forward(self, x):
            # x: [seq_len, hidden_dim]
            h = torch.relu(self.fc1(x))
            logits = self.fc2(h)
            # Return binary mask
            mask = (torch.sigmoid(logits) > 0.5).float()
            return mask
    
    # Create predictor
    hidden_dim = 4096
    intermediate_dim = 16384
    predictor = SimplePredictor(hidden_dim, intermediate_dim)
    
    # Create SparseTPFFN with predictor
    print("\n1. Creating SparseTPFFN with custom predictor...")
    config = SparseTPFFNConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_gpus=8,
        rank=0,
    )
    
    ffn = create_sparse_tp_ffn(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_gpus=8,
        rank=0,
    )
    ffn.predictor = predictor
    
    print("\n✓ Custom predictor example completed")


def example_solver_components():
    """Example using individual solver components."""
    print("\n" + "=" * 80)
    print("Example 3: Using Individual Solver Components")
    print("=" * 80)
    
    # Step 1: Neuron Partitioning
    print("\n1. Neuron Partitioning (Layer 1 Solver)")
    print("-" * 40)
    
    partitioner = NeuronPartitioner(
        num_neurons=16384,
        num_gpus=8,
        activation_threshold=0.1,
    )
    
    # Generate sample masks
    masks = np.random.random((500, 16384)) > 0.5
    
    # Perform partitioning
    partitions = partitioner.analyze_and_partition(
        masks=masks,
        method='spectral',
        balance_strategy='adaptive'
    )
    
    info = partitioner.get_partition_info()
    print(f"Total HANS neurons: {info['total_hans']}")
    print(f"Total LANS neurons: {info['total_lans']}")
    
    # Step 2: GPU Balancing
    print("\n2. GPU Balancing (Layer 2 Solver)")
    print("-" * 40)
    
    balancer = GPUBalancer(
        num_gpus=8,
        hidden_dim=4096,
        intermediate_dim=16384,
    )
    
    # Get activation frequencies from analyzer
    frequencies = partitioner.analyzer.compute_activation_frequencies()
    coactivation_weights = partitioner.analyzer.compute_coactivation_weights()
    
    balanced_partitions, balance_info = balancer.balance_partitions(
        partitions=partitions,
        coactivation_weights=coactivation_weights,
        activation_frequencies=frequencies,
        strategy='compute_aware'
    )
    
    print(f"Balance strategy: {balance_info['strategy']}")
    print(f"Compute time std: {balance_info['compute_time_std_ms']:.4f} ms")
    
    # Step 3: TC/CC Balancing
    print("\n3. TC/CC Balancing (Layer 2 Solver)")
    print("-" * 40)
    
    tc_cc_balancer = TCCCBalancer(
        hidden_dim=4096,
        intermediate_dim=16384,
    )
    
    # Balance for GPU 0
    partition = balanced_partitions[0]
    assignment = tc_cc_balancer.balance(
        hans_indices=partition.hans_indices,
        lans_indices=partition.lans_indices,
        activation_frequencies=frequencies,
        sequence_length=2048,
        search_method='heuristic'
    )
    
    print(f"TC neurons: {len(assignment.tc_indices)}")
    print(f"CC neurons: {len(assignment.cc_indices)}")
    print(f"TC/CC time difference: {abs(assignment.tc_time - assignment.cc_time):.4f} ms")
    
    # Step 4: Pipeline Scheduling
    print("\n4. Pipeline Scheduling (Layer 3 Solver)")
    print("-" * 40)
    
    scheduler = PipelineScheduler(
        num_gpus=8,
        hidden_dim=4096,
    )
    
    schedule = scheduler.schedule(
        sequence_length=2048,
        activation_sparsity=0.5,
        compute_time_per_token=0.001,
        strategy='two_chunk'
    )
    
    print(f"Number of chunks: {len(schedule.chunks)}")
    print(f"Total time: {schedule.total_time:.4f} ms")
    print(f"Overlap efficiency: {schedule.overlap_efficiency:.4f}")
    
    print("\n✓ Individual solver components example completed")


def example_communication_optimization():
    """Example of communication optimization."""
    print("\n" + "=" * 80)
    print("Example 4: Communication Optimization Analysis")
    print("=" * 80)
    
    from communication import estimate_sparse_communication_benefit
    
    # Analyze communication benefits for different sparsity levels
    print("\nCommunication benefit analysis:")
    print("-" * 80)
    print(f"{'Sparsity':<12} {'Dense (ms)':<12} {'Sparse (ms)':<12} {'Speedup':<10} {'Compression':<12}")
    print("-" * 80)
    
    for sparsity in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        benefit = estimate_sparse_communication_benefit(
            hidden_dim=4096,
            num_gpus=8,
            sparsity=sparsity,
        )
        
        print(f"{sparsity:<12.2f} {benefit['dense_time_ms']:<12.4f} "
              f"{benefit['sparse_time_ms']:<12.4f} {benefit['speedup']:<10.2f}x "
              f"{benefit['compression_ratio']:<12.2f}")
    
    print("\n✓ Communication optimization example completed")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("SPARSE-AWARE FFN TENSOR PARALLEL SYSTEM - USAGE EXAMPLES")
    print("=" * 80)
    
    example_basic_usage()
    example_custom_predictor()
    example_solver_components()
    example_communication_optimization()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
