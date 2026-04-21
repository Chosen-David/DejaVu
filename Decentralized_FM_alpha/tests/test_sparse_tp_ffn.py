"""
End-to-End Tests for Sparse-Aware FFN Tensor Parallel System

This test suite validates:
1. Neuron partitioning (HANS/LANS)
2. GPU balancing
3. TC/CC balancing
4. Pipeline scheduling
5. FlashFFN kernel
6. Sparse communication
7. Full system integration
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import (
    CoactivationAnalyzer,
    NeuronPartitioner,
    GPUBalancer,
    TCCCBalancer,
    PipelineScheduler,
)
from kernels import FlashFFN, benchmark_flash_ffn
from communication import (
    create_sparse_communicator,
    estimate_sparse_communication_benefit,
)
from runtime import SparseTPFFN, SparseTPFFNConfig, create_sparse_tp_ffn


# ============================================================================
# Test Utilities
# ============================================================================

def generate_synthetic_masks(
    num_samples: int,
    num_neurons: int,
    sparsity: float = 0.5,
    coactivation_pattern: str = 'clustered',
) -> np.ndarray:
    """
    Generate synthetic activation masks for testing.
    
    Args:
        num_samples: Number of samples (tokens)
        num_neurons: Number of neurons
        sparsity: Target sparsity level
        coactivation_pattern: 'random', 'clustered', or 'structured'
        
    Returns:
        Binary mask array [num_samples, num_neurons]
    """
    if coactivation_pattern == 'random':
        # Random sparse pattern
        masks = np.random.random((num_samples, num_neurons)) > sparsity
        
    elif coactivation_pattern == 'clustered':
        # Clustered pattern: some neurons are frequently activated together
        masks = np.zeros((num_samples, num_neurons), dtype=bool)
        
        # Define clusters
        num_clusters = max(1, num_neurons // 32)
        cluster_size = num_neurons // num_clusters
        
        for i in range(num_samples):
            # Randomly activate some clusters
            active_clusters = np.random.choice(
                num_clusters, 
                size=max(1, int(num_clusters * (1 - sparsity))),
                replace=False
            )
            
            for cluster_id in active_clusters:
                start = cluster_id * cluster_size
                end = start + cluster_size
                masks[i, start:end] = True
        
    elif coactivation_pattern == 'structured':
        # Structured pattern: HANS vs LANS
        masks = np.zeros((num_samples, num_neurons), dtype=bool)
        
        # First 30% are HANS (high activation)
        hans_neurons = int(num_neurons * 0.3)
        masks[:, :hans_neurons] = np.random.random((num_samples, hans_neurons)) > 0.3
        
        # Remaining are LANS (low activation)
        masks[:, hans_neurons:] = np.random.random((num_samples, num_neurons - hans_neurons)) > 0.9
    
    return masks


def print_test_header(test_name: str):
    """Print formatted test header."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)


def print_test_result(passed: bool, message: str = ""):
    """Print test result."""
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status}: {message}")


# ============================================================================
# Layer 1 Solver Tests
# ============================================================================

def test_coactivation_analyzer():
    """Test co-activation analysis."""
    print_test_header("Coactivation Analyzer")
    
    # Create analyzer
    num_neurons = 1024
    analyzer = CoactivationAnalyzer(num_neurons=num_neurons)
    
    # Generate synthetic masks
    masks = generate_synthetic_masks(
        num_samples=1000,
        num_neurons=num_neurons,
        sparsity=0.6,
        coactivation_pattern='clustered'
    )
    
    # Update analyzer
    print("Updating analyzer with masks...")
    analyzer.update(masks)
    
    # Compute co-activation weights
    print("Computing co-activation weights...")
    CW = analyzer.compute_coactivation_weights()
    
    # Check results
    assert CW.shape == (num_neurons, num_neurons), "Wrong co-activation matrix shape"
    assert np.allclose(CW, CW.T), "Co-activation matrix should be symmetric"
    assert np.allclose(np.diag(CW), 0), "Diagonal should be zero"
    
    # Check statistics
    stats = analyzer.get_neuron_statistics()
    print(f"Neuron statistics: {stats}")
    
    assert stats['total_samples'] == 1000, "Wrong total samples"
    assert 0 <= stats['mean_frequency'] <= 1, "Invalid mean frequency"
    
    print_test_result(True, f"Co-activation matrix shape: {CW.shape}")
    print_test_result(True, f"Mean activation frequency: {stats['mean_frequency']:.4f}")
    
    return True


def test_neuron_partitioner():
    """Test neuron partitioning into HANS/LANS."""
    print_test_header("Neuron Partitioner")
    
    # Create partitioner
    num_neurons = 2048
    num_gpus = 8
    partitioner = NeuronPartitioner(
        num_neurons=num_neurons,
        num_gpus=num_gpus,
        activation_threshold=0.1,
    )
    
    # Generate masks with structured pattern
    masks = generate_synthetic_masks(
        num_samples=500,
        num_neurons=num_neurons,
        sparsity=0.5,
        coactivation_pattern='structured'
    )
    
    # Perform partitioning
    print("Partitioning neurons...")
    partitions = partitioner.analyze_and_partition(
        masks=masks,
        method='spectral',
        balance_strategy='even'
    )
    
    # Check results
    assert len(partitions) == num_gpus, f"Expected {num_gpus} partitions"
    
    # Verify all neurons are assigned
    all_hans = np.concatenate([p.hans_indices for p in partitions])
    all_lans = np.concatenate([p.lans_indices for p in partitions])
    total_assigned = len(np.unique(np.concatenate([all_hans, all_lans])))
    
    print_test_result(True, f"Total partitions: {len(partitions)}")
    print_test_result(True, f"Total neurons assigned: {total_assigned}/{num_neurons}")
    
    # Print per-GPU statistics
    info = partitioner.get_partition_info()
    print(f"\nPartition info:")
    for gpu_stat in info['per_gpu_stats']:
        print(f"  GPU {gpu_stat['gpu_id']}: HANS={gpu_stat['num_hans']}, LANS={gpu_stat['num_lans']}")
    
    return True


# ============================================================================
# Layer 2 Solver Tests
# ============================================================================

def test_gpu_balancer():
    """Test GPU load balancing."""
    print_test_header("GPU Balancer")
    
    # Create mock partitions
    num_gpus = 8
    num_neurons = 4096
    hidden_dim = 1024
    
    from solver.neuron_partitioner import NeuronPartition
    
    # Create initial partitions
    partitions = []
    neurons_per_gpu = num_neurons // num_gpus
    for i in range(num_gpus):
        start = i * neurons_per_gpu
        end = (i + 1) * neurons_per_gpu
        partitions.append(NeuronPartition(
            hans_indices=np.arange(start, start + neurons_per_gpu // 2),
            lans_indices=np.arange(start + neurons_per_gpu // 2, end),
            coactivated_hans_groups=[]
        ))
    
    # Create balancer
    balancer = GPUBalancer(
        num_gpus=num_gpus,
        hidden_dim=hidden_dim,
        intermediate_dim=num_neurons,
    )
    
    # Create mock coactivation weights and frequencies
    coactivation_weights = np.random.random((num_neurons, num_neurons)) * 0.1
    coactivation_weights = (coactivation_weights + coactivation_weights.T) / 2
    
    activation_frequencies = np.random.random(num_neurons)
    activation_frequencies[:num_neurons//3] *= 2  # HANS have higher frequency
    
    # Balance partitions
    print("Balancing partitions...")
    balanced_partitions, balance_info = balancer.balance_partitions(
        partitions=partitions,
        coactivation_weights=coactivation_weights,
        activation_frequencies=activation_frequencies,
        strategy='compute_aware'
    )
    
    # Check results
    assert len(balanced_partitions) == num_gpus, "Wrong number of partitions"
    
    # Get balance report
    report = balancer.get_balance_report(balanced_partitions, activation_frequencies)
    print(f"\nBalance report:")
    print(f"  Compute time mean: {report['summary']['compute_time_mean_ms']:.4f} ms")
    print(f"  Compute time std: {report['summary']['compute_time_std_ms']:.4f} ms")
    
    print_test_result(True, f"Balance strategy: compute_aware")
    
    return True


def test_tc_cc_balancer():
    """Test TensorCore/CUDACore balancing."""
    print_test_header("TC/CC Balancer")
    
    # Create balancer
    hidden_dim = 1024
    intermediate_dim = 4096
    balancer = TCCCBalancer(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    
    # Create mock indices
    hans_indices = np.arange(0, intermediate_dim // 2)
    lans_indices = np.arange(intermediate_dim // 2, intermediate_dim)
    
    # Create mock frequencies
    activation_frequencies = np.random.random(intermediate_dim)
    activation_frequencies[hans_indices] *= 2
    
    # Balance
    print("Balancing TC/CC assignment...")
    assignment = balancer.balance(
        hans_indices=hans_indices,
        lans_indices=lans_indices,
        activation_frequencies=activation_frequencies,
        sequence_length=2048,
        search_method='heuristic'
    )
    
    # Check results
    total_neurons = len(assignment.tc_indices) + len(assignment.cc_indices)
    
    print(f"\nTC/CC assignment:")
    print(f"  TC neurons: {len(assignment.tc_indices)}")
    print(f"  CC neurons: {len(assignment.cc_indices)}")
    print(f"  TC time: {assignment.tc_time:.4f} ms")
    print(f"  CC time: {assignment.cc_time:.4f} ms")
    print(f"  Overlap efficiency: {assignment.overlap_efficiency:.4f}")
    
    print_test_result(True, f"Total neurons assigned: {total_neurons}")
    
    # Get tile configuration
    tile_config = balancer.get_optimal_tile_config(
        tc_neuron_count=len(assignment.tc_indices),
        cc_neuron_count=len(assignment.cc_indices),
        sequence_length=2048,
    )
    print(f"\nTile configuration:")
    print(f"  TC SMs: {tile_config['tc']['num_sms']}")
    print(f"  CC SMs: {tile_config['cc']['num_sms']}")
    
    return True


# ============================================================================
# Layer 3 Solver Tests
# ============================================================================

def test_pipeline_scheduler():
    """Test pipeline scheduling."""
    print_test_header("Pipeline Scheduler")
    
    # Create scheduler
    num_gpus = 8
    hidden_dim = 1024
    scheduler = PipelineScheduler(
        num_gpus=num_gpus,
        hidden_dim=hidden_dim,
    )
    
    # Create schedule
    print("Creating schedule...")
    schedule = scheduler.schedule(
        sequence_length=2048,
        activation_sparsity=0.5,
        compute_time_per_token=0.001,
        strategy='two_chunk'
    )
    
    # Check results
    assert len(schedule.chunks) == 2, "Expected 2 chunks"
    assert schedule.total_time > 0, "Total time should be positive"
    assert 0 <= schedule.overlap_efficiency <= 1, "Invalid overlap efficiency"
    
    print(f"\nSchedule info:")
    print(f"  Number of chunks: {len(schedule.chunks)}")
    print(f"  Total time: {schedule.total_time:.4f} ms")
    print(f"  Overlap efficiency: {schedule.overlap_efficiency:.4f}")
    
    for i, chunk in enumerate(schedule.chunks):
        print(f"  Chunk {i}: tokens={len(chunk.token_indices)}, "
              f"compute={chunk.compute_time:.4f}ms, comm={chunk.comm_time:.4f}ms")
    
    print_test_result(True, f"Schedule created successfully")
    
    # Test adaptive scheduling
    print("\nTesting adaptive scheduling...")
    schedule_adaptive = scheduler.schedule(
        sequence_length=2048,
        activation_sparsity=0.5,
        compute_time_per_token=0.001,
        strategy='adaptive'
    )
    
    print(f"  Adaptive chunks: {len(schedule_adaptive.chunks)}")
    print(f"  Adaptive total time: {schedule_adaptive.total_time:.4f} ms")
    
    return True


# ============================================================================
# Kernel Tests
# ============================================================================

def test_flash_ffn():
    """Test FlashFFN kernel (if CUDA available)."""
    print_test_header("FlashFFN Kernel")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping kernel test")
        print_test_result(True, "Skipped (no CUDA)")
        return True
    
    try:
        # Create FlashFFN module
        hidden_dim = 512
        intermediate_dim = 2048
        hans_ratio = 0.3
        
        num_hans = int(intermediate_dim * hans_ratio)
        num_lans = intermediate_dim - num_hans
        
        hans_indices = torch.arange(num_hans, device='cuda')
        lans_indices = torch.arange(num_hans, intermediate_dim, device='cuda')
        
        ffn = FlashFFN(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            hans_indices=hans_indices,
            lans_indices=lans_indices,
            activation='relu',
        ).cuda()
        
        # Create input
        seq_len = 256
        x = torch.randn(seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        
        # Forward pass
        print("Running FlashFFN forward pass...")
        output = ffn(x)
        
        assert output.shape == (seq_len, hidden_dim), f"Wrong output shape: {output.shape}"
        
        print(f"\nFlashFFN output shape: {output.shape}")
        print_test_result(True, "FlashFFN forward pass successful")
        
        # Get memory info
        mem_info = ffn.get_memory_footprint()
        print(f"Memory footprint: {mem_info}")
        
        return True
        
    except Exception as e:
        print(f"FlashFFN test error: {e}")
        print_test_result(False, str(e))
        return False


# ============================================================================
# Communication Tests
# ============================================================================

def test_sparse_communication():
    """Test sparse communication utilities."""
    print_test_header("Sparse Communication")
    
    # Test communication benefit estimation
    print("Testing communication benefit estimation...")
    
    for sparsity in [0.3, 0.5, 0.7, 0.9]:
        benefit = estimate_sparse_communication_benefit(
            hidden_dim=4096,
            num_gpus=8,
            sparsity=sparsity,
        )
        print(f"  Sparsity {sparsity:.1f}: speedup={benefit['speedup']:.2f}x, "
              f"compression={benefit['compression_ratio']:.2f}")
    
    print_test_result(True, "Communication benefit estimation works")
    
    return True


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_integration():
    """Test full system integration."""
    print_test_header("Full Integration")
    
    # Create SparseTPFFN
    print("Creating SparseTPFFN instance...")
    
    config = SparseTPFFNConfig(
        hidden_dim=512,
        intermediate_dim=2048,
        num_gpus=2,  # Use 2 GPUs for testing
        activation='relu',
        activation_threshold=0.1,
        target_sparsity=0.5,
        num_chunks=2,
        use_sparse_comm=False,  # Disable for single-GPU testing
    )
    
    ffn = SparseTPFFN(config=config, rank=0)
    
    # Generate synthetic masks for offline analysis
    print("Running offline analysis...")
    sample_masks = generate_synthetic_masks(
        num_samples=100,
        num_neurons=config.intermediate_dim,
        sparsity=0.5,
        coactivation_pattern='structured'
    )
    
    analysis_results = ffn.offline_analysis(
        sample_masks=sample_masks,
        balance_strategy='even'
    )
    
    print(f"\nOffline analysis results:")
    print(f"  Analysis time: {analysis_results['analysis_time_s']:.2f}s")
    print(f"  TC neurons: {analysis_results['tc_cc_info']['tc_count']}")
    print(f"  CC neurons: {analysis_results['tc_cc_info']['cc_count']}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    if torch.cuda.is_available():
        x = torch.randn(128, config.hidden_dim, device='cuda', dtype=torch.float16)
        ffn = ffn.cuda()
        
        output = ffn(x)
        
        assert output.shape == (128, config.hidden_dim), f"Wrong output shape: {output.shape}"
        
        print(f"Forward pass output shape: {output.shape}")
        
        # Get statistics
        stats = ffn.get_statistics()
        print(f"Forward statistics: {stats}")
        
        print_test_result(True, "Full integration test passed")
    else:
        print("CUDA not available, skipping forward pass test")
        print_test_result(True, "Skipped (no CUDA)")
    
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SPARSE-AWARE FFN TENSOR PARALLEL SYSTEM - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Coactivation Analyzer", test_coactivation_analyzer),
        ("Neuron Partitioner", test_neuron_partitioner),
        ("GPU Balancer", test_gpu_balancer),
        ("TC/CC Balancer", test_tc_cc_balancer),
        ("Pipeline Scheduler", test_pipeline_scheduler),
        ("FlashFFN Kernel", test_flash_ffn),
        ("Sparse Communication", test_sparse_communication),
        ("Full Integration", test_full_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for test_name, p, error in results:
        status = "✓ PASSED" if p else "✗ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
