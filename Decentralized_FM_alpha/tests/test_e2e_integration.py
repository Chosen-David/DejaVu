"""
End-to-End Integration Test with DejaVu

This script tests the complete integration of sparse-aware FFN TP
with the DejaVu framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import time
from typing import Optional


def setup_distributed(rank: int, world_size: int):
    """Setup distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


def generate_synthetic_data(
    num_samples: int,
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    sparsity: float = 0.5,
    device: str = 'cuda',
):
    """Generate synthetic input data and masks."""
    # Input
    x = torch.randn(num_samples, seq_len, hidden_dim, device=device, dtype=torch.float16)
    
    # Mask (simulating predictor output)
    mask = (torch.rand(num_samples, seq_len, intermediate_dim, device=device) > sparsity).float()
    
    return x, mask


def test_basic_functionality(rank: int, world_size: int):
    """Test basic functionality of SparseTPMLP."""
    print(f"[Rank {rank}] Testing basic functionality...")
    
    from integration import ParallelSparseTPMLP
    
    # Configuration
    hidden_dim = 512
    intermediate_dim = 2048
    layer_idx = 0
    batch_size = 2
    seq_len = 128
    
    # Create process group
    process_group = dist.new_group(list(range(world_size)))
    
    # Create model
    model = ParallelSparseTPMLP(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        layer_idx=layer_idx,
        process_group=process_group,
        model_name="test_model",
        activation='gelu',
        store_root="/tmp/test_analysis_results",
    ).cuda()
    
    # Generate sample masks for offline analysis
    sample_masks = np.random.random((100, intermediate_dim)) > 0.5
    
    # Run offline analysis
    analysis_results = model.offline_analysis(sample_masks)
    print(f"[Rank {rank}] Analysis results: {analysis_results}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
    mask = torch.rand(batch_size, seq_len, intermediate_dim // world_size, device='cuda') > 0.5
    mask = mask.float()
    
    output = model(x, predictor_output=mask)
    
    assert output.shape == (batch_size, seq_len, hidden_dim), f"Wrong output shape: {output.shape}"
    
    print(f"[Rank {rank}] ✓ Basic functionality test passed")
    print(f"[Rank {rank}]   Input shape: {x.shape}")
    print(f"[Rank {rank}]   Output shape: {output.shape}")


def test_performance(rank: int, world_size: int):
    """Test performance with different configurations."""
    print(f"[Rank {rank}] Testing performance...")
    
    from integration import ParallelSparseTPMLP
    
    # Configuration
    hidden_dim = 1024
    intermediate_dim = 4096
    layer_idx = 0
    
    process_group = dist.new_group(list(range(world_size)))
    
    # Test different batch sizes and sequence lengths
    configs = [
        (1, 512),
        (2, 512),
        (1, 1024),
        (4, 256),
    ]
    
    model = ParallelSparseTPMLP(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        layer_idx=layer_idx,
        process_group=process_group,
        model_name="perf_test",
        activation='gelu',
        store_root="/tmp/perf_test_results",
    ).cuda()
    
    # Run offline analysis
    sample_masks = np.random.random((200, intermediate_dim)) > 0.5
    model.offline_analysis(sample_masks)
    
    results = []
    
    for batch_size, seq_len in configs:
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        mask = torch.rand(batch_size, seq_len, intermediate_dim // world_size, device='cuda') > 0.5
        mask = mask.float()
        
        # Warmup
        for _ in range(5):
            _ = model(x, predictor_output=mask)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        num_iterations = 20
        for _ in range(num_iterations):
            _ = model(x, predictor_output=mask)
        
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iterations * 1000
        
        results.append({
            'batch_size': batch_size,
            'seq_len': seq_len,
            'time_ms': elapsed,
        })
        
        print(f"[Rank {rank}] Config (B={batch_size}, S={seq_len}): {elapsed:.3f} ms")
    
    print(f"[Rank {rank}] ✓ Performance test completed")
    
    return results


def test_analysis_storage(rank: int, world_size: int):
    """Test analysis result storage and loading."""
    print(f"[Rank {rank}] Testing analysis storage...")
    
    from solver import AnalysisResultStore
    
    store = AnalysisResultStore("/tmp/storage_test")
    
    # Test saving
    from solver import AnalysisResult, PartitionResult, TCCCAssignmentResult
    
    result = AnalysisResult(
        layer_id=0,
        model_name="storage_test_model",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        config_hash="test_hash_123",
        hidden_dim=512,
        intermediate_dim=2048,
        num_gpus=world_size,
        partition_results=[
            PartitionResult(
                gpu_id=i,
                hans_indices=list(range(100)),
                lans_indices=list(range(100, 200)),
                hans_count=100,
                lans_count=100,
            )
            for i in range(world_size)
        ],
        tc_cc_assignments=[
            TCCCAssignmentResult(
                gpu_id=i,
                tc_indices=list(range(50)),
                cc_indices=list(range(50, 100)),
                tc_time_ms=1.0,
                cc_time_ms=1.0,
                overlap_efficiency=0.9,
            )
            for i in range(world_size)
        ],
        total_hans=100 * world_size,
        total_lans=100 * world_size,
        mean_activation_frequency=0.3,
        analysis_time_s=5.0,
    )
    
    # Save
    cw = np.random.random((2048, 2048))
    freq = np.random.random(2048)
    
    path = store.save_result(result, cw, freq)
    print(f"[Rank {rank}] Saved result to: {path}")
    
    # Load
    loaded = store.load_result("storage_test_model", 0, load_matrices=True)
    
    assert loaded is not None, "Failed to load result"
    assert loaded.layer_id == 0, "Wrong layer_id"
    assert len(loaded.partition_results) == world_size, "Wrong number of partitions"
    
    print(f"[Rank {rank}] ✓ Analysis storage test passed")


def main(rank: int, world_size: int):
    """Main test function."""
    try:
        setup_distributed(rank, world_size)
        
        print(f"\n{'='*80}")
        print(f"End-to-End Integration Test - Rank {rank}")
        print(f"{'='*80}\n")
        
        # Run tests
        test_basic_functionality(rank, world_size)
        test_performance(rank, world_size)
        test_analysis_storage(rank, world_size)
        
        print(f"\n[Rank {rank}] {'='*80}")
        print(f"[Rank {rank}] All tests passed!")
        print(f"[Rank {rank}] {'='*80}\n")
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    # Single GPU test
    if not torch.cuda.is_available():
        print("CUDA not available, running CPU-only tests")
        # Run CPU tests
        from tests.test_sparse_tp_ffn import run_all_tests
        run_all_tests()
    else:
        # Get world size from environment or use 1
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        
        if world_size > 1:
            # Multi-GPU test - spawn processes
            import torch.multiprocessing as mp
            mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
        else:
            # Single GPU test
            main(0, 1)
