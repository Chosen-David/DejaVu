"""
Sparse-Aware FFN Tensor Parallel System

This module integrates all components for end-to-end sparse FFN execution:
1. Neuron partitioning (HANS/LANS)
2. GPU balancing
3. TC/CC balancing
4. Pipeline scheduling
5. FlashFFN kernel
6. Sparse communication

This provides a complete implementation of the design proposed in the paper.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from dataclasses import dataclass
import time

# Import solver components
from ..solver import (
    CoactivationAnalyzer,
    NeuronPartitioner,
    GPUBalancer,
    TCCCBalancer,
    PipelineScheduler,
)
from ..solver.neuron_partitioner import NeuronPartition

# Import kernel components
from ..kernels import FlashFFN

# Import communication components
from ..communication import create_sparse_communicator, AdaptiveSparseAllReducer


@dataclass
class SparseTPFFNConfig:
    """
    Configuration for Sparse-Aware TP FFN.
    
    Attributes:
        hidden_dim: Model hidden dimension (H)
        intermediate_dim: FFN intermediate dimension (F)
        num_gpus: Number of GPUs for tensor parallelism
        activation: Activation function ('relu', 'gelu', 'silu')
        activation_threshold: Threshold for HANS/LANS classification
        target_sparsity: Target sparsity level
        num_chunks: Number of chunks for pipeline scheduling
    """
    hidden_dim: int = 4096
    intermediate_dim: int = 16384
    num_gpus: int = 8
    activation: str = 'relu'
    activation_threshold: float = 0.1
    target_sparsity: float = 0.5
    num_chunks: int = 2
    use_sparse_comm: bool = True
    overlap_compute_comm: bool = True


class SparseTPFFN(nn.Module):
    """
    Sparse-Aware Tensor Parallel FFN Layer.
    
    This is the main entry point that combines all optimizations:
    1. Offline analysis and partitioning
    2. Runtime sparse computation
    3. Pipeline scheduling with overlap
    4. Sparse communication
    """
    
    def __init__(
        self,
        config: SparseTPFFNConfig,
        rank: int = 0,
        predictor: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.config = config
        self.rank = rank
        self.predictor = predictor
        
        # Initialize solvers
        self.neuron_partitioner = NeuronPartitioner(
            num_neurons=config.intermediate_dim,
            num_gpus=config.num_gpus,
            activation_threshold=config.activation_threshold,
        )
        
        self.gpu_balancer = GPUBalancer(
            num_gpus=config.num_gpus,
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )
        
        self.tc_cc_balancer = TCCCBalancer(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )
        
        self.pipeline_scheduler = PipelineScheduler(
            num_gpus=config.num_gpus,
            hidden_dim=config.hidden_dim,
        )
        
        # Initialize communication
        if config.use_sparse_comm:
            self.communicator = create_sparse_communicator(
                hidden_dim=config.hidden_dim,
                num_gpus=config.num_gpus,
                rank=rank,
                mode='adaptive',
            )
        else:
            self.communicator = None
        
        # Weights (will be partitioned during offline analysis)
        self.w1 = nn.Parameter(torch.empty(config.hidden_dim, config.intermediate_dim))
        self.w2 = nn.Parameter(torch.empty(config.intermediate_dim, config.hidden_dim))
        
        # Partition info (populated after offline analysis)
        self.partition: Optional[NeuronPartition] = None
        self.tc_indices: Optional[torch.Tensor] = None
        self.cc_indices: Optional[torch.Tensor] = None
        
        # FlashFFN kernel for this partition
        self.flash_ffn: Optional[FlashFFN] = None
        
        # Statistics
        self.stats = {
            'forward_calls': 0,
            'total_time_ms': 0.0,
            'compute_time_ms': 0.0,
            'comm_time_ms': 0.0,
        }
    
    def offline_analysis(
        self,
        sample_masks: np.ndarray,
        balance_strategy: str = 'hybrid',
    ) -> Dict:
        """
        Perform offline analysis and partitioning.
        
        This should be called once after initialization with representative
        activation masks from the predictor.
        
        Args:
            sample_masks: Sample activation masks [num_samples, intermediate_dim]
            balance_strategy: Balancing strategy ('compute_aware', 'memory_aware', 'hybrid')
            
        Returns:
            Analysis and partitioning results
        """
        print("Starting offline analysis...")
        start_time = time.time()
        
        # Step 1: Partition neurons into HANS/LANS
        print("Step 1: Partitioning neurons (HANS/LANS)...")
        partitions = self.neuron_partitioner.analyze_and_partition(
            masks=sample_masks,
            method='spectral',
            balance_strategy=balance_strategy,
        )
        
        # Get partition for this GPU
        self.partition = partitions[self.rank]
        
        # Step 2: Balance across GPUs
        print("Step 2: Balancing across GPUs...")
        analyzer = self.neuron_partitioner.analyzer
        coactivation_weights = analyzer.compute_coactivation_weights()
        activation_frequencies = analyzer.compute_activation_frequencies()
        
        balanced_partitions, balance_info = self.gpu_balancer.balance_partitions(
            partitions=partitions,
            coactivation_weights=coactivation_weights,
            activation_frequencies=activation_frequencies,
            strategy=balance_strategy,
        )
        
        self.partition = balanced_partitions[self.rank]
        
        # Step 3: Balance TC/CC within GPU
        print("Step 3: Balancing TC/CC within GPU...")
        tc_cc_assignment = self.tc_cc_balancer.balance(
            hans_indices=self.partition.hans_indices,
            lans_indices=self.partition.lans_indices,
            activation_frequencies=activation_frequencies,
            sequence_length=2048,  # Representative sequence length
            search_method='heuristic',
        )
        
        self.tc_indices = torch.from_numpy(tc_cc_assignment.tc_indices)
        self.cc_indices = torch.from_numpy(tc_cc_assignment.cc_indices)
        
        # Step 4: Reorder weights
        print("Step 4: Reordering weights...")
        self._reorder_weights()
        
        # Step 5: Initialize FlashFFN kernel
        print("Step 5: Initializing FlashFFN kernel...")
        self.flash_ffn = FlashFFN(
            hidden_dim=self.config.hidden_dim,
            intermediate_dim=len(self.tc_indices) + len(self.cc_indices),
            hans_indices=self.tc_indices,
            lans_indices=self.cc_indices,
            activation=self.config.activation,
        )
        
        elapsed_time = time.time() - start_time
        print(f"Offline analysis completed in {elapsed_time:.2f}s")
        
        # Return analysis results
        return {
            'partition_info': self.neuron_partitioner.get_partition_info(),
            'balance_info': balance_info,
            'tc_cc_info': {
                'tc_count': len(self.tc_indices),
                'cc_count': len(self.cc_indices),
                'tc_time_ms': tc_cc_assignment.tc_time,
                'cc_time_ms': tc_cc_assignment.cc_time,
                'overlap_efficiency': tc_cc_assignment.overlap_efficiency,
            },
            'analysis_time_s': elapsed_time,
        }
    
    def _reorder_weights(self):
        """Reorder weights to match TC/CC assignment."""
        # Combine TC and CC indices
        new_order = torch.cat([self.tc_indices, self.cc_indices])
        
        # Reorder W1 columns (neurons)
        self.w1.data = self.w1.data[:, new_order]
        
        # Reorder W2 rows (neurons)
        self.w2.data = self.w2.data[new_order, :]
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sparse optimization.
        
        Args:
            x: Input tensor [seq_len, hidden_dim]
            mask: Activation mask [seq_len, intermediate_dim] (optional)
            
        Returns:
            Output tensor [seq_len, hidden_dim]
        """
        self.stats['forward_calls'] += 1
        start_time = time.perf_counter()
        
        # Get prediction mask if predictor is provided
        if mask is None and self.predictor is not None:
            with torch.no_grad():
                mask = self.predictor(x)
        
        # Create pipeline schedule
        seq_len = x.shape[0]
        schedule = self.pipeline_scheduler.schedule(
            sequence_length=seq_len,
            activation_sparsity=self.config.target_sparsity,
            compute_time_per_token=0.001,  # Estimate
            strategy='two_chunk' if self.config.num_chunks == 2 else 'adaptive',
        )
        
        # Process chunks
        outputs = []
        
        for chunk in schedule.chunks:
            chunk_start = time.perf_counter()
            
            # Extract chunk
            chunk_x = x[chunk.start_pos:chunk.end_pos]
            chunk_mask = mask[chunk.start_pos:chunk.end_pos] if mask is not None else None
            
            # Compute using FlashFFN
            chunk_output = self._compute_chunk(chunk_x, chunk_mask)
            
            # Communication (if enabled and distributed)
            if self.communicator is not None and torch.distributed.is_initialized():
                chunk_output, _ = self.communicator.adaptive_allreduce(
                    chunk_output,
                    chunk_mask,
                    async_op=False,
                )
            
            outputs.append(chunk_output)
            
            chunk_time = (time.perf_counter() - chunk_start) * 1000
            self.stats['compute_time_ms'] += chunk_time
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=0)
        
        # Update statistics
        total_time = (time.perf_counter() - start_time) * 1000
        self.stats['total_time_ms'] += total_time
        
        return output
    
    def _compute_chunk(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute FFN for a chunk."""
        if self.flash_ffn is not None:
            return self.flash_ffn(x, mask)
        else:
            # Fallback to standard computation
            return self._standard_forward(x)
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard FFN forward pass (fallback)."""
        intermediate = torch.matmul(x, self.w1)
        
        if self.config.activation == 'relu':
            intermediate = torch.relu(intermediate)
        elif self.config.activation == 'gelu':
            intermediate = torch.nn.functional.gelu(intermediate)
        elif self.config.activation == 'silu':
            intermediate = torch.nn.functional.silu(intermediate)
        
        output = torch.matmul(intermediate, self.w2)
        return output
    
    def get_statistics(self) -> Dict:
        """Get runtime statistics."""
        stats = self.stats.copy()
        
        if stats['forward_calls'] > 0:
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['forward_calls']
            stats['avg_compute_time_ms'] = stats['compute_time_ms'] / stats['forward_calls']
        
        if self.communicator is not None and hasattr(self.communicator, 'get_adaptation_stats'):
            stats['comm_stats'] = self.communicator.get_adaptation_stats()
        
        return stats
    
    def get_memory_info(self) -> Dict:
        """Get memory usage information."""
        info = {
            'weight_size_mb': (self.w1.numel() + self.w2.numel()) * 2 / (1024 ** 2),
            'partition': {
                'tc_neurons': len(self.tc_indices) if self.tc_indices is not None else 0,
                'cc_neurons': len(self.cc_indices) if self.cc_indices is not None else 0,
            }
        }
        
        if self.flash_ffn is not None:
            info['flash_ffn'] = self.flash_ffn.get_memory_footprint()
        
        return info


def create_sparse_tp_ffn(
    hidden_dim: int = 4096,
    intermediate_dim: int = 16384,
    num_gpus: int = 8,
    rank: int = 0,
    **kwargs
) -> SparseTPFFN:
    """
    Factory function to create SparseTPFFN instance.
    
    Args:
        hidden_dim: Model hidden dimension
        intermediate_dim: FFN intermediate dimension
        num_gpus: Number of GPUs
        rank: Current GPU rank
        **kwargs: Additional configuration options
        
    Returns:
        Configured SparseTPFFN instance
    """
    config = SparseTPFFNConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_gpus=num_gpus,
        **kwargs
    )
    
    return SparseTPFFN(config=config, rank=rank)
