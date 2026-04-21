"""
Layer 2 Solver: TensorCore/CUDACore Balancer

This module optimizes the assignment of neurons to TensorCore (TC) and
CUDACore (CC) execution units within each GPU to minimize execution time.

Key insight: HANS neurons are computed on TC (dense, efficient),
while LANS neurons are computed on CC (sparse, efficient for low activation).
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import torch


class ExecutionUnit(Enum):
    TENSOR_CORE = "tensor_core"
    CUDA_CORE = "cuda_core"


@dataclass
class TCCCAssignment:
    """
    Assignment of neurons to execution units (TC/CC).
    
    Attributes:
        tc_indices: Neuron indices assigned to TensorCore
        cc_indices: Neuron indices assigned to CUDACore
        tc_time: Estimated TC execution time (ms)
        cc_time: Estimated CC execution time (ms)
        overlap_efficiency: Efficiency of TC/CC overlap
    """
    tc_indices: np.ndarray
    cc_indices: np.ndarray
    tc_time: float
    cc_time: float
    overlap_efficiency: float
    
    @property
    def total_time(self) -> float:
        """Total execution time considering overlap."""
        # With good overlap, total time ≈ max(tc_time, cc_time)
        return max(self.tc_time, self.cc_time)


class TCCCBalancer:
    """
    Balances neuron execution between TensorCore and CUDACore.
    
    Objective: minimize |T_TC - T_CC| to maximize overlap efficiency
    
    Strategy:
    - HANS neurons: typically go to TC (dense computation)
    - LANS neurons: typically go to CC (sparse computation)
    - But we can adjust the boundary for optimal balance
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        tc_tflops: float = 312.0,  # A100 TensorCore TFLOPS (FP16)
        cc_tflops: float = 19.5,   # A100 CUDACore TFLOPS (FP32)
        tc_efficiency: float = 0.8,
        cc_efficiency: float = 0.6,
        tc_memory_bw: float = 2039.0,  # GB/s
        cc_memory_bw: float = 2039.0,
    ):
        """
        Initialize TC/CC balancer.
        
        Args:
            hidden_dim: Model hidden dimension (H)
            intermediate_dim: FFN intermediate dimension (F)
            tc_tflops: TensorCore peak TFLOPS
            cc_tflops: CUDACore peak TFLOPS
            tc_efficiency: Typical TensorCore efficiency
            cc_efficiency: Typical CUDACore efficiency
            tc_memory_bw: TensorCore memory bandwidth (GB/s)
            cc_memory_bw: CUDACore memory bandwidth (GB/s)
        """
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.tc_tflops = tc_tflops
        self.cc_tflops = cc_tflops
        self.tc_efficiency = tc_efficiency
        self.cc_efficiency = cc_efficiency
        self.tc_memory_bw = tc_memory_bw
        self.cc_memory_bw = cc_memory_bw
        
        # Tile sizes for different execution units
        self.tc_tile_m = 16  # TensorCore tile size
        self.tc_tile_n = 16
        self.tc_tile_k = 16
        self.cc_tile_m = 32  # CUDACore tile size
        self.cc_tile_n = 32
    
    def balance(
        self,
        hans_indices: np.ndarray,
        lans_indices: np.ndarray,
        activation_frequencies: np.ndarray,
        sequence_length: int,
        search_method: str = 'heuristic'
    ) -> TCCCAssignment:
        """
        Balance neurons between TC and CC execution.
        
        Args:
            hans_indices: Indices of HANS neurons
            lans_indices: Indices of LANS neurons
            activation_frequencies: Activation frequency per neuron
            sequence_length: Number of tokens in sequence (S)
            search_method: 'heuristic', 'exhaustive', or 'binary'
            
        Returns:
            TCCCAssignment with optimal TC/CC split
        """
        if search_method == 'heuristic':
            return self._heuristic_balance(
                hans_indices, lans_indices, activation_frequencies, sequence_length
            )
        elif search_method == 'exhaustive':
            return self._exhaustive_balance(
                hans_indices, lans_indices, activation_frequencies, sequence_length
            )
        else:
            return self._binary_search_balance(
                hans_indices, lans_indices, activation_frequencies, sequence_length
            )
    
    def _heuristic_balance(
        self,
        hans_indices: np.ndarray,
        lans_indices: np.ndarray,
        activation_frequencies: np.ndarray,
        sequence_length: int,
    ) -> TCCCAssignment:
        """
        Heuristic approach: assign based on activation frequency threshold.
        """
        # Start with default assignment
        tc_indices = hans_indices.copy()
        cc_indices = lans_indices.copy()
        
        # Calculate initial times
        tc_time = self._estimate_tc_time(tc_indices, activation_frequencies, sequence_length)
        cc_time = self._estimate_cc_time(cc_indices, activation_frequencies, sequence_length)
        
        # Adjust boundary if imbalance
        max_iterations = 50
        for _ in range(max_iterations):
            time_diff = tc_time - cc_time
            
            if abs(time_diff) < 0.01 * max(tc_time, cc_time):
                break  # Balanced
            
            if time_diff > 0:
                # TC takes longer, move some neurons to CC
                # Find highest-frequency HANS neurons (best for CC)
                if len(tc_indices) == 0:
                    break
                
                hans_freqs = [(idx, activation_frequencies[idx]) for idx in tc_indices]
                hans_freqs.sort(key=lambda x: x[1], reverse=True)
                
                # Move top neurons
                move_count = max(1, len(tc_indices) // 20)
                move_indices = [idx for idx, _ in hans_freqs[:move_count]]
                
                tc_indices = np.array([idx for idx in tc_indices if idx not in move_indices])
                cc_indices = np.append(cc_indices, move_indices)
            else:
                # CC takes longer, move some neurons to TC
                if len(cc_indices) == 0:
                    break
                
                lans_freqs = [(idx, activation_frequencies[idx]) for idx in cc_indices]
                lans_freqs.sort(key=lambda x: x[1], reverse=True)
                
                # Move top neurons (highest frequency)
                move_count = max(1, len(cc_indices) // 20)
                move_indices = [idx for idx, _ in lans_freqs[:move_count]]
                
                cc_indices = np.array([idx for idx in cc_indices if idx not in move_indices])
                tc_indices = np.append(tc_indices, move_indices)
            
            # Recalculate times
            tc_time = self._estimate_tc_time(tc_indices, activation_frequencies, sequence_length)
            cc_time = self._estimate_cc_time(cc_indices, activation_frequencies, sequence_length)
        
        overlap_efficiency = self._calculate_overlap_efficiency(tc_time, cc_time)
        
        return TCCCAssignment(
            tc_indices=np.sort(tc_indices),
            cc_indices=np.sort(cc_indices),
            tc_time=tc_time,
            cc_time=cc_time,
            overlap_efficiency=overlap_efficiency
        )
    
    def _exhaustive_balance(
        self,
        hans_indices: np.ndarray,
        lans_indices: np.ndarray,
        activation_frequencies: np.ndarray,
        sequence_length: int,
    ) -> TCCCAssignment:
        """
        Exhaustive search for optimal TC/CC split.
        Only feasible for small numbers of neurons.
        """
        all_indices = np.concatenate([hans_indices, lans_indices])
        n = len(all_indices)
        
        if n > 20:
            # Fall back to heuristic for large n
            return self._heuristic_balance(
                hans_indices, lans_indices, activation_frequencies, sequence_length
            )
        
        best_assignment = None
        best_time = float('inf')
        
        # Try all possible splits
        for mask in range(1 << n):
            tc_indices = all_indices[[i for i in range(n) if mask & (1 << i)]]
            cc_indices = all_indices[[i for i in range(n) if not (mask & (1 << i))]]
            
            if len(tc_indices) == 0 or len(cc_indices) == 0:
                continue
            
            tc_time = self._estimate_tc_time(tc_indices, activation_frequencies, sequence_length)
            cc_time = self._estimate_cc_time(cc_indices, activation_frequencies, sequence_length)
            total_time = max(tc_time, cc_time)
            
            if total_time < best_time:
                best_time = total_time
                overlap_efficiency = self._calculate_overlap_efficiency(tc_time, cc_time)
                best_assignment = TCCCAssignment(
                    tc_indices=tc_indices,
                    cc_indices=cc_indices,
                    tc_time=tc_time,
                    cc_time=cc_time,
                    overlap_efficiency=overlap_efficiency
                )
        
        return best_assignment
    
    def _binary_search_balance(
        self,
        hans_indices: np.ndarray,
        lans_indices: np.ndarray,
        activation_frequencies: np.ndarray,
        sequence_length: int,
    ) -> TCCCAssignment:
        """
        Binary search on activation frequency threshold.
        """
        all_indices = np.concatenate([hans_indices, lans_indices])
        all_freqs = activation_frequencies[all_indices]
        
        # Sort by frequency
        sorted_pairs = sorted(zip(all_indices, all_freqs), key=lambda x: x[1], reverse=True)
        sorted_indices = np.array([idx for idx, _ in sorted_pairs])
        sorted_freqs = np.array([freq for _, freq in sorted_pairs])
        
        # Binary search for optimal split point
        left, right = 1, len(sorted_indices) - 1
        best_assignment = None
        best_time = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            
            tc_indices = sorted_indices[:mid]
            cc_indices = sorted_indices[mid:]
            
            tc_time = self._estimate_tc_time(tc_indices, activation_frequencies, sequence_length)
            cc_time = self._estimate_cc_time(cc_indices, activation_frequencies, sequence_length)
            total_time = max(tc_time, cc_time)
            
            if total_time < best_time:
                best_time = total_time
                overlap_efficiency = self._calculate_overlap_efficiency(tc_time, cc_time)
                best_assignment = TCCCAssignment(
                    tc_indices=tc_indices,
                    cc_indices=cc_indices,
                    tc_time=tc_time,
                    cc_time=cc_time,
                    overlap_efficiency=overlap_efficiency
                )
            
            # Adjust search direction
            if tc_time > cc_time:
                right = mid - 1  # Need fewer TC neurons
            else:
                left = mid + 1   # Need more TC neurons
        
        return best_assignment
    
    def _estimate_tc_time(
        self,
        indices: np.ndarray,
        activation_frequencies: np.ndarray,
        sequence_length: int,
    ) -> float:
        """
        Estimate TensorCore execution time.
        
        TC is efficient for dense computation with high activation rate.
        Time ≈ FLOPs / (TFLOPS * efficiency) + memory_overhead
        """
        if len(indices) == 0:
            return 0.0
        
        # Average activation rate for assigned neurons
        avg_activation = activation_frequencies[indices].mean()
        
        # FLOPs: 2 * S * H * F_neuron * activation_rate
        # First matrix multiply: S * H * F_neuron
        # Second matrix multiply: S * F_neuron * H
        flops = 2 * sequence_length * self.hidden_dim * len(indices) * avg_activation
        
        # Compute time
        compute_time = flops / (self.tc_tflops * 1e12 * self.tc_efficiency)
        
        # Memory access time
        # Read: X[S, H], W1[H, F], W2[F, H]
        # Write: Y[S, H]
        memory_bytes = (
            sequence_length * self.hidden_dim * 2 +  # X (fp16)
            self.hidden_dim * len(indices) * 2 * 2 +  # W1, W2 (fp16)
            sequence_length * len(indices) * 2 +  # Intermediate (fp16)
            sequence_length * self.hidden_dim * 2  # Output (fp16)
        )
        memory_time = memory_bytes / (self.tc_memory_bw * 1e9)
        
        # TC launch overhead
        launch_overhead = 0.001 * len(indices)  # ~1us per tile
        
        total_time = max(compute_time, memory_time) + launch_overhead
        
        return total_time * 1000  # Convert to ms
    
    def _estimate_cc_time(
        self,
        indices: np.ndarray,
        activation_frequencies: np.ndarray,
        sequence_length: int,
    ) -> float:
        """
        Estimate CUDACore execution time.
        
        CC is efficient for sparse computation with low activation rate.
        Uses sparse GEMV pattern.
        """
        if len(indices) == 0:
            return 0.0
        
        avg_activation = activation_frequencies[indices].mean()
        
        # Sparse computation: only compute active neurons
        active_neurons = int(len(indices) * avg_activation)
        
        # FLOPs for sparse computation
        flops = 2 * sequence_length * self.hidden_dim * active_neurons
        
        # Compute time (CC is slower but more efficient for sparse)
        effective_efficiency = self.cc_efficiency * (1 + 0.5 * (1 - avg_activation))  # Bonus for sparsity
        compute_time = flops / (self.cc_tflops * 1e9 * effective_efficiency)  # Note: GFLOPS for CC
        
        # Memory access (sparse pattern)
        memory_bytes = (
            sequence_length * self.hidden_dim * 2 +  # X (fp16)
            self.hidden_dim * active_neurons * 2 * 2 +  # W1, W2 for active neurons
            sequence_length * active_neurons * 2  # Intermediate
        )
        memory_time = memory_bytes / (self.cc_memory_bw * 1e9)
        
        # Sparse gather/scatter overhead
        gather_overhead = 0.0005 * active_neurons
        
        total_time = max(compute_time, memory_time) + gather_overhead
        
        return total_time * 1000  # Convert to ms
    
    def _calculate_overlap_efficiency(
        self,
        tc_time: float,
        cc_time: float,
    ) -> float:
        """
        Calculate how efficiently TC and CC can overlap.
        
        Perfect overlap: both finish at same time
        Poor overlap: one finishes much earlier
        """
        if tc_time == 0 or cc_time == 0:
            return 1.0
        
        # Overlap efficiency = min(tc, cc) / max(tc, cc)
        return min(tc_time, cc_time) / max(tc_time, cc_time)
    
    def get_optimal_tile_config(
        self,
        tc_neuron_count: int,
        cc_neuron_count: int,
        sequence_length: int,
    ) -> Dict:
        """
        Get optimal tile configuration for TC and CC execution.
        
        Returns:
            Dictionary with tile sizes and SM allocations
        """
        # Total SMs on A100 = 108
        total_sms = 108
        
        # Allocate SMs based on workload
        if tc_neuron_count == 0:
            tc_sms = 0
            cc_sms = total_sms
        elif cc_neuron_count == 0:
            tc_sms = total_sms
            cc_sms = 0
        else:
            # Proportional allocation based on neuron count
            total_neurons = tc_neuron_count + cc_neuron_count
            tc_sms = int(total_sms * tc_neuron_count / total_neurons)
            cc_sms = total_sms - tc_sms
        
        # Ensure minimum SMs
        if tc_sms > 0:
            tc_sms = max(tc_sms, 16)
        if cc_sms > 0:
            cc_sms = max(cc_sms, 16)
        
        return {
            'tc': {
                'num_sms': tc_sms,
                'tile_m': self.tc_tile_m,
                'tile_n': self.tc_tile_n,
                'tile_k': self.tc_tile_k,
            },
            'cc': {
                'num_sms': cc_sms,
                'tile_m': self.cc_tile_m,
                'tile_n': self.cc_tile_n,
            }
        }
