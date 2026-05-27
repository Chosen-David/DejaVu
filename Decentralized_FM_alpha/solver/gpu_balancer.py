"""
Layer 2 Solver: GPU Balancer

This module balances neuron assignments across GPUs to minimize
total execution time while considering:
1. Load balance (equal computation time per GPU)
2. Communication cost (minimize cross-GPU co-activation)
3. Memory balance (equal memory footprint per GPU)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import torch

from .neuron_partitioner import NeuronPartition


@dataclass
class GPUBalanceConfig:
    """
    Configuration for GPU balancing.
    
    Attributes:
        gpu_id: GPU identifier
        hans_count: Number of HANS neurons assigned
        lans_count: Number of LANS neurons assigned
        compute_time: Estimated compute time (ms)
        memory_usage: Estimated memory usage (MB)
        communication_cost: Estimated communication cost
    """
    gpu_id: int
    hans_count: int
    lans_count: int
    compute_time: float
    memory_usage: float
    communication_cost: float


class GPUBalancer:
    """
    Balances neuron distribution across GPUs for optimal TP performance.
    
    The balancer considers:
    1. Computation load: Each GPU should have similar compute time
    2. Memory balance: Each GPU should have similar memory usage
    3. Communication: Minimize cross-GPU co-activation communication
    """
    
    def __init__(
        self,
        num_gpus: int,
        hidden_dim: int,
        intermediate_dim: int,
        gpu_memory_gb: float = 80.0,
        gpu_compute_tflops: float = 312.0,  # A100 TFLOPS for FP16
        bandwidth_gbps: float = 600.0,  # NVLink bandwidth
    ):
        """
        Initialize GPU balancer.
        
        Args:
            num_gpus: Number of GPUs for tensor parallelism
            hidden_dim: Model hidden dimension (H)
            intermediate_dim: FFN intermediate dimension (F)
            gpu_memory_gb: GPU memory in GB
            gpu_compute_tflops: GPU compute capability in TFLOPS
            bandwidth_gbps: Inter-GPU bandwidth in GB/s
        """
        self.num_gpus = num_gpus
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gpu_memory_gb = gpu_memory_gb
        self.gpu_compute_tflops = gpu_compute_tflops
        self.bandwidth_gbps = bandwidth_gbps
        
        # Cost model parameters
        self.tc_efficiency = 0.8  # TensorCore efficiency
        self.cc_efficiency = 0.6  # CUDACore efficiency
        
    def balance_partitions(
        self,
        partitions: List[NeuronPartition],
        coactivation_weights: np.ndarray,
        activation_frequencies: np.ndarray,
        strategy: str = 'compute_aware'
    ) -> Tuple[List[NeuronPartition], Dict]:
        """
        Re-balance partitions across GPUs for optimal performance.
        
        Args:
            partitions: Initial neuron partitions per GPU
            coactivation_weights: Co-activation weight matrix [F, F]
            activation_frequencies: Activation frequency per neuron [F]
            strategy: Balancing strategy ('compute_aware', 'memory_aware', 'hybrid')
            
        Returns:
            Tuple of (balanced_partitions, balance_info)
        """
        if strategy == 'compute_aware':
            return self._balance_compute_aware(
                partitions, coactivation_weights, activation_frequencies
            )
        elif strategy == 'memory_aware':
            return self._balance_memory_aware(partitions)
        else:
            return self._balance_hybrid(
                partitions, coactivation_weights, activation_frequencies
            )
    
    def _balance_compute_aware(
        self,
        partitions: List[NeuronPartition],
        coactivation_weights: np.ndarray,
        activation_frequencies: np.ndarray,
    ) -> Tuple[List[NeuronPartition], Dict]:
        """
        Balance partitions to equalize compute time across GPUs.
        """
        # Estimate compute time for each GPU
        gpu_compute_times = self._estimate_compute_times(
            partitions, activation_frequencies
        )
        
        # Calculate target compute time (average)
        target_time = np.mean(gpu_compute_times)
        
        # Iteratively re-balance
        balanced_partitions = [NeuronPartition(
            hans_indices=p.hans_indices.copy(),
            lans_indices=p.lans_indices.copy(),
            coactivated_hans_groups=p.coactivated_hans_groups.copy()
        ) for p in partitions]
        
        # Identify overloaded and underloaded GPUs
        max_iterations = 100
        for iteration in range(max_iterations):
            compute_times = self._estimate_compute_times(
                balanced_partitions, activation_frequencies
            )
            
            # Check if balanced
            time_std = np.std(compute_times)
            if time_std < 0.01 * target_time:  # Within 1% of target
                break
            
            # Find most overloaded and underloaded GPUs
            max_gpu = np.argmax(compute_times)
            min_gpu = np.argmin(compute_times)
            
            if compute_times[max_gpu] - compute_times[min_gpu] < 0.05 * target_time:
                break
            
            # Move neurons from max_gpu to min_gpu
            self._transfer_neurons(
                balanced_partitions,
                max_gpu,
                min_gpu,
                coactivation_weights,
                activation_frequencies,
                'compute'
            )
        
        balance_info = {
            'strategy': 'compute_aware',
            'final_compute_times': compute_times.tolist(),
            'compute_time_std': float(np.std(compute_times)),
            'compute_time_mean': float(np.mean(compute_times)),
        }
        
        return balanced_partitions, balance_info
    
    def _balance_memory_aware(
        self,
        partitions: List[NeuronPartition],
    ) -> Tuple[List[NeuronPartition], Dict]:
        """
        Balance partitions to equalize memory usage across GPUs.
        """
        # Estimate memory usage for each GPU
        gpu_memory = self._estimate_memory_usage(partitions)
        target_memory = np.mean(gpu_memory)
        
        balanced_partitions = [NeuronPartition(
            hans_indices=p.hans_indices.copy(),
            lans_indices=p.lans_indices.copy(),
            coactivated_hans_groups=p.coactivated_hans_groups.copy()
        ) for p in partitions]
        
        # Iteratively re-balance
        max_iterations = 100
        for iteration in range(max_iterations):
            memory_usage = self._estimate_memory_usage(balanced_partitions)
            
            # Check if balanced
            memory_std = np.std(memory_usage)
            if memory_std < 0.01 * target_memory:
                break
            
            # Find most overloaded and underloaded GPUs
            max_gpu = np.argmax(memory_usage)
            min_gpu = np.argmin(memory_usage)
            
            if memory_usage[max_gpu] - memory_usage[min_gpu] < 0.05 * target_memory:
                break
            
            # Move neurons from max_gpu to min_gpu
            self._transfer_neurons_simple(balanced_partitions, max_gpu, min_gpu)
        
        balance_info = {
            'strategy': 'memory_aware',
            'final_memory_usage': memory_usage.tolist(),
            'memory_std': float(np.std(memory_usage)),
            'memory_mean': float(np.mean(memory_usage)),
        }
        
        return balanced_partitions, balance_info
    
    def _balance_hybrid(
        self,
        partitions: List[NeuronPartition],
        coactivation_weights: np.ndarray,
        activation_frequencies: np.ndarray,
    ) -> Tuple[List[NeuronPartition], Dict]:
        """
        Balance partitions considering both compute and memory.
        Uses weighted objective function.
        """
        compute_weight = 0.6
        memory_weight = 0.4
        
        balanced_partitions = [NeuronPartition(
            hans_indices=p.hans_indices.copy(),
            lans_indices=p.lans_indices.copy(),
            coactivated_hans_groups=p.coactivated_hans_groups.copy()
        ) for p in partitions]
        
        max_iterations = 100
        for iteration in range(max_iterations):
            compute_times = self._estimate_compute_times(
                balanced_partitions, activation_frequencies
            )
            memory_usage = self._estimate_memory_usage(balanced_partitions)
            
            # Normalize
            compute_norm = (np.array(compute_times) - np.min(compute_times)) / (np.max(compute_times) - np.min(compute_times) + 1e-6)
            memory_norm = (np.array(memory_usage) - np.min(memory_usage)) / (np.max(memory_usage) - np.min(memory_usage) + 1e-6)
            
            # Combined score
            scores = compute_weight * compute_norm + memory_weight * memory_norm
            
            # Check balance
            if np.std(scores) < 0.05:
                break
            
            # Find GPUs to transfer
            max_gpu = np.argmax(scores)
            min_gpu = np.argmin(scores)
            
            self._transfer_neurons(
                balanced_partitions,
                max_gpu,
                min_gpu,
                coactivation_weights,
                activation_frequencies,
                'hybrid'
            )
        
        balance_info = {
            'strategy': 'hybrid',
            'final_compute_times': compute_times.tolist(),
            'final_memory_usage': memory_usage.tolist(),
            'compute_time_std': float(np.std(compute_times)),
            'memory_std': float(np.std(memory_usage)),
        }
        
        return balanced_partitions, balance_info
    
    def _estimate_compute_times(
        self,
        partitions: List[NeuronPartition],
        activation_frequencies: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate compute time for each GPU.
        
        Time = (HANS_time + LANS_time) based on neuron counts and activation rates
        """
        compute_times = np.zeros(self.num_gpus)
        
        for i, partition in enumerate(partitions):
            hans_indices = np.asarray(partition.hans_indices, dtype=np.int64)
            lans_indices = np.asarray(partition.lans_indices, dtype=np.int64)

            # HANS compute time (TensorCore)
            hans_neurons = len(hans_indices)
            hans_activation_rate = activation_frequencies[hans_indices].mean() if hans_neurons > 0 else 0
            
            # Dense computation on TensorCore
            hans_flops = 2 * self.hidden_dim * hans_neurons * hans_activation_rate
            hans_time = hans_flops / (self.gpu_compute_tflops * 1e12 * self.tc_efficiency)
            
            # LANS compute time (CUDACore)
            lans_neurons = len(lans_indices)
            lans_activation_rate = activation_frequencies[lans_indices].mean() if lans_neurons > 0 else 0
            
            # Sparse computation on CUDACore
            lans_flops = 2 * self.hidden_dim * lans_neurons * lans_activation_rate
            lans_time = lans_flops / (self.gpu_compute_tflops * 1e12 * self.cc_efficiency)
            
            compute_times[i] = (hans_time + lans_time) * 1000  # Convert to ms
        
        return compute_times
    
    def _estimate_memory_usage(
        self,
        partitions: List[NeuronPartition],
    ) -> np.ndarray:
        """
        Estimate memory usage for each GPU.
        
        Memory = weights for assigned neurons
        """
        memory_usage = np.zeros(self.num_gpus)
        
        # Weight memory per neuron: 2 * H * sizeof(float16)
        bytes_per_neuron = 2 * self.hidden_dim * 2  # 2 matrices, fp16
        
        for i, partition in enumerate(partitions):
            total_neurons = partition.total_neurons
            memory_usage[i] = (total_neurons * bytes_per_neuron) / (1024 ** 2)  # MB
        
        return memory_usage
    
    def _transfer_neurons(
        self,
        partitions: List[NeuronPartition],
        from_gpu: int,
        to_gpu: int,
        coactivation_weights: np.ndarray,
        activation_frequencies: np.ndarray,
        mode: str,
    ) -> None:
        """
        Transfer neurons from one GPU to another.
        Selects neurons that minimize cost increase.
        """
        # Calculate transfer cost for each neuron
        from_partition = partitions[from_gpu]
        
        # Try to transfer LANS first (lower impact on co-activation)
        if len(from_partition.lans_indices) > 0:
            # Select LANS with lowest co-activation with HANS on from_gpu
            lans_costs = []
            for idx in from_partition.lans_indices:
                coact_cost = sum(
                    coactivation_weights[idx, h_idx]
                    for h_idx in from_partition.hans_indices
                )
                lans_costs.append((idx, coact_cost))
            
            # Sort by cost (ascending)
            lans_costs.sort(key=lambda x: x[1])
            
            # Transfer the neuron with lowest cost
            if len(lans_costs) > 0:
                transfer_idx = lans_costs[0][0]
                
                # Remove from from_gpu
                mask = from_partition.lans_indices != transfer_idx
                from_partition.lans_indices = from_partition.lans_indices[mask]
                
                # Add to to_gpu
                partitions[to_gpu].lans_indices = np.append(
                    partitions[to_gpu].lans_indices, transfer_idx
                )
                return
        
        # If no LANS to transfer, try HANS
        if len(from_partition.coactivated_hans_groups) > 0:
            # Transfer entire co-activation group if it improves balance
            # Find smallest group
            group_sizes = [len(g) for g in from_partition.coactivated_hans_groups]
            min_group_idx = np.argmin(group_sizes)
            
            transfer_group = from_partition.coactivated_hans_groups[min_group_idx]
            
            # Remove from from_gpu
            from_partition.coactivated_hans_groups.pop(min_group_idx)
            hans_set = set(from_partition.hans_indices)
            for idx in transfer_group:
                hans_set.discard(idx)
            from_partition.hans_indices = np.array(sorted(hans_set))
            
            # Add to to_gpu
            partitions[to_gpu].coactivated_hans_groups.append(transfer_group)
            partitions[to_gpu].hans_indices = np.append(
                partitions[to_gpu].hans_indices, transfer_group
            )
    
    def _transfer_neurons_simple(
        self,
        partitions: List[NeuronPartition],
        from_gpu: int,
        to_gpu: int,
    ) -> None:
        """Simple transfer without considering co-activation."""
        from_partition = partitions[from_gpu]
        
        # Transfer one LANS neuron
        if len(from_partition.lans_indices) > 0:
            transfer_idx = from_partition.lans_indices[0]
            from_partition.lans_indices = from_partition.lans_indices[1:]
            partitions[to_gpu].lans_indices = np.append(
                partitions[to_gpu].lans_indices, transfer_idx
            )
        # Or one HANS neuron
        elif len(from_partition.hans_indices) > 0:
            transfer_idx = from_partition.hans_indices[0]
            mask = from_partition.hans_indices != transfer_idx
            from_partition.hans_indices = from_partition.hans_indices[mask]
            partitions[to_gpu].hans_indices = np.append(
                partitions[to_gpu].hans_indices, transfer_idx
            )
    
    def get_balance_report(
        self,
        partitions: List[NeuronPartition],
        activation_frequencies: np.ndarray,
    ) -> Dict:
        """
        Generate detailed balance report.
        """
        compute_times = self._estimate_compute_times(partitions, activation_frequencies)
        memory_usage = self._estimate_memory_usage(partitions)
        
        report = {
            'summary': {
                'compute_time_mean_ms': float(np.mean(compute_times)),
                'compute_time_std_ms': float(np.std(compute_times)),
                'compute_time_max_ms': float(np.max(compute_times)),
                'compute_time_min_ms': float(np.min(compute_times)),
                'memory_mean_mb': float(np.mean(memory_usage)),
                'memory_std_mb': float(np.std(memory_usage)),
            },
            'per_gpu': []
        }
        
        for i, partition in enumerate(partitions):
            report['per_gpu'].append({
                'gpu_id': i,
                'hans_count': len(partition.hans_indices),
                'lans_count': len(partition.lans_indices),
                'compute_time_ms': float(compute_times[i]),
                'memory_mb': float(memory_usage[i]),
            })
        
        return report
