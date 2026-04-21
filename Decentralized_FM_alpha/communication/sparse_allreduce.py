"""
Sparse AllReduce Communication

This module implements sparse-aware AllReduce operations that only
communicate active neurons based on prediction masks, significantly
reducing communication volume for sparse FFN layers.

Key features:
1. Sparse AllReduce: Only transmit active values and indices
2. Custom AllGather for sparse results
3. Overlap-friendly design for pipeline scheduling
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple, List, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class SparseTensor:
    """
    Sparse tensor representation for communication.
    
    Attributes:
        values: Dense values [num_active]
        indices: Indices of active elements [num_active]
        original_shape: Original dense shape
    """
    values: torch.Tensor
    indices: torch.Tensor
    original_shape: Tuple[int, ...]
    
    def to_dense(self) -> torch.Tensor:
        """Convert to dense tensor."""
        dense = torch.zeros(self.original_shape, device=self.values.device, dtype=self.values.dtype)
        if len(self.original_shape) == 1:
            dense[self.indices] = self.values
        elif len(self.original_shape) == 2:
            # 2D sparse tensor
            rows = self.indices[:, 0]
            cols = self.indices[:, 1]
            dense[rows, cols] = self.values
        return dense
    
    @classmethod
    def from_dense(cls, tensor: torch.Tensor, threshold: float = 0.0) -> 'SparseTensor':
        """Create sparse tensor from dense tensor."""
        mask = tensor.abs() > threshold
        indices = torch.where(mask)
        values = tensor[mask]
        
        if len(indices) == 1:
            indices = indices[0]
        else:
            indices = torch.stack(indices, dim=1)
        
        return cls(values=values, indices=indices, original_shape=tensor.shape)


class SparseAllReducer:
    """
    Implements sparse AllReduce for FFN outputs.
    
    Traditional AllReduce: All-to-all communication of full tensors
    Sparse AllReduce: Only communicate active elements
    
    Communication pattern:
    1. Each GPU identifies active outputs based on local mask
    2. AllGather active indices to build global active set
    3. AllReduce only the active values
    4. Scatter results back based on original indices
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_gpus: int,
        rank: int,
        backend: str = 'nccl',
        compression_ratio: float = 0.3,  # Expected sparsity
    ):
        """
        Initialize sparse AllReducer.
        
        Args:
            hidden_dim: Model hidden dimension
            num_gpus: Number of GPUs in TP group
            rank: Current GPU rank
            backend: Communication backend ('nccl' or 'gloo')
            compression_ratio: Expected compression ratio from sparsity
        """
        self.hidden_dim = hidden_dim
        self.num_gpus = num_gpus
        self.rank = rank
        self.backend = backend
        self.compression_ratio = compression_ratio
        
        # Buffers for communication
        self.max_active_per_gpu = int(hidden_dim * compression_ratio * 1.5)
        
        # Buffers for AllGather indices
        self.index_buffer = None
        self.value_buffer = None
        self.count_buffer = None
        
        # Initialize communication group if not already done
        self._init_process_group()
    
    def _init_process_group(self):
        """Initialize distributed process group."""
        if not dist.is_initialized():
            # For standalone testing, create a simple group
            # In production, this should be initialized by the framework
            pass
    
    def sparse_allreduce(
        self,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        async_op: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Perform sparse AllReduce on a tensor.
        
        Args:
            tensor: Input tensor to reduce [seq_len, hidden_dim]
            mask: Activation mask [seq_len, hidden_dim] (optional)
            async_op: Whether to perform asynchronous operation
            
        Returns:
            Tuple of (reduced_tensor, work_handle)
        """
        if mask is None:
            # Fall back to dense AllReduce if no mask
            return self._dense_allreduce(tensor, async_op)
        
        # Extract sparse representation
        sparse_tensor = self._extract_sparse(tensor, mask)
        
        # AllReduce sparse values
        work = self._sparse_allreduce_impl(sparse_tensor, async_op)
        
        # Reconstruct dense tensor
        result = sparse_tensor.to_dense()
        
        return result, work
    
    def _extract_sparse(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> SparseTensor:
        """Extract sparse representation based on mask."""
        # Sum mask across sequence dimension to get per-neuron activation
        neuron_mask = mask.any(dim=0)  # [hidden_dim]
        
        # Get active indices
        active_indices = torch.where(neuron_mask)[0]
        
        # Get values for active neurons (sum across sequence)
        # For FFN output, we need to reduce across GPUs
        active_values = tensor[:, active_indices].sum(dim=0)
        
        return SparseTensor(
            values=active_values,
            indices=active_indices,
            original_shape=(self.hidden_dim,)
        )
    
    def _sparse_allreduce_impl(
        self,
        sparse_tensor: SparseTensor,
        async_op: bool,
    ) -> Optional[dist.Work]:
        """
        Implementation of sparse AllReduce.
        
        Uses a two-phase approach:
        1. AllGather indices to determine global active set
        2. AllReduce values for global active set
        """
        num_active = len(sparse_tensor.indices)
        
        # Phase 1: AllGather active counts
        if self.count_buffer is None:
            self.count_buffer = torch.zeros(self.num_gpus, dtype=torch.long, device=sparse_tensor.values.device)
        
        self.count_buffer[self.rank] = num_active
        work = dist.all_gather_into_tensor(
            self.count_buffer,
            torch.tensor([num_active], device=sparse_tensor.values.device),
            async_op=async_op
        )
        
        if async_op:
            return work
        
        # Phase 2: AllGather indices
        total_active = self.count_buffer.sum().item()
        max_active_per_gpu = self.count_buffer.max().item()
        
        if self.index_buffer is None or self.index_buffer.size(0) < max_active_per_gpu * self.num_gpus:
            self.index_buffer = torch.zeros(
                max_active_per_gpu * self.num_gpus,
                dtype=torch.long,
                device=sparse_tensor.values.device
            )
        
        # Pad indices to max length
        padded_indices = torch.zeros(max_active_per_gpu, dtype=torch.long, device=sparse_tensor.values.device)
        padded_indices[:num_active] = sparse_tensor.indices
        
        # AllGather indices
        gathered_indices = [torch.zeros(max_active_per_gpu, dtype=torch.long, device=sparse_tensor.values.device)
                          for _ in range(self.num_gpus)]
        dist.all_gather(gathered_indices, padded_indices)
        
        # Build global active set
        global_active_set = set()
        for i, indices in enumerate(gathered_indices):
            count = self.count_buffer[i].item()
            global_active_set.update(indices[:count].tolist())
        
        global_active_indices = torch.tensor(sorted(global_active_set), device=sparse_tensor.values.device)
        
        # Phase 3: AllReduce values for global active indices
        # Map local indices to global indices
        local_to_global_map = {}
        for local_idx in sparse_tensor.indices.tolist():
            global_pos = (global_active_indices == local_idx).nonzero(as_tuple=True)[0]
            if len(global_pos) > 0:
                local_to_global_map[local_idx] = global_pos[0].item()
        
        # Create value buffer for global active set
        global_values = torch.zeros(len(global_active_indices), dtype=sparse_tensor.values.dtype, device=sparse_tensor.values.device)
        for i, local_idx in enumerate(sparse_tensor.indices):
            if local_idx.item() in local_to_global_map:
                global_pos = local_to_global_map[local_idx.item()]
                global_values[global_pos] = sparse_tensor.values[i]
        
        # AllReduce
        dist.all_reduce(global_values, op=dist.ReduceOp.SUM)
        
        # Update sparse tensor with global values
        sparse_tensor.values = global_values
        sparse_tensor.indices = global_active_indices
        
        return None
    
    def _dense_allreduce(
        self,
        tensor: torch.Tensor,
        async_op: bool,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """Fallback to dense AllReduce."""
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)
        return tensor, work
    
    def get_communication_stats(self) -> Dict:
        """Get statistics about communication."""
        return {
            'max_active_per_gpu': self.max_active_per_gpu,
            'compression_ratio': self.compression_ratio,
            'hidden_dim': self.hidden_dim,
            'num_gpus': self.num_gpus,
        }


class ChunkedSparseAllReducer(SparseAllReducer):
    """
    Chunked sparse AllReduce for pipeline scheduling.
    
    Supports overlapping communication with computation by chunking
    the communication into smaller pieces.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_gpus: int,
        rank: int,
        num_chunks: int = 2,
        **kwargs
    ):
        super().__init__(hidden_dim, num_gpus, rank, **kwargs)
        self.num_chunks = num_chunks
        
        # Per-chunk buffers
        self.chunk_buffers = [None for _ in range(num_chunks)]
    
    def chunked_sparse_allreduce(
        self,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_id: int,
        async_op: bool = True,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Perform chunked sparse AllReduce.
        
        Args:
            tensor: Chunk tensor [chunk_size, hidden_dim]
            mask: Chunk mask [chunk_size, hidden_dim]
            chunk_id: Which chunk this is
            async_op: Whether to perform asynchronously
            
        Returns:
            Tuple of (reduced_tensor, work_handle)
        """
        # Reuse parent's sparse allreduce
        result, work = self.sparse_allreduce(tensor, mask, async_op)
        
        # Store result in chunk buffer
        self.chunk_buffers[chunk_id] = result
        
        return result, work
    
    def get_chunk_result(self, chunk_id: int) -> Optional[torch.Tensor]:
        """Get result for a completed chunk."""
        if 0 <= chunk_id < len(self.chunk_buffers):
            return self.chunk_buffers[chunk_id]
        return None
    
    def clear_chunk_buffer(self, chunk_id: int):
        """Clear buffer for a chunk."""
        if 0 <= chunk_id < len(self.chunk_buffers):
            self.chunk_buffers[chunk_id] = None


class AdaptiveSparseAllReducer(SparseAllReducer):
    """
    Adaptive sparse AllReduce that chooses between sparse and dense
    based on actual sparsity level.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_gpus: int,
        rank: int,
        sparsity_threshold: float = 0.5,  # Use sparse if >50% sparse
        **kwargs
    ):
        super().__init__(hidden_dim, num_gpus, rank, **kwargs)
        self.sparsity_threshold = sparsity_threshold
        
        # Statistics
        self.total_operations = 0
        self.sparse_operations = 0
        self.dense_operations = 0
    
    def adaptive_allreduce(
        self,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor],
        async_op: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Adaptively choose sparse or dense AllReduce.
        """
        self.total_operations += 1
        
        if mask is None:
            self.dense_operations += 1
            return self._dense_allreduce(tensor, async_op)
        
        # Calculate actual sparsity
        actual_sparsity = 1.0 - mask.float().mean().item()
        
        if actual_sparsity > self.sparsity_threshold:
            # Use sparse AllReduce
            self.sparse_operations += 1
            return self.sparse_allreduce(tensor, mask, async_op)
        else:
            # Use dense AllReduce (more efficient for dense data)
            self.dense_operations += 1
            return self._dense_allreduce(tensor, async_op)
    
    def get_adaptation_stats(self) -> Dict:
        """Get statistics about adaptive behavior."""
        return {
            'total_operations': self.total_operations,
            'sparse_operations': self.sparse_operations,
            'dense_operations': self.dense_operations,
            'sparse_ratio': self.sparse_operations / max(1, self.total_operations),
            'sparsity_threshold': self.sparsity_threshold,
        }


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_sparse_communication_benefit(
    hidden_dim: int,
    num_gpus: int,
    sparsity: float,
    bandwidth_gbps: float = 600.0,
    latency_us: float = 10.0,
) -> Dict:
    """
    Estimate communication time savings from sparse AllReduce.
    
    Args:
        hidden_dim: Model hidden dimension
        num_gpus: Number of GPUs
        sparsity: Fraction of zeros (0-1)
        bandwidth_gbps: Network bandwidth
        latency_us: Network latency
        
    Returns:
        Dictionary with timing estimates
    """
    # Dense communication
    dense_bytes = hidden_dim * 2  # FP16
    dense_time = (dense_bytes / (bandwidth_gbps * 1e9) + latency_us * 1e-6) * 1000  # ms
    
    # Sparse communication
    # Need to send: values + indices
    active_elements = int(hidden_dim * (1 - sparsity))
    sparse_bytes = active_elements * 2 + active_elements * 4  # values (FP16) + indices (INT32)
    sparse_time = (sparse_bytes / (bandwidth_gbps * 1e9) + latency_us * 1e-6) * 1000  # ms
    
    # Overhead for index gathering
    gather_overhead = 0.01 * active_elements  # ~10us per 1000 elements
    
    sparse_total = sparse_time + gather_overhead
    
    return {
        'dense_time_ms': dense_time,
        'sparse_time_ms': sparse_total,
        'speedup': dense_time / sparse_total if sparse_total > 0 else 1.0,
        'bytes_saved': dense_bytes - sparse_bytes,
        'compression_ratio': sparse_bytes / dense_bytes if dense_bytes > 0 else 1.0,
    }


def create_sparse_communicator(
    hidden_dim: int,
    num_gpus: int,
    rank: int,
    mode: str = 'adaptive',
    **kwargs
) -> SparseAllReducer:
    """
    Factory function to create appropriate sparse communicator.
    
    Args:
        hidden_dim: Model hidden dimension
        num_gpus: Number of GPUs
        rank: Current rank
        mode: 'basic', 'chunked', or 'adaptive'
        
    Returns:
        Appropriate SparseAllReducer instance
    """
    if mode == 'chunked':
        return ChunkedSparseAllReducer(hidden_dim, num_gpus, rank, **kwargs)
    elif mode == 'adaptive':
        return AdaptiveSparseAllReducer(hidden_dim, num_gpus, rank, **kwargs)
    else:
        return SparseAllReducer(hidden_dim, num_gpus, rank, **kwargs)
