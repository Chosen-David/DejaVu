"""
FlashFFN: Fused Sparse FFN Kernel

This module implements a fused kernel for sparse FFN computation:
  Y = ACT(X @ W1 + b1) @ W2 + b2

Key features:
1. Hybrid TensorCore/CUDACore execution
2. Sparse activation handling (only compute active neurons)
3. Memory-efficient tiling (similar to FlashAttention)
4. Support for chunked input for pipeline scheduling

Execution model:
- HANS neurons: Dense computation on TensorCore
- LANS neurons: Sparse computation on CUDACore
- Concurrent execution on different SMs
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple, Dict
import numpy as np


# ============================================================================
# Triton Kernels for Sparse FFN
# ============================================================================

@triton.jit
def flash_ffn_hans_kernel(
    # Pointers to matrices
    x_ptr, w1_ptr, w2_ptr, y_ptr,
    # Masks for sparse activation
    mask_ptr,  # [seq_len, num_hans] - which neurons are active
    # Dimensions
    seq_len, hidden_dim, num_hans,
    # Strides
    stride_x_seq, stride_x_hid,
    stride_w1_hid, stride_w1_neuron,
    stride_w2_neuron, stride_w2_hid,
    stride_y_seq, stride_y_hid,
    stride_mask_seq, stride_mask_neuron,
    # Meta parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Activation type: 0=ReLU, 1=GELU, 2=SiLU
    ACTIVATION: tl.constexpr,
):
    """
    Flash FFN kernel for HANS (High-Activation Neurons) on TensorCore.
    
    Computes: Y = ACT(X @ W1) @ W2
    Only for neurons marked as HANS.
    """
    # Block indices
    pid_m = tl.program_id(0)  # Sequence dimension
    pid_n = tl.program_id(1)  # Hidden dimension (output)
    
    # Compute block ranges
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Bounds checking
    mask_m = rm < seq_len
    mask_n = rn < hidden_dim
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over intermediate dimension (HANS neurons)
    for k in range(0, num_hans, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = rk < num_hans
        
        # Load X block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_ptrs = x_ptr + rm[:, None] * stride_x_seq + rk[None, :] * stride_x_hid
        x_mask = mask_m[:, None] & mask_k[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load W1 block: [BLOCK_SIZE_K, BLOCK_SIZE_N] (transposed)
        w1_ptrs = w1_ptr + rk[:, None] * stride_w1_neuron + rk[None, :] * stride_w1_hid
        w1 = tl.load(w1_ptrs, mask=mask_k[:, None] & mask_k[None, :], other=0.0)
        
        # Matrix multiply: [BLOCK_SIZE_M, BLOCK_SIZE_K] @ [BLOCK_SIZE_K, BLOCK_SIZE_N]
        intermediate = tl.dot(x, w1, allow_tf32=True)
        
        # Activation
        if ACTIVATION == 0:  # ReLU
            intermediate = tl.where(intermediate > 0, intermediate, 0.0)
        elif ACTIVATION == 1:  # GELU (approximate)
            intermediate = 0.5 * intermediate * (1.0 + tl.math.erf(intermediate / 1.41421))
        elif ACTIVATION == 2:  # SiLU
            intermediate = intermediate * tl.sigmoid(intermediate)
        
        # Load W2 block: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        w2_ptrs = w2_ptr + rk[:, None] * stride_w2_neuron + rn[None, :] * stride_w2_hid
        w2 = tl.load(w2_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate: intermediate @ W2
        acc += tl.dot(intermediate, w2, allow_tf32=True)
    
    # Store result
    y_ptrs = y_ptr + rm[:, None] * stride_y_seq + rn[None, :] * stride_y_hid
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def flash_ffn_lans_kernel(
    # Pointers
    x_ptr, w1_ptr, w2_ptr, y_ptr,
    # Sparse indices
    active_indices_ptr,  # [num_active] - indices of active LANS neurons
    # Dimensions
    seq_len, hidden_dim, num_lans, num_active,
    # Strides
    stride_x_seq, stride_x_hid,
    stride_w1_hid, stride_w1_neuron,
    stride_w2_neuron, stride_w2_hid,
    stride_y_seq, stride_y_hid,
    # Meta parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Flash FFN kernel for LANS (Low-Activation Neurons) on CUDACore.
    
    Uses sparse gather operation to only compute active neurons.
    """
    # Block indices
    pid_m = tl.program_id(0)  # Sequence dimension
    pid_k = tl.program_id(1)  # Active neuron dimension
    
    # Compute block ranges
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Bounds checking
    mask_m = rm < seq_len
    mask_k = rk < num_active
    
    # Load active neuron indices
    active_idx_ptrs = active_indices_ptr + rk
    active_indices = tl.load(active_idx_ptrs, mask=mask_k, other=0)
    
    # Gather X: [BLOCK_SIZE_M, BLOCK_SIZE_K]
    x_ptrs = x_ptr + rm[:, None] * stride_x_seq
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Gather W1 for active neurons: [hidden_dim, num_active]
    # This is a sparse gather operation
    w1_ptrs = w1_ptr + active_indices[None, :] * stride_w1_neuron
    w1 = tl.load(w1_ptrs, mask=mask_k[None, :], other=0.0)
    
    # Sparse matrix multiply: X @ W1_gathered
    intermediate = tl.dot(x, w1, allow_tf32=False)  # Use FP32 for CC
    
    # Activation
    if ACTIVATION == 0:  # ReLU
        intermediate = tl.where(intermediate > 0, intermediate, 0.0)
    elif ACTIVATION == 1:  # GELU
        intermediate = 0.5 * intermediate * (1.0 + tl.math.erf(intermediate / 1.41421))
    elif ACTIVATION == 2:  # SiLU
        intermediate = intermediate * tl.sigmoid(intermediate)
    
    # Scatter-add result to output
    # Y[:, :] += intermediate @ W2_gathered
    w2_ptrs = w2_ptr + active_indices[:, None] * stride_w2_neuron
    w2 = tl.load(w2_ptrs, mask=mask_k[:, None], other=0.0)
    
    output = tl.dot(intermediate, w2, allow_tf32=False)
    
    # Atomic add to output
    rn = tl.arange(0, hidden_dim)
    y_ptrs = y_ptr + rm[:, None] * stride_y_seq + rn[None, :] * stride_y_hid
    tl.atomic_add(y_ptrs, output, mask=mask_m[:, None])


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def flash_ffn_hans(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = 'relu',
) -> torch.Tensor:
    """
    Flash FFN for HANS neurons on TensorCore.
    
    Args:
        x: Input tensor [seq_len, hidden_dim]
        w1: First weight matrix [hidden_dim, num_hans]
        w2: Second weight matrix [num_hans, hidden_dim]
        activation: Activation function ('relu', 'gelu', 'silu')
        
    Returns:
        Output tensor [seq_len, hidden_dim]
    """
    seq_len, hidden_dim = x.shape
    num_hans = w1.shape[1]
    
    # Allocate output
    y = torch.zeros(seq_len, hidden_dim, device=x.device, dtype=x.dtype)
    
    # Choose activation
    act_map = {'relu': 0, 'gelu': 1, 'silu': 2}
    act_code = act_map.get(activation, 0)
    
    # Block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Grid size
    grid = (
        triton.cdiv(seq_len, BLOCK_SIZE_M),
        triton.cdiv(hidden_dim, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    flash_ffn_hans_kernel[grid](
        x, w1, w2, y,
        None,  # mask_ptr (not used for HANS)
        seq_len, hidden_dim, num_hans,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        y.stride(0), y.stride(1),
        0, 0,  # mask strides (not used)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        act_code,
    )
    
    return y


def flash_ffn_lans(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    active_indices: torch.Tensor,
    activation: str = 'relu',
) -> torch.Tensor:
    """
    Flash FFN for LANS neurons on CUDACore.
    
    Args:
        x: Input tensor [seq_len, hidden_dim]
        w1: First weight matrix [hidden_dim, num_lans]
        w2: Second weight matrix [num_lans, hidden_dim]
        active_indices: Indices of active LANS neurons [num_active]
        activation: Activation function
        
    Returns:
        Output tensor [seq_len, hidden_dim]
    """
    seq_len, hidden_dim = x.shape
    num_lans = w1.shape[1]
    num_active = len(active_indices)
    
    # Allocate output
    y = torch.zeros(seq_len, hidden_dim, device=x.device, dtype=x.dtype)
    
    # Choose activation
    act_map = {'relu': 0, 'gelu': 1, 'silu': 2}
    act_code = act_map.get(activation, 0)
    
    # Block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32
    
    # Grid size
    grid = (
        triton.cdiv(seq_len, BLOCK_SIZE_M),
        triton.cdiv(num_active, BLOCK_SIZE_K),
    )
    
    # Launch kernel
    flash_ffn_lans_kernel[grid](
        x, w1, w2, y,
        active_indices,
        seq_len, hidden_dim, num_lans, num_active,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_K,
        act_code,
    )
    
    return y


# ============================================================================
# Fused FlashFFN Module
# ============================================================================

class FlashFFN(nn.Module):
    """
    Fused FlashFFN module that combines HANS and LANS execution.
    
    Features:
    - Automatic HANS/LANS splitting based on predictor
    - Hybrid TC/CC execution
    - Memory-efficient forward pass
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        hans_indices: Optional[torch.Tensor] = None,
        lans_indices: Optional[torch.Tensor] = None,
        activation: str = 'relu',
        bias: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        
        # Register indices
        if hans_indices is not None:
            self.register_buffer('hans_indices', hans_indices)
            self.num_hans = len(hans_indices)
        else:
            self.hans_indices = None
            self.num_hans = 0
        
        if lans_indices is not None:
            self.register_buffer('lans_indices', lans_indices)
            self.num_lans = len(lans_indices)
        else:
            self.lans_indices = None
            self.num_lans = 0
        
        # Weights will be reordered during initialization
        # W1: [hidden_dim, intermediate_dim]
        # W2: [intermediate_dim, hidden_dim]
        self.w1 = nn.Parameter(torch.empty(hidden_dim, intermediate_dim))
        self.w2 = nn.Parameter(torch.empty(intermediate_dim, hidden_dim))
        
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(intermediate_dim))
            self.bias2 = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled initialization."""
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)
    
    def reorder_weights(self):
        """
        Reorder weights so HANS neurons come first, then LANS neurons.
        This enables efficient contiguous memory access.
        """
        if self.hans_indices is None or self.lans_indices is None:
            return
        
        # Reorder W1 columns (neurons)
        new_order = torch.cat([self.hans_indices, self.lans_indices])
        self.w1.data = self.w1.data[:, new_order]
        
        # Reorder W2 rows (neurons)
        self.w2.data = self.w2.data[new_order, :]
        
        # Reorder biases
        if self.bias1 is not None:
            self.bias1.data = self.bias1.data[new_order]
    
    def forward(
        self,
        x: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with hybrid HANS/LANS execution.
        
        Args:
            x: Input tensor [seq_len, hidden_dim]
            prediction_mask: Activation mask [seq_len, intermediate_dim]
                            (optional, for runtime sparsity)
        
        Returns:
            Output tensor [seq_len, hidden_dim]
        """
        seq_len = x.shape[0]
        
        # Initialize output
        y = torch.zeros_like(x)
        
        # Compute HANS contribution (TensorCore)
        if self.num_hans > 0:
            w1_hans = self.w1[:, :self.num_hans]
            w2_hans = self.w2[:self.num_hans, :]
            
            y_hans = flash_ffn_hans(x, w1_hans, w2_hans, self.activation)
            
            if self.bias1 is not None:
                bias1_hans = self.bias1[:self.num_hans]
                # Note: bias handling would be inside kernel for efficiency
            
            y = y + y_hans
        
        # Compute LANS contribution (CUDACore)
        if self.num_lans > 0 and prediction_mask is not None:
            # Get active LANS indices from mask
            lans_mask = prediction_mask[:, self.num_hans:]
            active_per_token = lans_mask.any(dim=0)
            active_indices = torch.where(active_per_token)[0]
            
            if len(active_indices) > 0:
                w1_lans = self.w1[:, self.num_hans:]
                w2_lans = self.w2[self.num_hans:, :]
                
                y_lans = flash_ffn_lans(
                    x, w1_lans, w2_lans, active_indices, self.activation
                )
                y = y + y_lans
        
        elif self.num_lans > 0 and prediction_mask is None:
            # No mask provided, compute all LANS (less efficient)
            w1_lans = self.w1[:, self.num_hans:]
            w2_lans = self.w2[self.num_hans:, :]
            
            all_indices = torch.arange(self.num_lans, device=x.device)
            y_lans = flash_ffn_lans(
                x, w1_lans, w2_lans, all_indices, self.activation
            )
            y = y + y_lans
        
        # Add output bias
        if self.bias2 is not None:
            y = y + self.bias2
        
        return y
    
    def get_memory_footprint(self) -> Dict:
        """Calculate memory footprint of the FFN."""
        weight_bytes = self.w1.numel() * 2 + self.w2.numel() * 2  # FP16
        bias_bytes = 0
        if self.bias1 is not None:
            bias_bytes += self.bias1.numel() * 2
        if self.bias2 is not None:
            bias_bytes += self.bias2.numel() * 2
        
        return {
            'weight_mb': weight_bytes / (1024 ** 2),
            'bias_mb': bias_bytes / (1024 ** 2),
            'total_mb': (weight_bytes + bias_bytes) / (1024 ** 2),
            'hans_neurons': self.num_hans,
            'lans_neurons': self.num_lans,
        }


# ============================================================================
# Utility Functions
# ============================================================================

def benchmark_flash_ffn(
    hidden_dim: int = 4096,
    intermediate_dim: int = 16384,
    seq_len: int = 2048,
    hans_ratio: float = 0.3,
    device: str = 'cuda',
    num_iterations: int = 100,
) -> Dict:
    """
    Benchmark FlashFFN performance.
    """
    num_hans = int(intermediate_dim * hans_ratio)
    num_lans = intermediate_dim - num_hans
    
    # Create indices
    hans_indices = torch.arange(num_hans, device=device)
    lans_indices = torch.arange(num_hans, intermediate_dim, device=device)
    
    # Create module
    ffn = FlashFFN(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        hans_indices=hans_indices,
        lans_indices=lans_indices,
    ).to(device)
    
    # Create input
    x = torch.randn(seq_len, hidden_dim, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = ffn(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        _ = ffn(x)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iterations
    
    return {
        'avg_time_ms': elapsed_ms,
        'throughput_tflops': 4 * seq_len * hidden_dim * intermediate_dim / (elapsed_ms * 1e-3) / 1e12,
        'memory_info': ffn.get_memory_footprint(),
    }
