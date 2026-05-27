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
import torch.nn.functional as F
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


@triton.jit
def flash_ffn_fused_kernel(
    x_ptr, w1_ptr, w2_ptr, bias1_ptr, bias2_ptr, mask_ptr, y_ptr,
    stride_x_m, stride_x_h,
    stride_w1_h, stride_w1_f,
    stride_w2_f, stride_w2_h,
    stride_b1_f,
    stride_b2_h,
    stride_mask_m, stride_mask_f,
    stride_y_m, stride_y_h,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    intermediate_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS1: tl.constexpr,
    HAS_BIAS2: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    Fused FFN kernel:
        Y = ACT(X @ W1 + b1) @ W2 + b2

    The kernel keeps the intermediate tile in registers and streams across
    the FFN intermediate dimension, avoiding a full intermediate tensor.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    offs_f = tl.arange(0, BLOCK_F)

    mask_m = offs_m < seq_len
    mask_n = offs_n < hidden_dim
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for f0 in tl.range(0, intermediate_dim, BLOCK_F):
        cur_f = f0 + offs_f
        mask_f = cur_f < intermediate_dim
        inter = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)

        for h0 in tl.range(0, hidden_dim, BLOCK_H):
            cur_h = h0 + offs_h
            mask_h = cur_h < hidden_dim
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x_m + cur_h[None, :] * stride_x_h,
                mask=mask_m[:, None] & mask_h[None, :],
                other=0.0,
            )
            w1 = tl.load(
                w1_ptr + cur_h[:, None] * stride_w1_h + cur_f[None, :] * stride_w1_f,
                mask=mask_h[:, None] & mask_f[None, :],
                other=0.0,
            )
            inter += tl.dot(x, w1, allow_tf32=True)

        if HAS_BIAS1:
            b1 = tl.load(
                bias1_ptr + cur_f * stride_b1_f,
                mask=mask_f,
                other=0.0,
            )
            inter += b1[None, :]

        if HAS_MASK:
            pred = tl.load(
                mask_ptr + offs_m[:, None] * stride_mask_m + cur_f[None, :] * stride_mask_f,
                mask=mask_m[:, None] & mask_f[None, :],
                other=0.0,
            )
            inter *= pred

        if ACTIVATION == 0:
            inter = tl.maximum(inter, 0.0)
        elif ACTIVATION == 1:
            inter = 0.5 * inter * (1.0 + tl.math.erf(inter * 0.7071067811865476))
        elif ACTIVATION == 2:
            inter = inter * tl.sigmoid(inter)

        w2 = tl.load(
            w2_ptr + cur_f[:, None] * stride_w2_f + offs_n[None, :] * stride_w2_h,
            mask=mask_f[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(inter, w2.to(tl.float32), allow_tf32=True)

    if HAS_BIAS2:
        b2 = tl.load(
            bias2_ptr + offs_n * stride_b2_h,
            mask=mask_n,
            other=0.0,
        )
        acc += b2[None, :]

    tl.store(
        y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_h,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def flash_ffn_lans_indexed_kernel(
    x_ptr, w1_ptr, w2_ptr, bias1_ptr, mask_ptr, active_idx_ptr, y_ptr,
    stride_x_m, stride_x_h,
    stride_w1_h, stride_w1_f,
    stride_w2_f, stride_w2_h,
    stride_b1_f,
    stride_mask_m, stride_mask_f,
    stride_y_m, stride_y_h,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_active: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_A: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS1: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    Indexed sparse LANS kernel.

    Each CTA handles one [tokens x output-hidden] tile and a block of active
    LANS neurons. It accumulates those active neurons locally and stores once,
    avoiding cross-CTA atomic adds in the common sparse case. This path is
    intentionally scalar-reduction based instead of tl.dot based so LANS work
    maps to CUDA cores while HANS stays on the TensorCore GEMM path.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_ab = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    offs_a = pid_ab * BLOCK_A + tl.arange(0, BLOCK_A)

    mask_m = offs_m < seq_len
    mask_n = offs_n < hidden_dim
    mask_a = offs_a < num_active
    f_idx = tl.load(active_idx_ptr + offs_a, mask=mask_a, other=0)

    inter = tl.zeros((BLOCK_M, BLOCK_A), dtype=tl.float32)
    for h0 in tl.range(0, hidden_dim, BLOCK_H):
        cur_h = h0 + offs_h
        mask_h = cur_h < hidden_dim
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x_m + cur_h[None, :] * stride_x_h,
            mask=mask_m[:, None] & mask_h[None, :],
            other=0.0,
        )
        w1 = tl.load(
            w1_ptr + cur_h[:, None] * stride_w1_h + f_idx[None, :] * stride_w1_f,
            mask=mask_h[:, None] & mask_a[None, :],
            other=0.0,
        )
        prod = x[:, :, None].to(tl.float32) * w1[None, :, :].to(tl.float32)
        prod = tl.where(mask_h[None, :, None] & mask_a[None, None, :], prod, 0.0)
        inter += tl.sum(prod, axis=1)

    if HAS_BIAS1:
        b1 = tl.load(bias1_ptr + f_idx * stride_b1_f, mask=mask_a, other=0.0)
        inter += b1[None, :]

    if HAS_MASK:
        pred = tl.load(
            mask_ptr + offs_m[:, None] * stride_mask_m + f_idx[None, :] * stride_mask_f,
            mask=mask_m[:, None] & mask_a[None, :],
            other=0.0,
        )
        inter *= pred

    if ACTIVATION == 0:
        inter = tl.maximum(inter, 0.0)
    elif ACTIVATION == 1:
        inter = 0.5 * inter * (1.0 + tl.math.erf(inter * 0.7071067811865476))
    elif ACTIVATION == 2:
        inter = inter * tl.sigmoid(inter)

    w2 = tl.load(
        w2_ptr + f_idx[:, None] * stride_w2_f + offs_n[None, :] * stride_w2_h,
        mask=mask_a[:, None] & mask_n[None, :],
        other=0.0,
    )
    out_prod = inter[:, :, None] * w2[None, :, :].to(tl.float32)
    out_prod = tl.where(mask_a[None, :, None] & mask_n[None, None, :], out_prod, 0.0)
    out = tl.sum(out_prod, axis=1)

    tl.store(
        y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_h,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


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

    This public wrapper uses the indexed scalar-reduction implementation. The
    older atomic kernel is kept only for historical comparison and is not used
    by the runtime hybrid path.
    
    Args:
        x: Input tensor [seq_len, hidden_dim]
        w1: First weight matrix [hidden_dim, num_lans]
        w2: Second weight matrix [num_lans, hidden_dim]
        active_indices: Indices of active LANS neurons [num_active]
        activation: Activation function
        
    Returns:
        Output tensor [seq_len, hidden_dim]
    """
    return flash_ffn_lans_indexed(
        x=x,
        w1_lans=w1,
        w2_lans=w2,
        active_indices=active_indices,
        activation=activation,
    )


def flash_ffn_fused(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    prediction_mask: Optional[torch.Tensor] = None,
    activation: str = 'relu',
) -> torch.Tensor:
    """Launch the fused Triton FFN kernel."""
    if x.dim() != 2:
        raise ValueError(f"FlashFFN expects a 2D tensor, got shape {tuple(x.shape)}")

    x = x.contiguous()
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    seq_len, hidden_dim = x.shape
    intermediate_dim = w1.shape[1]
    y = torch.empty((seq_len, hidden_dim), device=x.device, dtype=x.dtype)

    if prediction_mask is not None:
        prediction_mask = prediction_mask[:, :intermediate_dim].contiguous()
    else:
        prediction_mask = x

    if bias1 is None:
        bias1 = w1
        has_bias1 = False
    else:
        bias1 = bias1.contiguous()
        has_bias1 = True

    if bias2 is None:
        bias2 = w2
        has_bias2 = False
    else:
        bias2 = bias2.contiguous()
        has_bias2 = True

    act_map = {'relu': 0, 'gelu': 1, 'silu': 2}
    if activation not in act_map:
        raise ValueError(f"Unsupported activation: {activation}")

    block_m = 16
    block_n = 32
    block_f = 32
    block_h = 32
    grid = (triton.cdiv(seq_len, block_m), triton.cdiv(hidden_dim, block_n))

    flash_ffn_fused_kernel[grid](
        x, w1, w2, bias1, bias2, prediction_mask, y,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        bias1.stride(0) if has_bias1 else 0,
        bias2.stride(0) if has_bias2 else 0,
        prediction_mask.stride(0), prediction_mask.stride(1),
        y.stride(0), y.stride(1),
        seq_len,
        hidden_dim,
        intermediate_dim,
        block_m,
        block_n,
        block_f,
        block_h,
        act_map[activation],
        has_bias1,
        has_bias2,
        prediction_mask is not x,
        num_warps=4,
    )

    return y


def flash_ffn_lans_indexed(
    x: torch.Tensor,
    w1_lans: torch.Tensor,
    w2_lans: torch.Tensor,
    active_indices: torch.Tensor,
    bias1_lans: Optional[torch.Tensor] = None,
    prediction_mask_lans: Optional[torch.Tensor] = None,
    activation: str = 'relu',
) -> torch.Tensor:
    """Launch the sparse indexed LANS kernel."""
    seq_len, hidden_dim = x.shape
    y = torch.zeros((seq_len, hidden_dim), device=x.device, dtype=x.dtype)

    if active_indices.numel() == 0:
        return y

    x = x.contiguous()
    w1_lans = w1_lans.contiguous()
    w2_lans = w2_lans.contiguous()
    active_indices = active_indices.to(device=x.device, dtype=torch.long).contiguous()

    if prediction_mask_lans is None:
        prediction_mask_lans = x
        has_mask = False
    else:
        prediction_mask_lans = prediction_mask_lans.contiguous()
        has_mask = True

    if bias1_lans is None:
        bias1_lans = w1_lans
        has_bias1 = False
    else:
        bias1_lans = bias1_lans.contiguous()
        has_bias1 = True

    act_map = {'relu': 0, 'gelu': 1, 'silu': 2}
    if activation not in act_map:
        raise ValueError(f"Unsupported activation: {activation}")

    block_m = 16
    block_n = 32
    block_h = 64
    block_a = max(16, triton.next_power_of_2(active_indices.numel()))
    grid = (
        triton.cdiv(seq_len, block_m),
        triton.cdiv(hidden_dim, block_n),
        1,
    )

    flash_ffn_lans_indexed_kernel[grid](
        x, w1_lans, w2_lans, bias1_lans, prediction_mask_lans, active_indices, y,
        x.stride(0), x.stride(1),
        w1_lans.stride(0), w1_lans.stride(1),
        w2_lans.stride(0), w2_lans.stride(1),
        bias1_lans.stride(0) if has_bias1 else 0,
        prediction_mask_lans.stride(0), prediction_mask_lans.stride(1),
        y.stride(0), y.stride(1),
        seq_len,
        hidden_dim,
        active_indices.numel(),
        block_m,
        block_n,
        block_h,
        block_a,
        act_map[activation],
        has_bias1,
        has_mask,
        num_warps=4,
    )

    return y


def flash_ffn_hybrid(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_hans: int,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    prediction_mask: Optional[torch.Tensor] = None,
    activation: str = 'relu',
    sparse_lans_threshold: float = 0.25,
    max_sparse_lans_active: int = 64,
) -> torch.Tensor:
    """
    Hybrid HANS/LANS execution.

    HANS neurons are evaluated by the dense fused TensorCore-oriented path.
    LANS neurons are evaluated by the indexed sparse CUDA-core-oriented path
    using only neurons active in the current token block.
    """
    seq_len, hidden_dim = x.shape
    y = torch.zeros((seq_len, hidden_dim), device=x.device, dtype=x.dtype)
    num_hans = max(0, min(int(num_hans), w1.shape[1]))
    num_lans = w1.shape[1] - num_hans

    if prediction_mask is None:
        return flash_ffn_fused(
            x=x,
            w1=w1,
            w2=w2,
            bias1=bias1,
            bias2=bias2,
            prediction_mask=None,
            activation=activation,
        )

    lans_active_indices = None
    if num_lans > 0:
        lans_mask = prediction_mask[:, num_hans:]
        lans_active_indices = torch.nonzero(lans_mask.any(dim=0), as_tuple=False).flatten()
        lans_active_ratio = lans_active_indices.numel() / max(1, num_lans)
        if (
            lans_active_ratio > sparse_lans_threshold
            or lans_active_indices.numel() > max_sparse_lans_active
        ):
            return flash_ffn_fused(
                x=x,
                w1=w1,
                w2=w2,
                bias1=bias1,
                bias2=bias2,
                prediction_mask=prediction_mask,
                activation=activation,
            )

    if num_hans > 0:
        hans_mask = prediction_mask[:, :num_hans] if prediction_mask is not None else None
        y += flash_ffn_fused(
            x=x,
            w1=w1[:, :num_hans],
            w2=w2[:num_hans, :],
            bias1=bias1[:num_hans] if bias1 is not None else None,
            bias2=None,
            prediction_mask=hans_mask,
            activation=activation,
        )

    if num_lans > 0:
        lans_mask = prediction_mask[:, num_hans:]
        y += flash_ffn_lans_indexed(
            x=x,
            w1_lans=w1[:, num_hans:],
            w2_lans=w2[num_hans:, :],
            active_indices=lans_active_indices,
            bias1_lans=bias1[num_hans:] if bias1 is not None else None,
            prediction_mask_lans=lans_mask,
            activation=activation,
        )

    if bias2 is not None:
        y += bias2

    return y


def flash_ffn_hybrid_fast(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_hans: int,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    prediction_mask: Optional[torch.Tensor] = None,
    activation: str = 'relu',
    sparse_lans_threshold: float = 0.25,
    max_sparse_lans_active: int = 64,
    w1_hans: Optional[torch.Tensor] = None,
    w2_hans: Optional[torch.Tensor] = None,
    bias1_hans: Optional[torch.Tensor] = None,
    w1_lans: Optional[torch.Tensor] = None,
    w2_lans: Optional[torch.Tensor] = None,
    bias1_lans: Optional[torch.Tensor] = None,
    active_lans_indices: Optional[torch.Tensor] = None,
    active_lans_by_tile: Optional[list] = None,
) -> torch.Tensor:
    """
    High-performance hybrid FFN.

    HANS uses cuBLAS/TensorCore GEMM via torch.matmul. LANS uses the
    output-tile aggregated sparse Triton kernel when it is truly sparse,
    otherwise it falls back to dense GEMM to avoid sparse overhead.
    """
    num_hans = max(0, min(int(num_hans), w1.shape[1]))
    num_lans = w1.shape[1] - num_hans
    lans_mask = prediction_mask[:, num_hans:] if (prediction_mask is not None and num_lans > 0) else None
    active_indices = active_lans_indices
    if lans_mask is not None:
        if active_indices is None and active_lans_by_tile is None:
            dense = x @ w1
            if bias1 is not None:
                dense = dense + bias1
            dense = dense * prediction_mask.to(device=x.device, dtype=x.dtype)
            if activation == 'relu':
                dense = F.relu(dense)
            elif activation == 'gelu':
                dense = F.gelu(dense)
            elif activation == 'silu':
                dense = F.silu(dense)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            y = dense @ w2
            if bias2 is not None:
                y = y + bias2
            return y
        if active_indices is not None:
            active_indices = active_indices.to(device=x.device, dtype=torch.long)
            active_ratio = active_indices.numel() / max(1, num_lans)
        else:
            active_ratio = 0.0

        if active_indices is not None and (
            active_indices.numel() > 0
            and (
                active_ratio > sparse_lans_threshold
                or active_indices.numel() > max_sparse_lans_active
            )
        ):
            dense = x @ w1
            if bias1 is not None:
                dense = dense + bias1
            dense = dense * prediction_mask.to(device=x.device, dtype=x.dtype)
            if activation == 'relu':
                dense = F.relu(dense)
            elif activation == 'gelu':
                dense = F.gelu(dense)
            elif activation == 'silu':
                dense = F.silu(dense)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            y = dense @ w2
            if bias2 is not None:
                y = y + bias2
            return y

    y = None

    if num_hans > 0:
        w1_h = w1_hans if w1_hans is not None else w1[:, :num_hans]
        w2_h = w2_hans if w2_hans is not None else w2[:num_hans, :]
        b1_h = bias1_hans if bias1_hans is not None else (bias1[:num_hans] if bias1 is not None else None)
        hans = x @ w1_h
        if b1_h is not None:
            hans = hans + b1_h
        if prediction_mask is not None:
            hans = hans * prediction_mask[:, :num_hans].to(device=x.device, dtype=x.dtype)

        if activation == 'relu':
            hans = F.relu(hans)
        elif activation == 'gelu':
            hans = F.gelu(hans)
        elif activation == 'silu':
            hans = F.silu(hans)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        y = hans @ w2_h

    if num_lans > 0:
        if active_lans_by_tile is not None and lans_mask is not None:
            if y is None:
                y = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
            w1_l = w1_lans if w1_lans is not None else w1[:, num_hans:]
            w2_l = w2_lans if w2_lans is not None else w2[num_hans:, :]
            b1_l = bias1_lans if bias1_lans is not None else (bias1[num_hans:] if bias1 is not None else None)
            for tile in active_lans_by_tile:
                start, end, tile_active = tile
                tile_active = tile_active.to(device=x.device, dtype=torch.long)
                if tile_active.numel() == 0:
                    continue
                tile_ratio = tile_active.numel() / max(1, num_lans)
                if tile_ratio > sparse_lans_threshold or tile_active.numel() > max_sparse_lans_active:
                    lans = x[start:end] @ w1_l
                    if b1_l is not None:
                        lans = lans + b1_l
                    lans = lans * lans_mask[start:end].to(device=x.device, dtype=x.dtype)
                    if activation == 'relu':
                        lans = F.relu(lans)
                    elif activation == 'gelu':
                        lans = F.gelu(lans)
                    elif activation == 'silu':
                        lans = F.silu(lans)
                    else:
                        raise ValueError(f"Unsupported activation: {activation}")
                    y[start:end] = y[start:end] + lans @ w2_l
                else:
                    y[start:end] = y[start:end] + flash_ffn_lans_indexed(
                        x=x[start:end],
                        w1_lans=w1_l,
                        w2_lans=w2_l,
                        active_indices=tile_active,
                        bias1_lans=b1_l,
                        prediction_mask_lans=lans_mask[start:end],
                        activation=activation,
                    )
        else:
            if lans_mask is None:
                active_indices = None
            else:
                active_indices = active_indices

            use_sparse_lans = (
                lans_mask is not None
                and active_indices is not None
                and active_indices.numel() > 0
                and active_ratio <= sparse_lans_threshold
                and active_indices.numel() <= max_sparse_lans_active
            )

            if use_sparse_lans:
                lans_y = flash_ffn_lans_indexed(
                    x=x,
                    w1_lans=w1_lans if w1_lans is not None else w1[:, num_hans:],
                    w2_lans=w2_lans if w2_lans is not None else w2[num_hans:, :],
                    active_indices=active_indices,
                    bias1_lans=bias1_lans if bias1_lans is not None else (bias1[num_hans:] if bias1 is not None else None),
                    prediction_mask_lans=lans_mask,
                    activation=activation,
                )
                y = lans_y if y is None else y + lans_y
            elif lans_mask is None or active_indices is None or active_indices.numel() > 0:
                w1_l = w1_lans if w1_lans is not None else w1[:, num_hans:]
                w2_l = w2_lans if w2_lans is not None else w2[num_hans:, :]
                b1_l = bias1_lans if bias1_lans is not None else (bias1[num_hans:] if bias1 is not None else None)
                lans = x @ w1_l
                if b1_l is not None:
                    lans = lans + b1_l
                if lans_mask is not None:
                    lans = lans * lans_mask.to(device=x.device, dtype=x.dtype)

                if activation == 'relu':
                    lans = F.relu(lans)
                elif activation == 'gelu':
                    lans = F.gelu(lans)
                elif activation == 'silu':
                    lans = F.silu(lans)
                else:
                    raise ValueError(f"Unsupported activation: {activation}")

                lans_y = lans @ w2_l
                y = lans_y if y is None else y + lans_y

    if y is None:
        y = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)

    if bias2 is not None:
        y += bias2

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

        self._weight_cache = None
        
        self._init_weights()

    def _get_weight_cache(self, dtype: torch.dtype, device: torch.device) -> Dict:
        key = (
            self.w1.data_ptr(),
            self.w2.data_ptr(),
            self.bias1.data_ptr() if self.bias1 is not None else 0,
            self.bias2.data_ptr() if self.bias2 is not None else 0,
            self.num_hans,
            self.num_lans,
            dtype,
            device,
        )
        if self._weight_cache is not None and self._weight_cache.get('key') == key:
            return self._weight_cache

        num_hans = self.num_hans
        cache = {
            'key': key,
            'w1': self.w1.to(device=device, dtype=dtype).contiguous(),
            'w2': self.w2.to(device=device, dtype=dtype).contiguous(),
            'bias1': self.bias1.to(device=device, dtype=dtype).contiguous() if self.bias1 is not None else None,
            'bias2': self.bias2.to(device=device, dtype=dtype).contiguous() if self.bias2 is not None else None,
        }
        if num_hans > 0:
            cache['w1_hans'] = self.w1[:, :num_hans].to(device=device, dtype=dtype).contiguous()
            cache['w2_hans'] = self.w2[:num_hans, :].to(device=device, dtype=dtype).contiguous()
            cache['bias1_hans'] = (
                self.bias1[:num_hans].to(device=device, dtype=dtype).contiguous()
                if self.bias1 is not None else None
            )
        if self.num_lans > 0:
            cache['w1_lans'] = self.w1[:, num_hans:].to(device=device, dtype=dtype).contiguous()
            cache['w2_lans'] = self.w2[num_hans:, :].to(device=device, dtype=dtype).contiguous()
            cache['bias1_lans'] = (
                self.bias1[num_hans:].to(device=device, dtype=dtype).contiguous()
                if self.bias1 is not None else None
            )
        self._weight_cache = cache
        return cache
    
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
        active_lans_indices = None
        active_lans_by_tile = None
        if isinstance(prediction_mask, dict):
            active_lans_indices = prediction_mask.get('active_lans_indices')
            active_lans_by_tile = prediction_mask.get('active_lans_by_tile')
            mask_payload = prediction_mask.get('mask')
            if mask_payload is None:
                mask_payload = prediction_mask.get('prediction_mask')
            if mask_payload is None:
                mask_payload = prediction_mask.get('activation_mask')
            prediction_mask = mask_payload
        elif isinstance(prediction_mask, (tuple, list)):
            if len(prediction_mask) == 0:
                prediction_mask = None
            else:
                active_lans_indices = prediction_mask[1] if len(prediction_mask) > 1 else None
                active_lans_by_tile = prediction_mask[2] if len(prediction_mask) > 2 else None
                prediction_mask = prediction_mask[0]

        cache = self._get_weight_cache(x.dtype, x.device)
        w1 = cache['w1']
        w2 = cache['w2']
        bias1 = cache['bias1']
        bias2 = cache['bias2']

        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            if self.num_hans > 0 and self.num_lans > 0:
                return flash_ffn_hybrid_fast(
                    x=x,
                    w1=w1,
                    w2=w2,
                    num_hans=self.num_hans,
                    bias1=bias1,
                    bias2=bias2,
                    prediction_mask=prediction_mask,
                    activation=self.activation,
                    sparse_lans_threshold=0.25,
                    max_sparse_lans_active=64,
                    w1_hans=cache.get('w1_hans'),
                    w2_hans=cache.get('w2_hans'),
                    bias1_hans=cache.get('bias1_hans'),
                    w1_lans=cache.get('w1_lans'),
                    w2_lans=cache.get('w2_lans'),
                    bias1_lans=cache.get('bias1_lans'),
                    active_lans_indices=active_lans_indices,
                    active_lans_by_tile=active_lans_by_tile,
                )

            return flash_ffn_fused(
                x=x,
                w1=w1,
                w2=w2,
                bias1=bias1,
                bias2=bias2,
                prediction_mask=prediction_mask,
                activation=self.activation,
            )

        intermediate = x @ w1

        if bias1 is not None:
            intermediate = intermediate + bias1

        if prediction_mask is not None:
            mask = prediction_mask[:, : intermediate.shape[1]].to(
                device=intermediate.device,
                dtype=intermediate.dtype,
            )
            intermediate = intermediate * mask

        if self.activation == 'relu':
            intermediate = F.relu(intermediate)
        elif self.activation == 'gelu':
            intermediate = F.gelu(intermediate)
        elif self.activation == 'silu':
            intermediate = F.silu(intermediate)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        y = intermediate @ w2

        if bias2 is not None:
            y = y + bias2

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
