from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _resolve_full_res_kernel(
    out_ptr,
    c_ptr,
    z_ptr,
    v_ptr,
    shape_ptr,
    S: tl.constexpr,
    F: tl.constexpr,
    max_s: tl.constexpr,
    max_f: tl.constexpr,
    stride_c0,
    stride_c1,
    stride_z0,
    stride_z1,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_shape0,
    stride_shape1,
    stride_out0,
    stride_out1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_shard = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    s_loc = tl.load(shape_ptr + pid_shard * stride_shape0 + 0 * stride_shape1)
    f_loc = tl.load(shape_ptr + pid_shard * stride_shape0 + 1 * stride_shape1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    valid_m = m_offs < s_loc
    valid_n = n_offs < f_loc

    rows = tl.load(
        c_ptr + pid_shard * stride_c0 + m_offs * stride_c1, mask=valid_m, other=-1
    )
    cols = tl.load(
        z_ptr + pid_shard * stride_z0 + n_offs * stride_z1, mask=valid_n, other=-1
    )

    vals = tl.load(
        v_ptr
        + pid_shard * stride_v0
        + m_offs[:, None] * stride_v1
        + n_offs[None, :] * stride_v2,
        mask=valid_m[:, None] & valid_n[None, :],
        other=0.0,
    )

    valid_rows = (rows >= 0) & (rows < S)
    valid_cols = (cols >= 0) & (cols < F)
    mask = valid_rows[:, None] & valid_cols[None, :]
    tl.store(
        out_ptr + rows[:, None] * stride_out0 + cols[None, :] * stride_out1,
        vals,
        mask=mask,
    )


def resolve_full_res_packed_triton(
    c_all: torch.Tensor,
    z_all: torch.Tensor,
    r_all: torch.Tensor,
    shape_all: torch.Tensor,
    S: int,
    F: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if c_all.dim() != 2 or z_all.dim() != 2 or r_all.dim() != 3 or shape_all.dim() != 2:
        raise ValueError("Expected shapes c_all=[P,max_s], z_all=[P,max_f], r_all=[P,max_s,max_f], shape_all=[P,2]")

    if not (c_all.is_cuda and z_all.is_cuda and r_all.is_cuda and shape_all.is_cuda):
        raise ValueError("Triton resolve requires CUDA tensors")

    world_size, max_s = c_all.shape
    _, max_f = z_all.shape
    if r_all.shape != (world_size, max_s, max_f):
        raise ValueError("Packed shard tensor shape mismatch")

    if out is None:
        out = torch.zeros((S, F), dtype=r_all.dtype, device=r_all.device)
    else:
        out.zero_()

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (world_size, triton.cdiv(max_s, BLOCK_M), triton.cdiv(max_f, BLOCK_N))
    _resolve_full_res_kernel[grid](
        out,
        c_all,
        z_all,
        r_all,
        shape_all,
        S=S,
        F=F,
        max_s=max_s,
        max_f=max_f,
        stride_c0=c_all.stride(0),
        stride_c1=c_all.stride(1),
        stride_z0=z_all.stride(0),
        stride_z1=z_all.stride(1),
        stride_v0=r_all.stride(0),
        stride_v1=r_all.stride(1),
        stride_v2=r_all.stride(2),
        stride_shape0=shape_all.stride(0),
        stride_shape1=shape_all.stride(1),
        stride_out0=out.stride(0),
        stride_out1=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out