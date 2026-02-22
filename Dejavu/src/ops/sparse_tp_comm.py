from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import fused_dense_lib as fused_dense_cuda

from flash_attn.ops.gelu_activation import gelu_bwd
from flash_attn.utils.distributed import (
    all_gather_raw,
    reduce_scatter_raw,
    all_reduce_raw,
)
from flash_attn.utils.distributed import reduce_scatter, all_reduce



TensorLike = Union[np.ndarray, torch.Tensor]


@dataclass
class ResShard:
    k: int
    i: int
    C_k: TensorLike  # shape = [S_loc], dtype=int32
    Res_vals: TensorLike  # shape = [S_loc, F_loc]
    Z_i: TensorLike  # shape = [F_loc], dtype=int32

    @property
    def S_loc(self) -> int:
        return self.C_k.shape[0]

    @property
    def F_loc(self) -> int:
        return self.Z_i.shape[0]


def is_sparse_tp_comm_enabled() -> bool:
    return envs.SGLANG_ENABLE_SPARSE_TP_COMM.get()


def _to_device_tensor(
    x: TensorLike,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, non_blocking=True)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype=dtype)
        return t.contiguous()
    t = torch.as_tensor(x, device=device)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t.contiguous()


def all_gather_res_shard(
    local_shard: ResShard,
    tp_group: Optional[GroupCoordinator] = None,
) -> List[ResShard]:
    group = tp_group or get_tp_group()
    device = group.device
    world_size = group.world_size

    C_local = _to_device_tensor(local_shard.C_k, device=device, dtype=torch.int32)
    Z_local = _to_device_tensor(local_shard.Z_i, device=device, dtype=torch.int32)
    Res_local = _to_device_tensor(local_shard.Res_vals, device=device)

    # shape 检查
    if C_local.dim() != 1 or Z_local.dim() != 1 or Res_local.dim() != 2:
        raise ValueError("ResShard shapes must be C_k=[S_loc], Z_i=[F_loc], Res_vals=[S_loc,F_loc]")
    if Res_local.shape != (C_local.numel(), Z_local.numel()):
        raise ValueError("Res_vals shape mismatch")

    # allocate output buffers for all_gather
    all_C = torch.empty((world_size, C_local.shape[0]),
                        dtype=C_local.dtype, device=device)
    all_Z = torch.empty((world_size, Z_local.shape[0]),
                        dtype=Z_local.dtype, device=device)
    all_R = torch.empty((world_size, Res_local.shape[0], Res_local.shape[1]),
                        dtype=Res_local.dtype, device=device)

    # all_gather into fixed-size buffers
    dist.all_gather_into_tensor(all_C, C_local, group=group.device_group)
    dist.all_gather_into_tensor(all_Z, Z_local, group=group.device_group)
    dist.all_gather_into_tensor(all_R, Res_local, group=group.device_group)

    # gather k/i identifiers
    ki_local = torch.tensor([local_shard.k, local_shard.i], dtype=torch.int32, device=device)
    ki_all = torch.empty((world_size, 2), dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(ki_all, ki_local, group=group.device_group)

    # construct ResShard list
    shards: List[ResShard] = []
    for r in range(world_size):
        shards.append(
            ResShard(
                k=int(ki_all[r, 0].item()),
                i=int(ki_all[r, 1].item()),
                C_k=all_C[r].clone(),
                Res_vals=all_R[r].clone(),
                Z_i=all_Z[r].clone(),
            )
        )

    return shards


# # 用于triton运行失败的备选
# def resolve_full_res(
#     shards: Sequence[ResShard],
#     S: int,
#     F: int,
#     *,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
# ) -> torch.Tensor:
#     if len(shards) == 0:
#         raise ValueError("No shards provided")
#     if device is None:
#         first = shards[0].Res_vals
#         if isinstance(first, torch.Tensor):
#             device = first.device
#         else:
#             device = torch.device("cpu")

#     if dtype is None:
#         first = shards[0].Res_vals
#         if isinstance(first, torch.Tensor):
#             dtype = first.dtype
#         else:
#             dtype = torch.float32

#     out = torch.zeros((S, F), dtype=dtype, device=device)
#     for shard in shards:
#         rows = _to_device_tensor(shard.C_k, device=device, dtype=torch.int64)
#         cols = _to_device_tensor(shard.Z_i, device=device, dtype=torch.int64)
#         vals = _to_device_tensor(shard.Res_vals, device=device, dtype=dtype)
#         if vals.shape != (rows.numel(), cols.numel()):
#             raise ValueError(
#                 "Shard value/index shape mismatch, got "
#                 f"vals={tuple(vals.shape)} rows={rows.numel()} cols={cols.numel()}"
#             )
#         out[rows[:, None], cols[None, :]] = vals
#     return out


def _resolve_full_res_from_packed(
    shape_all: torch.Tensor,
    c_all: torch.Tensor,
    z_all: torch.Tensor,
    r_all: torch.Tensor,
    S: int,
    F: int,
) -> torch.Tensor:
    if envs.SGLANG_USE_TRITON_SPARSE_TP_RESOLVE.get():
        try:
            from .triton.triton_sparse_tp import resolve_full_res_packed_triton

            return resolve_full_res_packed_triton(c_all, z_all, r_all, shape_all, S=S, F=F)
        except Exception as e:
            print(f"[WARNING] triton resolve failed: {e}, 直接报错")
            # fallback 到 Python 版本

    # shards: List[ResShard] = []
    # world_size = shape_all.shape[0]
    # for r in range(world_size):
    #     s_loc = int(shape_all[r, 0].item())
    #     f_loc = int(shape_all[r, 1].item())
    #     shards.append(
    #         ResShard(
    #             k=0,
    #             i=r,
    #             C_k=c_all[r, :s_loc],
    #             Res_vals=r_all[r, :s_loc, :f_loc],
    #             Z_i=z_all[r, :f_loc],
    #         )
    #     )
    # return resolve_full_res(shards, S=S, F=F)


def all_gather_and_resolve_res_shard(
    local_shard: ResShard,
    S: int,
    F: int,
    tp_group: Optional[GroupCoordinator] = None,
) -> torch.Tensor:
    shape_all, c_all, z_all, r_all, _ = _all_gather_res_shard_packed(
        local_shard, tp_group=tp_group
    )
    return _resolve_full_res_from_packed(shape_all, c_all, z_all, r_all, S=S, F=F)