#!/usr/bin/env python3
"""Synthetic LLaMA-shape TC/CC FFN benchmark.

This benchmark does not download model weights. It uses LLaMA-family FFN
dimensions and random fp16 weights to evaluate the runtime mask pre-traversal,
TC/CC tile scheduling, and FlashFFN execution paths.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from kernels.flash_ffn import FlashFFN
from runtime.sparse_tp_ffn import SparseTPFFN, SparseTPFFNConfig


MODEL_SHAPES = {
    "llama-7b": (4096, 11008),
    "llama-13b": (5120, 13824),
    "llama2-7b": (4096, 11008),
    "llama2-13b": (5120, 13824),
    "llama3-8b": (4096, 14336),
    "llama2-70b": (8192, 28672),
}


def cuda_time_ms(fn, warmup: int, iters: int):
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, y


def make_mask(seq_len, intermediate_dim, num_hans, active_lans, device, dtype):
    mask = torch.ones((seq_len, intermediate_dim), device=device, dtype=dtype)
    mask[:, num_hans:] = 0
    num_lans = intermediate_dim - num_hans
    active_lans = min(active_lans, num_lans)
    for token in range(seq_len):
        if active_lans == 0:
            continue
        indices = torch.randperm(num_lans, device=device)[:active_lans] + num_hans
        mask[token, indices] = 1
    return mask


def run_case(model_name, hidden_dim, intermediate_dim, seq_len, active_lans, hans_ratio, warmup, iters):
    device = "cuda"
    dtype = torch.float16
    num_hans = max(1, min(intermediate_dim - 1, int(intermediate_dim * hans_ratio)))
    num_lans = intermediate_dim - num_hans

    torch.manual_seed(1234 + hidden_dim + intermediate_dim + seq_len + active_lans)
    x = torch.randn((seq_len, hidden_dim), device=device, dtype=dtype)
    ffn = FlashFFN(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        hans_indices=torch.arange(num_hans),
        lans_indices=torch.arange(num_hans, intermediate_dim),
        activation="gelu",
        bias=True,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        ffn.w1.normal_(mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        ffn.w2.normal_(mean=0.0, std=1.0 / math.sqrt(intermediate_dim))
        ffn.bias1.zero_()
        ffn.bias2.zero_()

    runtime = SparseTPFFN(SparseTPFFNConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        activation="gelu",
        use_sparse_comm=False,
    ))

    mask = make_mask(seq_len, intermediate_dim, num_hans, active_lans, device, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    plan = SparseTPFFN.build_lans_active_tile_plan(mask, num_hans, cc_token_tile=1)
    schedule = runtime.tc_cc_balancer.build_tc_cc_tile_schedule(plan, num_hans=num_hans)
    torch.cuda.synchronize()
    pre_ms = (time.perf_counter() - t0) * 1000

    dense_ms, dense_y = cuda_time_ms(lambda: ffn(x, mask), warmup, iters)
    tile_ms, tile_y = cuda_time_ms(
        lambda: ffn(x, {"mask": mask, "active_lans_by_tile": plan}),
        warmup,
        iters,
    )
    err = (tile_y - dense_y).abs().max().item()

    return {
        "model": model_name,
        "hidden": hidden_dim,
        "intermediate": intermediate_dim,
        "seq": seq_len,
        "num_hans": num_hans,
        "num_lans": num_lans,
        "active_lans_per_token": active_lans,
        "pretraverse_ms": pre_ms,
        "dense_mask_ms": dense_ms,
        "tc_cc_tile_ms": tile_ms,
        "speedup_vs_dense": dense_ms / tile_ms if tile_ms > 0 else 0.0,
        "mean_active_lans": schedule.mean_active_lans,
        "max_active_lans": schedule.max_active_lans,
        "mean_tc_token_tile": schedule.mean_tc_token_tile,
        "mean_overlap_efficiency": schedule.mean_overlap_efficiency,
        "max_err": err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama-7b", "llama3-8b"])
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--active-lans", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--hans-ratio", type=float, default=0.75)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=str, default="/tmp/llama_tc_cc_results.csv")
    args = parser.parse_args()

    rows = []
    for model in args.models:
        if model not in MODEL_SHAPES:
            raise ValueError(f"Unknown model shape: {model}")
        hidden, intermediate = MODEL_SHAPES[model]
        for seq_len in args.seq:
            for active_lans in args.active_lans:
                row = run_case(
                    model, hidden, intermediate, seq_len, active_lans,
                    args.hans_ratio, args.warmup, args.iters,
                )
                rows.append(row)
                print(",".join(f"{k}={v}" for k, v in row.items()))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
