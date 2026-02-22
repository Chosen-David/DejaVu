#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import torch
import numpy as np

class SystemCommProfiler:
    def __init__(self, nccl_test_path: str, timeout: int = 30):
        self.nccl_test_path = Path(nccl_test_path)
        self.timeout = timeout

        if not self.nccl_test_path.exists():
            raise FileNotFoundError(f"NCCL test not found: {nccl_test_path}")

        # 动态获取 GPU 数量
        self.n_gpus = torch.cuda.device_count()
        if self.n_gpus <= 0:
            raise RuntimeError("No GPUs available")

        # 存放采样数据
        self.samples = []

    def _run_nccl_test(self, bytes_size: int):
        """
        Run a single all_reduce_perf call and return stdout.
        """
        cmd = [
            str(self.nccl_test_path),
            "-b", str(bytes_size),
            "-e", str(bytes_size),
            "-f", "2",
            "-g", str(self.n_gpus)
        ]
        print(f"[CommProfiler] Running: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"NCCL test failed:\n{proc.stderr}")
        return proc.stdout

    def _parse_output(self, output: str):
        """
        Parse a single all_reduce_perf output and return a triple:
        (time_us, algbw_GBs, busbw_GBs)
        """
        for line in output.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit():
                try:
                    time_us  = float(parts[-3])
                    algbw    = float(parts[-2])
                    busbw    = float(parts[-1])
                    return time_us, algbw, busbw
                except:
                    pass
        raise RuntimeError("Failed to parse output")

    def profile_allreduce(self, sizes):
        """
        Profile a list of message sizes and store in self.samples.
        """
        self.samples.clear()
        for size in sizes:
            out = self._run_nccl_test(size)
            time_us, algbw, busbw = self._parse_output(out)
            self.samples.append((size, time_us, algbw, busbw))
            print(f" size={size}, time={time_us:.2f}us, algbw={algbw:.2f}GB/s, busbw={busbw:.2f}GB/s")
        return self.samples

    def fit_alpha_beta(self):
        """
        Fit the alpha-beta model:
            time (us) ≈ alpha + size / bandwidth
        
        自动智能 fallback:
        - 如果 busbw 全为 0，则使用 algbw
        - 否则优先使用 busbw
        """
        if not self.samples:
            raise RuntimeError("No samples to fit")

        # 检查 busbw 是否有效（是否全为 0）
        all_busbw_zero = all(b == 0.0 for _, _, _, b in self.samples)
        use_busbw = not all_busbw_zero
        mode = "busbw" if use_busbw else "algbw"
        print(f"[CommProfiler] Using '{mode}' for β fitting")
        X = []
        y = []
        for size, time_us, algbw, busbw in self.samples:
            # select bandwidth to use
            bw = busbw if use_busbw else algbw

            # skip invalid small/zero bandwidth
            if bw <= 0:
                continue

            # convert bandwidth GB/s -> B/us
            bw_b_per_us = bw * 1e3

            X.append([1.0, size / bw_b_per_us])
            y.append(time_us)

        if len(X) < 2:
            raise RuntimeError("Not enough valid samples to fit")

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # solve for [alpha, coefficient]
        solution, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha = solution[0]
        inv_beta = solution[1]

        # beta = 1 / inv_beta
        beta = 1.0 / inv_beta if inv_beta != 0 else 0.0
        return alpha, beta

    def predict(self, size, alpha, beta):
        """
        Predict time using fitted alpha and beta.
        beta should be in GB/s.
        """
        # convert beta to B/us
        bw_b_per_us = beta * 1e3
        return alpha + size / bw_b_per_us

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Communication Profiler α-β model")
    parser.add_argument(
        "--nccl-test-path",
        type=str,
        default="thirty-part/nccl-test/build/all_reduce_perf",
        help="Path to nccl all_reduce_perf binary",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[256, 1024, 8*1024, 64*1024, 512*1024, 1*1024*1024, 4*1024*1024],
        help="Message sizes to benchmark",
    )

    args = parser.parse_args()

    profiler = SystemCommProfiler(args.nccl_test_path)
    print(f"\nDetected GPUs: {profiler.n_gpus}\n")

    # Do the benchmarks
    profiler.profile_allreduce(args.sizes)

    # Fit α-β model using busbw
    alpha, inv_beta = profiler.fit_alpha_beta()
    beta = 1.0 / inv_beta
    print(f"\nFitted model (busbw):")
    print(f"  alpha (latency) = {alpha:.3f} us")
    print(f"  beta  (bandwidth) = {beta:.3f} GB/s")

    # Show predictions
    print("\nPredicted times:")
    for size in args.sizes:
        t_pred = profiler.predict(size, alpha, beta)
        print(f"  size={size:8d}, predicted time = {t_pred:.2f} us")
        
        
