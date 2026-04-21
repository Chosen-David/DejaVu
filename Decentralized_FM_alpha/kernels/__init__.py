"""
High-performance CUDA/Triton kernels for sparse FFN computation.
"""

from .flash_ffn import (
    FlashFFN,
    flash_ffn_hans,
    flash_ffn_lans,
    benchmark_flash_ffn,
)

__all__ = [
    'FlashFFN',
    'flash_ffn_hans',
    'flash_ffn_lans',
    'benchmark_flash_ffn',
]
