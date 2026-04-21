"""
Integration module for DejaVu framework.
"""

from .sparse_tp_ffn_dejavu import (
    ParallelSparseTPMLP,
    convert_dejavu_mlp_to_sparse_tp,
)

__all__ = [
    'ParallelSparseTPMLP',
    'convert_dejavu_mlp_to_sparse_tp',
]
