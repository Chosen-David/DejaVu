"""
Runtime system for sparse-aware tensor parallel FFN.
"""

from .sparse_tp_ffn import (
    SparseTPFFN,
    SparseTPFFNConfig,
    create_sparse_tp_ffn,
)

__all__ = [
    'SparseTPFFN',
    'SparseTPFFNConfig',
    'create_sparse_tp_ffn',
]
