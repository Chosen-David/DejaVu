"""
Communication modules for sparse-aware tensor parallelism.
"""

from .sparse_allreduce import (
    SparseAllReducer,
    ChunkedSparseAllReducer,
    AdaptiveSparseAllReducer,
    SparseTensor,
    create_sparse_communicator,
    estimate_sparse_communication_benefit,
)

__all__ = [
    'SparseAllReducer',
    'ChunkedSparseAllReducer',
    'AdaptiveSparseAllReducer',
    'SparseTensor',
    'create_sparse_communicator',
    'estimate_sparse_communication_benefit',
]
