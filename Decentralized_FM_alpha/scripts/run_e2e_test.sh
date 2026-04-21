#!/bin/bash

# End-to-End Test Runner for Sparse TP FFN

set -e

echo "=========================================="
echo "Sparse TP FFN - End-to-End Test Runner"
echo "=========================================="

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Run tests
echo ""
echo "Running unit tests..."
python -m tests.test_sparse_tp_ffn

echo ""
echo "Running integration tests..."
python -m tests.test_e2e_integration

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
