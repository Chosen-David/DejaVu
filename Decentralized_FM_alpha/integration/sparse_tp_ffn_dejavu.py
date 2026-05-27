"""
Integration with DejaVu Framework

This module provides DejaVu-compatible sparse FFN implementation
that integrates with the existing DejaVu architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import os

# Import from solver
from solver import (
    NeuronPartitioner,
    GPUBalancer,
    TCCCBalancer,
    PipelineScheduler,
    AnalysisResultStore,
    create_analysis_result_from_partitioner,
)

# Import from kernels
from kernels import FlashFFN

# Import from communication
from communication import create_sparse_communicator


class ParallelSparseTPMLP(nn.Module):
    """
    Sparse Tensor Parallel MLP integrated with DejaVu.
    
    This module replaces the original ParallelFusedMLPDejavu with
    sparse-aware optimizations.
    
    Features:
    - Compatible with DejaVu's predictor interface
    - Supports tensor parallelism with sequence parallel
    - Integrates offline analysis results
    - Hybrid TC/CC execution
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        layer_idx: int,
        process_group,
        model_name: str = "default",
        activation: str = "gelu",
        sequence_parallel: bool = True,
        store_root: str = "./analysis_results",
        use_sparse_comm: bool = True,
        activation_threshold: float = 0.1,
        target_sparsity: float = 0.5,
    ):
        """
        Initialize Sparse TP MLP.
        
        Args:
            hidden_dim: Model hidden dimension
            intermediate_dim: FFN intermediate dimension
            layer_idx: Layer index
            process_group: Process group for tensor parallelism
            model_name: Model name for storing analysis results
            activation: Activation function
            sequence_parallel: Whether to use sequence parallelism
            store_root: Root directory for analysis results
            use_sparse_comm: Whether to use sparse communication
            activation_threshold: Threshold for HANS/LANS classification
            target_sparsity: Target sparsity level
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.layer_idx = layer_idx
        self.process_group = process_group
        self.model_name = model_name
        self.activation = activation
        self.sequence_parallel = sequence_parallel
        self.use_sparse_comm = use_sparse_comm
        self.activation_threshold = activation_threshold
        self.target_sparsity = target_sparsity
        
        self.world_size = process_group.size()
        self.rank = process_group.rank()
        
        # Initialize solvers
        self.neuron_partitioner = NeuronPartitioner(
            num_neurons=intermediate_dim,
            num_gpus=self.world_size,
            activation_threshold=activation_threshold,
        )
        
        self.gpu_balancer = GPUBalancer(
            num_gpus=self.world_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        
        self.tc_cc_balancer = TCCCBalancer(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        
        self.pipeline_scheduler = PipelineScheduler(
            num_gpus=self.world_size,
            hidden_dim=hidden_dim,
        )
        
        # Result store
        self.result_store = AnalysisResultStore(store_root)
        
        # Weights
        # Note: Weights are split across GPUs
        self.intermediate_dim_per_gpu = intermediate_dim // self.world_size
        
        # These will be reordered after offline analysis
        self.fc1 = nn.Parameter(
            torch.empty(hidden_dim, self.intermediate_dim_per_gpu)
        )
        self.fc2 = nn.Parameter(
            torch.empty(self.intermediate_dim_per_gpu, hidden_dim)
        )
        
        # Partition info (loaded from offline analysis)
        self.hans_indices = None
        self.lans_indices = None
        self.tc_indices = None
        self.cc_indices = None
        
        # FlashFFN kernel
        self.flash_ffn = None
        
        # Communication
        if use_sparse_comm and torch.distributed.is_initialized():
            self.communicator = create_sparse_communicator(
                hidden_dim=hidden_dim,
                num_gpus=self.world_size,
                rank=self.rank,
                mode='adaptive',
            )
        else:
            self.communicator = None
        
        # Analysis state
        self.is_analyzed = False
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.fc1, a=5**0.5)
        nn.init.kaiming_uniform_(self.fc2, a=5**0.5)
    
    def offline_analysis(
        self,
        sample_masks: np.ndarray,
        force_reanalyze: bool = False,
    ) -> Dict:
        """
        Perform offline analysis or load existing results.
        
        Args:
            sample_masks: Sample activation masks [num_samples, intermediate_dim]
            force_reanalyze: Force reanalysis even if cached results exist
            
        Returns:
            Analysis results
        """
        # Check if cached result exists
        if not force_reanalyze:
            cached_result = self.result_store.load_result(
                model_name=self.model_name,
                layer_id=self.layer_idx,
            )
            
            cache_matches = (
                cached_result is not None
                and cached_result.hidden_dim == self.hidden_dim
                and cached_result.intermediate_dim == self.intermediate_dim
                and cached_result.num_gpus == self.world_size
                and len(cached_result.partition_results) > self.rank
                and len(cached_result.tc_cc_assignments) > self.rank
            )

            if cache_matches:
                print(f"[Layer {self.layer_idx}] Loading cached analysis results")
                self._load_from_result(cached_result)
                return {'status': 'loaded', 'layer_id': self.layer_idx}
        
        # Perform analysis
        print(f"[Layer {self.layer_idx}] Performing offline analysis...")
        import time
        start_time = time.time()
        
        # Step 1: Partition neurons
        partitions = self.neuron_partitioner.analyze_and_partition(
            masks=sample_masks,
            method='spectral',
            balance_strategy='adaptive',
        )
        
        # Step 2: Balance across GPUs
        analyzer = self.neuron_partitioner.analyzer
        coactivation_weights = analyzer.compute_coactivation_weights()
        activation_frequencies = analyzer.compute_activation_frequencies()
        
        balanced_partitions, _ = self.gpu_balancer.balance_partitions(
            partitions=partitions,
            coactivation_weights=coactivation_weights,
            activation_frequencies=activation_frequencies,
            strategy='hybrid',
        )
        
        # Step 3: TC/CC balancing for this GPU
        partition = balanced_partitions[self.rank]
        tc_cc_assignment = self.tc_cc_balancer.balance(
            hans_indices=partition.hans_indices,
            lans_indices=partition.lans_indices,
            activation_frequencies=activation_frequencies,
            sequence_length=2048,
            search_method='heuristic',
        )
        
        # Store indices
        self.hans_indices = torch.from_numpy(partition.hans_indices)
        self.lans_indices = torch.from_numpy(partition.lans_indices)
        self.tc_indices = torch.from_numpy(tc_cc_assignment.tc_indices)
        self.cc_indices = torch.from_numpy(tc_cc_assignment.cc_indices)
        
        # Reorder weights
        self._reorder_weights()
        
        # Initialize FlashFFN
        self._init_flash_ffn()
        
        # Save results
        from solver import TCCCAssignmentResult
        
        tc_cc_results = []
        for i, p in enumerate(balanced_partitions):
            if i == self.rank:
                assignment = tc_cc_assignment
            else:
                # For other GPUs, we would need to compute their assignments
                # For now, use placeholder
                assignment = tc_cc_assignment
            
            tc_cc_results.append(TCCCAssignmentResult(
                gpu_id=i,
                tc_indices=assignment.tc_indices.tolist(),
                cc_indices=assignment.cc_indices.tolist(),
                tc_time_ms=assignment.tc_time,
                cc_time_ms=assignment.cc_time,
                overlap_efficiency=assignment.overlap_efficiency,
            ))
        
        result = create_analysis_result_from_partitioner(
            layer_id=self.layer_idx,
            model_name=self.model_name,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_gpus=self.world_size,
            partitioner=self.neuron_partitioner,
            tc_cc_assignments=tc_cc_results,
            analysis_time_s=time.time() - start_time,
        )
        
        # Save to store
        self.result_store.save_result(
            result=result,
            coactivation_matrix=coactivation_weights,
            activation_frequencies=activation_frequencies,
        )
        
        self.is_analyzed = True
        
        return {
            'status': 'analyzed',
            'layer_id': self.layer_idx,
            'analysis_time_s': result.analysis_time_s,
            'total_hans': result.total_hans,
            'total_lans': result.total_lans,
        }
    
    def _load_from_result(self, result):
        """Load partition info from analysis result."""
        partition = result.partition_results[self.rank]
        tc_cc = result.tc_cc_assignments[self.rank]
        
        self.hans_indices = torch.tensor(partition.hans_indices)
        self.lans_indices = torch.tensor(partition.lans_indices)
        self.tc_indices = torch.tensor(tc_cc.tc_indices)
        self.cc_indices = torch.tensor(tc_cc.cc_indices)
        
        self._reorder_weights()
        self._init_flash_ffn()
        self.is_analyzed = True
    
    def _reorder_weights(self):
        """Reorder weights to match TC/CC assignment."""
        if self.tc_indices is None or self.cc_indices is None:
            return
        
        # Combine TC and CC indices
        new_order = torch.cat([self.tc_indices, self.cc_indices])
        if len(new_order) == 0 or new_order.max().item() >= self.intermediate_dim_per_gpu:
            # Partitions are expressed in global neuron ids while fc1/fc2 are
            # already local TP shards. Keep the local shard order for smoke
            # experiments unless a local index mapping is available.
            return
        
        # Reorder fc1 columns
        self.fc1.data = self.fc1.data[:, new_order[:self.intermediate_dim_per_gpu]]
        
        # Reorder fc2 rows
        self.fc2.data = self.fc2.data[new_order[:self.intermediate_dim_per_gpu], :]
    
    def _init_flash_ffn(self):
        """Initialize FlashFFN kernel."""
        if self.tc_indices is None:
            return
        
        local_intermediate_dim = self.fc1.shape[1]
        num_tc = min(len(self.tc_indices), local_intermediate_dim)
        num_cc = max(0, local_intermediate_dim - num_tc)
        
        if num_tc == 0 and num_cc == 0:
            num_tc = local_intermediate_dim
        
        device = self.fc1.device
        local_tc_idx = torch.arange(num_tc, dtype=torch.long, device=device)
        local_cc_idx = torch.arange(num_tc, num_tc + num_cc, dtype=torch.long, device=device)
        
        self.flash_ffn = FlashFFN(
            hidden_dim=self.hidden_dim,
            intermediate_dim=num_tc + num_cc,
            hans_indices=local_tc_idx,
            lans_indices=local_cc_idx,
            activation=self.activation,
        ).to(device=device, dtype=self.fc1.dtype)

        with torch.no_grad():
            self.flash_ffn.w1.copy_(self.fc1[:, :num_tc + num_cc])
            self.flash_ffn.w2.copy_(self.fc2[:num_tc + num_cc, :])
    
    def forward(
        self,
        x: torch.Tensor,
        predictor_output: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq, hidden] or [seq, hidden]
            predictor_output: Output from predictor (sparse mask or logits)
            
        Returns:
            Output tensor
        """
        # Handle different input shapes
        input_shape = x.shape
        if len(input_shape) == 3:
            batch, seq_len, hidden = input_shape
            x = x.reshape(batch * seq_len, hidden)
        else:
            seq_len = input_shape[0]
        
        active_lans_indices = None
        active_lans_by_tile = None
        if isinstance(predictor_output, dict):
            active_lans_indices = predictor_output.get('active_lans_indices')
            active_lans_by_tile = predictor_output.get('active_lans_by_tile')
            predictor_payload = predictor_output.get('mask')
            if predictor_payload is None:
                predictor_payload = predictor_output.get('prediction_mask')
            if predictor_payload is None:
                predictor_payload = predictor_output.get('activation_mask')
            predictor_output = predictor_payload
        elif isinstance(predictor_output, (tuple, list)):
            if len(predictor_output) == 0:
                predictor_output = None
            else:
                active_lans_indices = predictor_output[1] if len(predictor_output) > 1 else None
                active_lans_by_tile = predictor_output[2] if len(predictor_output) > 2 else None
                predictor_output = predictor_output[0]

        # Get mask from predictor
        if predictor_output is not None:
            # predictor_output shape: [batch, seq, intermediate_per_gpu] or [seq, intermediate_per_gpu]
            if len(predictor_output.shape) == 3:
                predictor_output = predictor_output.reshape(-1, predictor_output.shape[-1])
            
            # Convert logits to mask if needed
            if predictor_output.dtype in [torch.float16, torch.float32]:
                if predictor_output.min() >= 0 and predictor_output.max() <= 1:
                    mask = predictor_output.float()
                else:
                    mask = (torch.sigmoid(predictor_output) > 0.5).float()
            else:
                mask = predictor_output.float()
        else:
            mask = None

        if active_lans_by_tile is not None:
            mask_payload = {
                'mask': mask,
                'active_lans_by_tile': active_lans_by_tile,
            }
        elif active_lans_indices is not None:
            mask_payload = (mask, active_lans_indices)
        else:
            mask_payload = mask
        
        # Compute
        if self.flash_ffn is not None and self.is_analyzed:
            output = self.flash_ffn(x, mask_payload)
        else:
            # Fallback to standard computation
            output = self._standard_forward(x)
        
        # Reshape output
        if len(input_shape) == 3:
            output = output.reshape(batch, seq_len, hidden)
        
        return output
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard FFN forward pass."""
        fc1 = self.fc1.to(dtype=x.dtype)
        fc2 = self.fc2.to(dtype=x.dtype)
        intermediate = F.linear(x, fc1.t())
        
        if self.activation == "gelu":
            intermediate = F.gelu(intermediate)
        elif self.activation == "relu":
            intermediate = F.relu(intermediate)
        elif self.activation == "silu":
            intermediate = F.silu(intermediate)
        
        output = F.linear(intermediate, fc2.t())
        return output


def convert_dejavu_mlp_to_sparse_tp(
    original_mlp,
    layer_idx: int,
    model_name: str,
    store_root: str = "./analysis_results",
) -> ParallelSparseTPMLP:
    """
    Convert original DejaVu MLP to Sparse TP MLP.
    
    Args:
        original_mlp: Original ParallelFusedMLPDejavu instance
        layer_idx: Layer index
        model_name: Model name
        store_root: Root directory for analysis results
        
    Returns:
        Converted ParallelSparseTPMLP instance
    """
    # Extract configuration from original MLP
    hidden_dim = original_mlp.fc1.in_features
    intermediate_dim = original_mlp.fc1.out_features
    process_group = original_mlp.process_group
    activation = getattr(original_mlp, 'activation', 'gelu')
    sequence_parallel = getattr(original_mlp, 'sequence_parallel', True)
    
    # Create new MLP
    new_mlp = ParallelSparseTPMLP(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        layer_idx=layer_idx,
        process_group=process_group,
        model_name=model_name,
        activation=activation,
        sequence_parallel=sequence_parallel,
        store_root=store_root,
    )
    
    # Copy weights
    with torch.no_grad():
        new_mlp.fc1.copy_(original_mlp.fc1.weight)
        new_mlp.fc2.copy_(original_mlp.fc2.weight)
    
    return new_mlp
