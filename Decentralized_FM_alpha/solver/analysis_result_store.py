"""
Analysis Result Store for Offline Optimization

This module provides persistent storage for offline analysis results,
including neuron partitions, TC/CC assignments, and optimization configurations.

Storage format: JSON + NPY (for large arrays)
Storage location: configurable, default under model checkpoint directory
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import tempfile


def _json_safe(value):
    """Convert numpy scalar/list values to standard JSON-serializable Python types."""
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _atomic_save_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _atomic_save_npy(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.npy",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'wb') as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@dataclass
class PartitionResult:
    """Neuron partition result for a single GPU."""
    gpu_id: int
    hans_indices: List[int]
    lans_indices: List[int]
    hans_count: int
    lans_count: int
    
    def to_dict(self) -> Dict:
        return _json_safe({
            'gpu_id': self.gpu_id,
            'hans_indices': self.hans_indices,
            'lans_indices': self.lans_indices,
            'hans_count': self.hans_count,
            'lans_count': self.lans_count,
        })
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PartitionResult':
        return cls(
            gpu_id=data['gpu_id'],
            hans_indices=data['hans_indices'],
            lans_indices=data['lans_indices'],
            hans_count=data['hans_count'],
            lans_count=data['lans_count'],
        )


@dataclass
class TCCCAssignmentResult:
    """TC/CC assignment result for a single GPU."""
    gpu_id: int
    tc_indices: List[int]
    cc_indices: List[int]
    tc_time_ms: float
    cc_time_ms: float
    overlap_efficiency: float
    
    def to_dict(self) -> Dict:
        return _json_safe({
            'gpu_id': self.gpu_id,
            'tc_indices': self.tc_indices,
            'cc_indices': self.cc_indices,
            'tc_time_ms': self.tc_time_ms,
            'cc_time_ms': self.cc_time_ms,
            'overlap_efficiency': self.overlap_efficiency,
        })
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TCCCAssignmentResult':
        return cls(
            gpu_id=data['gpu_id'],
            tc_indices=data['tc_indices'],
            cc_indices=data['cc_indices'],
            tc_time_ms=data['tc_time_ms'],
            cc_time_ms=data['cc_time_ms'],
            overlap_efficiency=data['overlap_efficiency'],
        )


@dataclass
class AnalysisResult:
    """
    Complete analysis result for a single layer.
    
    Contains all information needed for runtime sparse FFN execution.
    """
    # Metadata
    layer_id: int
    model_name: str
    timestamp: str
    config_hash: str  # Hash of model config for validation
    
    # Dimensions
    hidden_dim: int
    intermediate_dim: int
    num_gpus: int
    
    # Analysis results
    partition_results: List[PartitionResult]
    tc_cc_assignments: List[TCCCAssignmentResult]
    
    # Statistics
    total_hans: int
    total_lans: int
    mean_activation_frequency: float
    analysis_time_s: float
    
    # Co-activation statistics (stored separately as NPY)
    coactivation_matrix_path: Optional[str] = None
    activation_frequencies_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return _json_safe({
            'layer_id': self.layer_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'config_hash': self.config_hash,
            'hidden_dim': self.hidden_dim,
            'intermediate_dim': self.intermediate_dim,
            'num_gpus': self.num_gpus,
            'partition_results': [p.to_dict() for p in self.partition_results],
            'tc_cc_assignments': [a.to_dict() for a in self.tc_cc_assignments],
            'total_hans': self.total_hans,
            'total_lans': self.total_lans,
            'mean_activation_frequency': self.mean_activation_frequency,
            'analysis_time_s': self.analysis_time_s,
            'coactivation_matrix_path': self.coactivation_matrix_path,
            'activation_frequencies_path': self.activation_frequencies_path,
        })
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisResult':
        return cls(
            layer_id=data['layer_id'],
            model_name=data['model_name'],
            timestamp=data['timestamp'],
            config_hash=data['config_hash'],
            hidden_dim=data['hidden_dim'],
            intermediate_dim=data['intermediate_dim'],
            num_gpus=data['num_gpus'],
            partition_results=[PartitionResult.from_dict(p) for p in data['partition_results']],
            tc_cc_assignments=[TCCCAssignmentResult.from_dict(a) for a in data['tc_cc_assignments']],
            total_hans=data['total_hans'],
            total_lans=data['total_lans'],
            mean_activation_frequency=data['mean_activation_frequency'],
            analysis_time_s=data['analysis_time_s'],
            coactivation_matrix_path=data.get('coactivation_matrix_path'),
            activation_frequencies_path=data.get('activation_frequencies_path'),
        )


class AnalysisResultStore:
    """
    Persistent storage manager for analysis results.
    
    Directory structure:
    store_root/
    ├── model_name/
    │   ├── config.json                    # Model configuration
    │   ├── layer_0/
    │   │   ├── result.json                # Analysis result metadata
    │   │   ├── coactivation_matrix.npy    # Co-activation matrix
    │   │   ├── activation_frequencies.npy # Activation frequencies
    │   │   └── partition_weights/         # Reordered weights (optional)
    │   ├── layer_1/
    │   │   └── ...
    │   └── ...
    └── index.json                         # Global index
    """
    
    def __init__(self, store_root: str = "./analysis_results"):
        """
        Initialize result store.
        
        Args:
            store_root: Root directory for storing results
        """
        self.store_root = Path(store_root)
        self.store_root.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self.index_path = self.store_root / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load global index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {'models': {}}
    
    def _save_index(self):
        """Save global index."""
        _atomic_save_json(self.index_path, self.index)
    
    def save_result(
        self,
        result: AnalysisResult,
        coactivation_matrix: Optional[np.ndarray] = None,
        activation_frequencies: Optional[np.ndarray] = None,
    ) -> str:
        """
        Save analysis result to disk.
        
        Args:
            result: Analysis result to save
            coactivation_matrix: Optional co-activation matrix [F, F]
            activation_frequencies: Optional activation frequencies [F]
            
        Returns:
            Path to saved result directory
        """
        # Create layer directory
        model_dir = self.store_root / result.model_name
        layer_dir = model_dir / f"layer_{result.layer_id}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save co-activation matrix
        if coactivation_matrix is not None:
            cw_path = layer_dir / "coactivation_matrix.npy"
            _atomic_save_npy(cw_path, coactivation_matrix)
            result.coactivation_matrix_path = str(cw_path)
        
        # Save activation frequencies
        if activation_frequencies is not None:
            freq_path = layer_dir / "activation_frequencies.npy"
            _atomic_save_npy(freq_path, activation_frequencies)
            result.activation_frequencies_path = str(freq_path)
        
        # Save result metadata
        result_path = layer_dir / "result.json"
        _atomic_save_json(result_path, result.to_dict())
        
        # Update index
        if result.model_name not in self.index['models']:
            self.index['models'][result.model_name] = {
                'hidden_dim': result.hidden_dim,
                'intermediate_dim': result.intermediate_dim,
                'num_gpus': result.num_gpus,
                'layers': {}
            }
        
        self.index['models'][result.model_name]['layers'][str(result.layer_id)] = {
            'timestamp': result.timestamp,
            'total_hans': result.total_hans,
            'total_lans': result.total_lans,
            'analysis_time_s': result.analysis_time_s,
        }
        
        self._save_index()
        
        return str(layer_dir)
    
    def load_result(
        self,
        model_name: str,
        layer_id: int,
        load_matrices: bool = False,
    ) -> Optional[AnalysisResult]:
        """
        Load analysis result from disk.
        
        Args:
            model_name: Model name
            layer_id: Layer ID
            load_matrices: Whether to load co-activation matrix and frequencies
            
        Returns:
            AnalysisResult or None if not found
        """
        layer_dir = self.store_root / model_name / f"layer_{layer_id}"
        result_path = layer_dir / "result.json"
        
        if not result_path.exists():
            return None
        
        # Load result metadata
        with open(result_path, 'r') as f:
            result_dict = json.load(f)
        
        result = AnalysisResult.from_dict(result_dict)
        
        # Load matrices if requested
        if load_matrices:
            if result.coactivation_matrix_path and os.path.exists(result.coactivation_matrix_path):
                result.coactivation_matrix = np.load(result.coactivation_matrix_path)
            if result.activation_frequencies_path and os.path.exists(result.activation_frequencies_path):
                result.activation_frequencies = np.load(result.activation_frequencies_path)
        
        return result
    
    def load_result_for_gpu(
        self,
        model_name: str,
        layer_id: int,
        gpu_id: int,
    ) -> Optional[Dict]:
        """
        Load partition and TC/CC assignment for a specific GPU.
        
        Args:
            model_name: Model name
            layer_id: Layer ID
            gpu_id: GPU rank
            
        Returns:
            Dictionary with partition and assignment for this GPU
        """
        result = self.load_result(model_name, layer_id)
        
        if result is None or gpu_id >= len(result.partition_results):
            return None
        
        partition = result.partition_results[gpu_id]
        tc_cc = result.tc_cc_assignments[gpu_id]
        
        return {
            'hans_indices': np.array(partition.hans_indices),
            'lans_indices': np.array(partition.lans_indices),
            'tc_indices': np.array(tc_cc.tc_indices),
            'cc_indices': np.array(tc_cc.cc_indices),
            'hidden_dim': result.hidden_dim,
            'intermediate_dim': result.intermediate_dim,
        }
    
    def list_models(self) -> List[str]:
        """List all models with stored results."""
        return list(self.index['models'].keys())
    
    def list_layers(self, model_name: str) -> List[int]:
        """List all layers with stored results for a model."""
        if model_name not in self.index['models']:
            return []
        return [int(k) for k in self.index['models'][model_name]['layers'].keys()]
    
    def get_result_summary(self, model_name: str, layer_id: int) -> Optional[Dict]:
        """Get summary of stored result without loading full data."""
        result = self.load_result(model_name, layer_id)
        if result is None:
            return None
        
        return {
            'layer_id': result.layer_id,
            'total_hans': result.total_hans,
            'total_lans': result.total_lans,
            'hans_ratio': result.total_hans / (result.total_hans + result.total_lans),
            'mean_activation_frequency': result.mean_activation_frequency,
            'analysis_time_s': result.analysis_time_s,
            'timestamp': result.timestamp,
        }
    
    def delete_result(self, model_name: str, layer_id: int):
        """Delete stored result for a layer."""
        layer_dir = self.store_root / model_name / f"layer_{layer_id}"
        
        if layer_dir.exists():
            import shutil
            shutil.rmtree(layer_dir)
        
        # Update index
        if model_name in self.index['models']:
            if str(layer_id) in self.index['models'][model_name]['layers']:
                del self.index['models'][model_name]['layers'][str(layer_id)]
                self._save_index()
    
    def validate_result(self, model_name: str, layer_id: int, config_hash: str) -> bool:
        """
        Validate that stored result matches current model configuration.
        
        Args:
            model_name: Model name
            layer_id: Layer ID
            config_hash: Hash of current model config
            
        Returns:
            True if result is valid for current config
        """
        result = self.load_result(model_name, layer_id)
        
        if result is None:
            return False
        
        return result.config_hash == config_hash


def compute_config_hash(
    hidden_dim: int,
    intermediate_dim: int,
    num_gpus: int,
    activation_threshold: float,
) -> str:
    """
    Compute hash for model configuration.
    
    Used to validate that stored results match current configuration.
    """
    import hashlib
    
    config_str = f"{hidden_dim}_{intermediate_dim}_{num_gpus}_{activation_threshold}"
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def create_analysis_result_from_partitioner(
    layer_id: int,
    model_name: str,
    hidden_dim: int,
    intermediate_dim: int,
    num_gpus: int,
    partitioner,  # NeuronPartitioner instance
    tc_cc_assignments: List[TCCCAssignmentResult],
    analysis_time_s: float,
) -> AnalysisResult:
    """
    Create AnalysisResult from partitioner output.
    
    Convenience function to create result object from solver outputs.
    """
    config_hash = compute_config_hash(
        hidden_dim, intermediate_dim, num_gpus,
        partitioner.activation_threshold
    )
    
    # Get partition results
    partition_results = []
    for i, partition in enumerate(partitioner.partitions):
        partition_results.append(PartitionResult(
            gpu_id=i,
            hans_indices=partition.hans_indices.tolist(),
            lans_indices=partition.lans_indices.tolist(),
            hans_count=len(partition.hans_indices),
            lans_count=len(partition.lans_indices),
        ))
    
    # Get statistics
    frequencies = partitioner.analyzer.compute_activation_frequencies()
    
    return AnalysisResult(
        layer_id=layer_id,
        model_name=model_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        config_hash=config_hash,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_gpus=num_gpus,
        partition_results=partition_results,
        tc_cc_assignments=tc_cc_assignments,
        total_hans=sum(p.hans_count for p in partition_results),
        total_lans=sum(p.lans_count for p in partition_results),
        mean_activation_frequency=float(frequencies.mean()),
        analysis_time_s=analysis_time_s,
    )
