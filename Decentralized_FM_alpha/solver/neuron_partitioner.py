"""
Layer 1 Solver: Neuron Partitioner

This module partitions neurons into HANS (High-Activation Neuron Sets)
and LANS (Low-Activation Neuron Sets) based on co-activation analysis.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import torch

from .coactivation_analyzer import CoactivationAnalyzer, SpectralCoactivationAnalyzer


@dataclass
class NeuronPartition:
    """
    Represents a partition of neurons for one GPU.
    
    Attributes:
        hans_indices: Indices of HANS neurons (high-activation)
        lans_indices: Indices of LANS neurons (low-activation)
        coactivated_hans_groups: List of co-activated HANS groups
    """
    hans_indices: np.ndarray
    lans_indices: np.ndarray
    coactivated_hans_groups: List[np.ndarray]

    def __post_init__(self):
        self.hans_indices = np.asarray(self.hans_indices, dtype=np.int64)
        self.lans_indices = np.asarray(self.lans_indices, dtype=np.int64)
        self.coactivated_hans_groups = [
            np.asarray(group, dtype=np.int64)
            for group in self.coactivated_hans_groups
        ]
    
    @property
    def total_neurons(self) -> int:
        return len(self.hans_indices) + len(self.lans_indices)
    
    def to_dict(self) -> Dict:
        return {
            'hans_indices': self.hans_indices.tolist(),
            'lans_indices': self.lans_indices.tolist(),
            'coactivated_hans_groups': [g.tolist() for g in self.coactivated_hans_groups],
        }


class NeuronPartitioner:
    """
    Partitions neurons into HANS and LANS based on activation patterns.
    
    The partitioning strategy considers:
    1. Activation frequency - HANS neurons are activated more frequently
    2. Co-activation patterns - neurons that are frequently co-activated
       should be grouped together to minimize cross-GPU communication
    3. Load balancing - distribute neurons evenly across GPUs
    """
    
    def __init__(
        self,
        num_neurons: int,
        num_gpus: int,
        activation_threshold: float = 0.1,
        coactivation_threshold: float = 0.05,
        use_coactivation_grouping: bool = True,
        min_hans_ratio: float = 0.2,
        max_hans_ratio: float = 0.8,
    ):
        """
        Initialize the neuron partitioner.
        
        Args:
            num_neurons: Total number of neurons (FFN intermediate dimension)
            num_gpus: Number of GPUs for tensor parallelism
            activation_threshold: Threshold for HANS/LANS classification
            coactivation_threshold: Threshold for co-activation grouping
            use_coactivation_grouping: Whether to group co-activated neurons
            min_hans_ratio: Minimum ratio of HANS neurons per GPU
            max_hans_ratio: Maximum ratio of HANS neurons per GPU
        """
        self.num_neurons = num_neurons
        self.num_gpus = num_gpus
        self.activation_threshold = activation_threshold
        self.coactivation_threshold = coactivation_threshold
        self.use_coactivation_grouping = use_coactivation_grouping
        self.min_hans_ratio = min_hans_ratio
        self.max_hans_ratio = max_hans_ratio
        
        # Initialize analyzer
        self.analyzer = SpectralCoactivationAnalyzer(num_neurons)
        
        # Partition results
        self.partitions: List[NeuronPartition] = []
        self.hans_mask: Optional[np.ndarray] = None
        self.coactivation_groups: List[np.ndarray] = []
    
    def analyze_and_partition(
        self,
        masks: np.ndarray,
        method: str = 'spectral',
        balance_strategy: str = 'even'
    ) -> List[NeuronPartition]:
        """
        Analyze activation patterns and partition neurons.
        
        Args:
            masks: Activation mask matrix of shape [S, F]
            method: Clustering method ('spectral', 'frequency', 'hierarchical')
            balance_strategy: Load balancing strategy ('even', 'proportional', 'adaptive')
            
        Returns:
            List of NeuronPartition, one per GPU
        """
        # Update analyzer with masks
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        if len(masks.shape) == 3:
            # Multiple layers, process each layer
            # For now, we assume single layer
            masks = masks.reshape(-1, masks.shape[-1])
        
        # Update co-activation statistics
        batch_size = min(1000, len(masks))
        for i in range(0, len(masks), batch_size):
            self.analyzer.update(masks[i:i+batch_size])
        
        # Get activation frequencies
        frequencies = self.analyzer.compute_activation_frequencies()
        
        # Classify neurons into HANS and LANS based on activation frequency
        self.hans_mask = frequencies >= self.activation_threshold
        
        # Identify co-activation groups within HANS
        if self.use_coactivation_grouping and method == 'spectral':
            self._identify_coactivation_groups_spectral()
        elif method == 'frequency':
            self._identify_coactivation_groups_frequency()
        else:
            self._identify_coactivation_groups_hierarchical()
        
        # Partition neurons across GPUs
        self.partitions = self._partition_across_gpus(balance_strategy)
        
        return self.partitions
    
    def _identify_coactivation_groups_spectral(self) -> None:
        """Use spectral clustering to identify co-activation groups within HANS."""
        hans_indices = np.where(self.hans_mask)[0]
        
        if len(hans_indices) < 2:
            # Not enough HANS neurons for clustering
            self.coactivation_groups = [hans_indices] if len(hans_indices) > 0 else []
            return
        
        # Get co-activation weights for HANS neurons
        CW = self.analyzer.compute_coactivation_weights()
        CW_hans = CW[np.ix_(hans_indices, hans_indices)]
        
        # Determine number of groups based on co-activation strength
        # Stronger co-activation -> fewer groups (more neurons clustered together)
        mean_coactivation = CW_hans.mean()
        n_groups = max(1, min(len(hans_indices) // 4, int(1.0 / (mean_coactivation + 0.1))))
        n_groups = min(n_groups, len(hans_indices))
        
        if n_groups <= 1:
            self.coactivation_groups = [hans_indices]
            return
        
        # Perform spectral clustering
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(
            n_clusters=n_groups,
            affinity='precomputed',
            random_state=42
        )
        
        try:
            labels = clustering.fit_predict(CW_hans)
            # Group HANS indices by cluster label
            self.coactivation_groups = []
            for i in range(n_groups):
                group = hans_indices[labels == i]
                if len(group) > 0:
                    self.coactivation_groups.append(group)
        except Exception as e:
            print(f"Spectral clustering failed: {e}, using single group")
            self.coactivation_groups = [hans_indices]
    
    def _identify_coactivation_groups_frequency(self) -> None:
        """Simple frequency-based grouping for HANS."""
        hans_indices = np.where(self.hans_mask)[0]
        frequencies = self.analyzer.compute_activation_frequencies()
        hans_freqs = frequencies[hans_indices]
        
        # Sort by frequency and split into groups
        sorted_indices = np.argsort(hans_freqs)[::-1]  # Descending order
        
        # Create groups of similar size
        n_groups = max(1, len(hans_indices) // 32)
        group_size = (len(hans_indices) + n_groups - 1) // n_groups
        
        self.coactivation_groups = []
        for i in range(0, len(hans_indices), group_size):
            group = hans_indices[sorted_indices[i:i+group_size]]
            self.coactivation_groups.append(group)
    
    def _identify_coactivation_groups_hierarchical(self) -> None:
        """Use hierarchical clustering for co-activation groups."""
        hans_indices = np.where(self.hans_mask)[0]
        
        if len(hans_indices) < 2:
            self.coactivation_groups = [hans_indices] if len(hans_indices) > 0 else []
            return
        
        # Get co-activation weights for HANS neurons
        CW = self.analyzer.compute_coactivation_weights()
        CW_hans = CW[np.ix_(hans_indices, hans_indices)]
        
        # Convert to distance matrix
        distance_matrix = 1.0 / (CW_hans + 1e-6)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        condensed = squareform(distance_matrix)
        linkage_matrix = linkage(condensed, method='average')
        
        # Determine number of clusters
        n_groups = max(1, min(len(hans_indices) // 4, 8))
        
        labels = fcluster(linkage_matrix, n_groups, criterion='maxclust')
        
        self.coactivation_groups = []
        for i in range(1, n_groups + 1):
            group = hans_indices[labels == i]
            if len(group) > 0:
                self.coactivation_groups.append(group)
    
    def _partition_across_gpus(
        self,
        balance_strategy: str
    ) -> List[NeuronPartition]:
        """
        Partition neurons across GPUs considering load balance.
        
        Args:
            balance_strategy: 'even' - equal number of neurons per GPU
                            'proportional' - based on activation frequency
                            'adaptive' - based on co-activation patterns
        """
        lans_indices = np.where(~self.hans_mask)[0]
        
        partitions = []
        
        if balance_strategy == 'even':
            partitions = self._even_partition(lans_indices)
        elif balance_strategy == 'proportional':
            partitions = self._proportional_partition(lans_indices)
        else:
            partitions = self._adaptive_partition(lans_indices)
        
        return partitions
    
    def _even_partition(self, lans_indices: np.ndarray) -> List[NeuronPartition]:
        """Evenly distribute neurons across GPUs."""
        partitions = []
        
        # Distribute co-activation groups across GPUs
        # Try to keep co-activated neurons on the same GPU
        group_assignments = self._assign_groups_to_gpus()
        
        # Distribute LANS evenly
        lans_per_gpu = len(lans_indices) // self.num_gpus
        lans_shuffled = lans_indices.copy()
        np.random.shuffle(lans_shuffled)
        
        for gpu_id in range(self.num_gpus):
            # Get HANS groups assigned to this GPU
            hans_groups = [
                group for i, group in enumerate(self.coactivation_groups)
                if group_assignments[i] == gpu_id
            ]
            hans_indices = np.concatenate(hans_groups) if hans_groups else np.array([], dtype=int)
            
            # Get LANS for this GPU
            start_idx = gpu_id * lans_per_gpu
            if gpu_id == self.num_gpus - 1:
                gpu_lans = lans_shuffled[start_idx:]
            else:
                gpu_lans = lans_shuffled[start_idx:start_idx + lans_per_gpu]
            
            partitions.append(NeuronPartition(
                hans_indices=np.sort(hans_indices),
                lans_indices=np.sort(gpu_lans),
                coactivated_hans_groups=hans_groups
            ))
        
        return partitions
    
    def _proportional_partition(self, lans_indices: np.ndarray) -> List[NeuronPartition]:
        """Distribute neurons proportionally based on activation frequency."""
        frequencies = self.analyzer.compute_activation_frequencies()
        
        partitions = []
        group_assignments = self._assign_groups_to_gpus()
        
        # Calculate total activation weight per GPU target
        total_activation = frequencies.sum()
        target_per_gpu = total_activation / self.num_gpus
        
        # Greedy assignment of LANS
        lans_freqs = [(idx, frequencies[idx]) for idx in lans_indices]
        lans_freqs.sort(key=lambda x: x[1], reverse=True)
        
        gpu_weights = [0.0] * self.num_gpus
        gpu_lans = [[] for _ in range(self.num_gpus)]
        
        for idx, freq in lans_freqs:
            # Assign to GPU with lowest current weight
            min_gpu = np.argmin(gpu_weights)
            gpu_lans[min_gpu].append(idx)
            gpu_weights[min_gpu] += freq
        
        for gpu_id in range(self.num_gpus):
            hans_groups = [
                group for i, group in enumerate(self.coactivation_groups)
                if group_assignments[i] == gpu_id
            ]
            hans_indices = np.concatenate(hans_groups) if hans_groups else np.array([], dtype=int)
            
            partitions.append(NeuronPartition(
                hans_indices=np.sort(hans_indices),
                lans_indices=np.sort(gpu_lans[gpu_id]),
                coactivated_hans_groups=hans_groups
            ))
        
        return partitions
    
    def _adaptive_partition(self, lans_indices: np.ndarray) -> List[NeuronPartition]:
        """
        Adaptively partition considering both co-activation and load balance.
        Uses a hybrid approach for optimal distribution.
        """
        frequencies = self.analyzer.compute_activation_frequencies()
        CW = self.analyzer.compute_coactivation_weights()
        
        partitions = []
        
        # Build a cost matrix for assigning neurons to GPUs
        # Cost considers:
        # 1. Load balance (activation frequency)
        # 2. Co-activation (try to keep co-activated neurons together)
        # 3. Communication (minimize cross-GPU co-activation)
        
        # Assign co-activation groups first
        group_assignments = self._assign_groups_to_gpus()
        
        # For each GPU, initialize with HANS groups
        gpu_hans = [[] for _ in range(self.num_gpus)]
        for group_id, group in enumerate(self.coactivation_groups):
            gpu_id = group_assignments[group_id]
            gpu_hans[gpu_id].extend(group.tolist())
        
        # Calculate current load per GPU from HANS
        gpu_loads = [
            sum(frequencies[idx] for idx in hans)
            for hans in gpu_hans
        ]
        
        # Assign LANS using greedy algorithm
        lans_with_freq = [(idx, frequencies[idx]) for idx in lans_indices]
        lans_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        gpu_lans = [[] for _ in range(self.num_gpus)]
        
        for idx, freq in lans_with_freq:
            # Calculate cost for each GPU
            costs = []
            for gpu_id in range(self.num_gpus):
                # Load balance cost
                load_cost = gpu_loads[gpu_id]
                
                # Communication cost (co-activation with neurons on other GPUs)
                comm_cost = 0
                for other_gpu in range(self.num_gpus):
                    if other_gpu != gpu_id:
                        for other_idx in gpu_hans[other_gpu] + gpu_lans[other_gpu]:
                            comm_cost += CW[idx, other_idx]
                
                costs.append(load_cost + comm_cost * 0.1)
            
            # Assign to GPU with minimum cost
            best_gpu = np.argmin(costs)
            gpu_lans[best_gpu].append(idx)
            gpu_loads[best_gpu] += freq
        
        # Create partitions
        for gpu_id in range(self.num_gpus):
            hans_indices = np.array(gpu_hans[gpu_id], dtype=int)
            hans_groups = [
                group for i, group in enumerate(self.coactivation_groups)
                if group_assignments[i] == gpu_id
            ]
            
            partitions.append(NeuronPartition(
                hans_indices=np.sort(hans_indices),
                lans_indices=np.sort(gpu_lans[gpu_id]),
                coactivated_hans_groups=hans_groups
            ))
        
        return partitions
    
    def _assign_groups_to_gpus(self) -> np.ndarray:
        """
        Assign co-activation groups to GPUs.
        Tries to balance group sizes across GPUs.
        """
        if len(self.coactivation_groups) == 0:
            return np.array([])
        
        # Calculate group sizes
        group_sizes = np.array([len(g) for g in self.coactivation_groups])
        
        # Greedy assignment: assign each group to GPU with least total neurons
        assignments = np.zeros(len(self.coactivation_groups), dtype=int)
        gpu_totals = np.zeros(self.num_gpus)
        
        # Sort groups by size (descending)
        sorted_indices = np.argsort(group_sizes)[::-1]
        
        for idx in sorted_indices:
            # Assign to GPU with minimum total
            best_gpu = np.argmin(gpu_totals)
            assignments[idx] = best_gpu
            gpu_totals[best_gpu] += group_sizes[idx]
        
        return assignments
    
    def get_partition_info(self) -> Dict:
        """Get detailed information about the partitioning."""
        if not self.partitions:
            return {}
        
        frequencies = self.analyzer.compute_activation_frequencies()
        
        info = {
            'num_gpus': self.num_gpus,
            'total_neurons': self.num_neurons,
            'total_hans': sum(len(p.hans_indices) for p in self.partitions),
            'total_lans': sum(len(p.lans_indices) for p in self.partitions),
            'num_coactivation_groups': len(self.coactivation_groups),
            'per_gpu_stats': []
        }
        
        for i, partition in enumerate(self.partitions):
            hans_freq_mean = frequencies[partition.hans_indices].mean() if len(partition.hans_indices) > 0 else 0
            lans_freq_mean = frequencies[partition.lans_indices].mean() if len(partition.lans_indices) > 0 else 0
            
            info['per_gpu_stats'].append({
                'gpu_id': i,
                'num_hans': len(partition.hans_indices),
                'num_lans': len(partition.lans_indices),
                'total_neurons': partition.total_neurons,
                'hans_freq_mean': hans_freq_mean,
                'lans_freq_mean': lans_freq_mean,
                'num_coactivated_groups': len(partition.coactivated_hans_groups),
            })
        
        return info
    
    def save_partitions(self, path: str) -> None:
        """Save partition configuration to file."""
        data = {
            'partitions': [p.to_dict() for p in self.partitions],
            'config': {
                'num_neurons': self.num_neurons,
                'num_gpus': self.num_gpus,
                'activation_threshold': self.activation_threshold,
                'coactivation_threshold': self.coactivation_threshold,
            },
            'info': self.get_partition_info(),
        }
        np.save(path, data)
    
    def load_partitions(self, path: str) -> None:
        """Load partition configuration from file."""
        data = np.load(path, allow_pickle=True).item()
        
        self.partitions = []
        for p_dict in data['partitions']:
            self.partitions.append(NeuronPartition(
                hans_indices=np.array(p_dict['hans_indices']),
                lans_indices=np.array(p_dict['lans_indices']),
                coactivated_hans_groups=[np.array(g) for g in p_dict['coactivated_hans_groups']],
            ))
