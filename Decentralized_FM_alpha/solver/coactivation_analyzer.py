"""
Layer 1 Solver: Neuron Co-activation Analysis

This module analyzes co-activation patterns from activation masks
and computes co-activation weights between neuron pairs.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy.sparse import csr_matrix, lil_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering
import torch


class CoactivationAnalyzer:
    """
    Analyzes neuron co-activation patterns from prediction masks.
    
    Given a binary mask matrix M[S, F] where:
    - S: number of tokens/samples
    - F: number of neurons (FFN intermediate dimension)
    
    Computes co-activation matrix CW[F, F] where CW[u,v] represents
    the co-activation strength between neuron u and v.
    """
    
    def __init__(
        self,
        num_neurons: int,
        threshold_ratio: float = 0.1,
        use_sparse: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize the co-activation analyzer.
        
        Args:
            num_neurons: Number of neurons in FFN layer (F)
            threshold_ratio: Threshold ratio for activation frequency to be considered HANS
            use_sparse: Whether to use sparse matrix for memory efficiency
            device: Device to run computations on
        """
        self.num_neurons = num_neurons
        self.threshold_ratio = threshold_ratio
        self.use_sparse = use_sparse
        self.device = device
        
        # Accumulators for online computation
        self.total_samples = 0
        self.activation_counts = np.zeros(num_neurons, dtype=np.float32)
        if use_sparse:
            self.coactivation_sum = lil_matrix((num_neurons, num_neurons), dtype=np.float32)
        else:
            self.coactivation_sum = np.zeros((num_neurons, num_neurons), dtype=np.float32)
    
    def update(self, mask: np.ndarray) -> None:
        """
        Update co-activation statistics with new batch of masks.
        
        Args:
            mask: Binary mask matrix of shape [S, F]
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        S, F = mask.shape
        assert F == self.num_neurons, f"Expected {self.num_neurons} neurons, got {F}"
        
        # Update total samples
        self.total_samples += S
        
        # Update activation counts (sum over all samples)
        self.activation_counts += mask.sum(axis=0)
        
        # Compute co-activation for this batch
        if self.use_sparse:
            mask_sparse = csr_matrix(mask)
            # Co-activation: M^T @ M gives the co-activation count
            batch_coactivation = mask_sparse.T @ mask_sparse
            self.coactivation_sum += batch_coactivation.tolil()
        else:
            # Dense computation
            batch_coactivation = mask.T @ mask
            self.coactivation_sum += batch_coactivation
    
    def compute_coactivation_weights(self, normalize: bool = True) -> np.ndarray:
        """
        Compute the final co-activation weight matrix.
        
        Args:
            normalize: Whether to normalize by total samples
            
        Returns:
            Co-activation weight matrix CW[F, F]
        """
        if self.use_sparse:
            CW = self.coactivation_sum.toarray()
        else:
            CW = self.coactivation_sum.copy()
        
        if normalize and self.total_samples > 0:
            CW = CW / self.total_samples
        
        # Set diagonal to 0 (no self-coactivation)
        np.fill_diagonal(CW, 0)
        
        return CW
    
    def compute_activation_frequencies(self) -> np.ndarray:
        """
        Compute activation frequency for each neuron.
        
        Returns:
            Array of shape [F] with activation frequencies
        """
        if self.total_samples == 0:
            return np.zeros(self.num_neurons)
        return self.activation_counts / self.total_samples
    
    def get_neuron_statistics(self) -> Dict:
        """
        Get statistics about neuron activations.
        
        Returns:
            Dictionary with activation statistics
        """
        frequencies = self.compute_activation_frequencies()
        
        return {
            'total_samples': self.total_samples,
            'mean_frequency': frequencies.mean(),
            'std_frequency': frequencies.std(),
            'max_frequency': frequencies.max(),
            'min_frequency': frequencies.min(),
            'median_frequency': np.median(frequencies),
            'num_neurons': self.num_neurons,
        }
    
    def save_state(self, path: str) -> None:
        """Save analyzer state to file."""
        state = {
            'total_samples': self.total_samples,
            'activation_counts': self.activation_counts,
            'coactivation_sum': self.coactivation_sum if not self.use_sparse else self.coactivation_sum.toarray(),
            'num_neurons': self.num_neurons,
            'threshold_ratio': self.threshold_ratio,
        }
        np.savez(path, **state)
    
    def load_state(self, path: str) -> None:
        """Load analyzer state from file."""
        state = np.load(path)
        self.total_samples = int(state['total_samples'])
        self.activation_counts = state['activation_counts']
        coactivation_data = state['coactivation_sum']
        if self.use_sparse:
            self.coactivation_sum = lil_matrix(coactivation_data)
        else:
            self.coactivation_sum = coactivation_data
        self.num_neurons = int(state['num_neurons'])
        self.threshold_ratio = float(state['threshold_ratio'])


class HierarchicalCoactivationAnalyzer(CoactivationAnalyzer):
    """
    Advanced analyzer with hierarchical clustering for identifying
    co-activation groups at different granularities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linkage_matrix = None
        self.cluster_labels = None
    
    def hierarchical_cluster(
        self,
        method: str = 'average',
        metric: str = 'precomputed'
    ) -> np.ndarray:
        """
        Perform hierarchical clustering on neurons based on co-activation.
        
        Args:
            method: Linkage method ('average', 'complete', 'ward', etc.)
            metric: Distance metric (use 'precomputed' for distance matrix)
            
        Returns:
            Linkage matrix for hierarchical clustering
        """
        CW = self.compute_coactivation_weights()
        
        # Convert co-activation to distance (higher co-activation = lower distance)
        # Add small epsilon to avoid division by zero
        distance_matrix = 1.0 / (CW + 1e-6)
        
        # Make symmetric and zero diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed form for scipy
        from scipy.spatial.distance import squareform
        condensed_distance = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(condensed_distance, method=method)
        
        return self.linkage_matrix
    
    def get_clusters_at_level(
        self,
        n_clusters: int = 2
    ) -> np.ndarray:
        """
        Get cluster assignments at a specific level.
        
        Args:
            n_clusters: Number of clusters to form
            
        Returns:
            Cluster labels for each neuron
        """
        if self.linkage_matrix is None:
            raise ValueError("Run hierarchical_cluster first")
        
        self.cluster_labels = fcluster(
            self.linkage_matrix,
            n_clusters,
            criterion='maxclust'
        )
        
        return self.cluster_labels


class SpectralCoactivationAnalyzer(CoactivationAnalyzer):
    """
    Analyzer using spectral clustering for identifying HANS/LANS groups.
    """
    
    def spectral_cluster(
        self,
        n_clusters: int = 2,
        n_init: int = 10,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Perform spectral clustering on the co-activation graph.
        
        Args:
            n_clusters: Number of clusters (default 2 for HANS/LANS)
            n_init: Number of initializations
            random_state: Random seed
            
        Returns:
            Cluster labels for each neuron
        """
        CW = self.compute_coactivation_weights(normalize=True)
        
        # Create similarity matrix from co-activation
        similarity_matrix = CW
        
        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            n_init=n_init,
            random_state=random_state,
            assign_labels='kmeans'
        )
        
        self.cluster_labels = clustering.fit_predict(similarity_matrix)
        
        return self.cluster_labels
