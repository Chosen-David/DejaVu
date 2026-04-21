"""
Layer 3 Solver: Pipeline Scheduler for Token Chunks

This module schedules token chunks for compute-communication overlap.
Tokens are divided into chunks that can overlap computation with AllReduce
communication from previous chunks.

Pipeline pattern:
  Chunk 0: Compute -> AllReduce
  Chunk 1:       Compute -> AllReduce
  Chunk 2:             Compute -> AllReduce

Goal: Overlap AllReduce of chunk i with Compute of chunk i+1
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import torch


class ChunkState(Enum):
    PENDING = "pending"
    COMPUTING = "computing"
    REDUCING = "reducing"
    COMPLETED = "completed"


@dataclass
class Chunk:
    """
    Represents a chunk of tokens.
    
    Attributes:
        chunk_id: Chunk identifier
        token_indices: Original token indices in the sequence
        start_pos: Starting position in sequence
        end_pos: Ending position in sequence
        state: Current state of the chunk
        compute_time: Estimated compute time (ms)
        comm_time: Estimated communication time (ms)
    """
    chunk_id: int
    token_indices: np.ndarray
    start_pos: int
    end_pos: int
    state: ChunkState = ChunkState.PENDING
    compute_time: float = 0.0
    comm_time: float = 0.0


@dataclass
class PipelineSchedule:
    """
    Complete pipeline schedule for all chunks.
    
    Attributes:
        chunks: List of chunks with their schedules
        total_time: Total execution time (ms)
        overlap_efficiency: Efficiency of compute-communication overlap
        timeline: Event timeline for visualization
    """
    chunks: List[Chunk]
    total_time: float
    overlap_efficiency: float
    timeline: List[Dict]
    
    def get_chunk_order(self) -> List[int]:
        """Get chunk processing order."""
        return [c.chunk_id for c in self.chunks]


class PipelineScheduler:
    """
    Schedules token chunks for optimal compute-communication overlap.
    
    Strategy:
    1. Divide sequence into chunks
    2. Schedule chunks to overlap AllReduce with computation
    3. Consider activation sparsity for communication optimization
    """
    
    def __init__(
        self,
        num_gpus: int,
        hidden_dim: int,
        bandwidth_gbps: float = 600.0,  # NVLink bandwidth
        latency_us: float = 10.0,  # NVLink latency
        target_chunk_size: int = 512,  # Target tokens per chunk
        max_chunks: int = 8,  # Maximum number of chunks
    ):
        """
        Initialize pipeline scheduler.
        
        Args:
            num_gpus: Number of GPUs for TP
            hidden_dim: Model hidden dimension
            bandwidth_gbps: Inter-GPU bandwidth in GB/s
            latency_us: Communication latency in microseconds
            target_chunk_size: Target number of tokens per chunk
            max_chunks: Maximum number of chunks to create
        """
        self.num_gpus = num_gpus
        self.hidden_dim = hidden_dim
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_us = latency_us
        self.target_chunk_size = target_chunk_size
        self.max_chunks = max_chunks
    
    def schedule(
        self,
        sequence_length: int,
        activation_sparsity: float = 0.5,
        compute_time_per_token: float = 0.001,  # ms per token
        strategy: str = 'two_chunk'
    ) -> PipelineSchedule:
        """
        Create optimal chunk schedule.
        
        Args:
            sequence_length: Total number of tokens (S)
            activation_sparsity: Fraction of neurons activated (0-1)
            compute_time_per_token: Compute time per token (ms)
            strategy: 'two_chunk', 'adaptive', or 'optimal'
            
        Returns:
            PipelineSchedule with chunk assignments
        """
        if strategy == 'two_chunk':
            return self._two_chunk_schedule(
                sequence_length, activation_sparsity, compute_time_per_token
            )
        elif strategy == 'adaptive':
            return self._adaptive_schedule(
                sequence_length, activation_sparsity, compute_time_per_token
            )
        else:
            return self._optimal_schedule(
                sequence_length, activation_sparsity, compute_time_per_token
            )
    
    def _two_chunk_schedule(
        self,
        sequence_length: int,
        activation_sparsity: float,
        compute_time_per_token: float,
    ) -> PipelineSchedule:
        """
        Simple two-chunk schedule as described in the design.
        Divide tokens into two equal halves for maximal overlap.
        """
        # Create two equal chunks
        chunk_size = sequence_length // 2
        
        chunks = [
            Chunk(
                chunk_id=0,
                token_indices=np.arange(0, chunk_size),
                start_pos=0,
                end_pos=chunk_size
            ),
            Chunk(
                chunk_id=1,
                token_indices=np.arange(chunk_size, sequence_length),
                start_pos=chunk_size,
                end_pos=sequence_length
            )
        ]
        
        # Estimate times for each chunk
        for chunk in chunks:
            chunk_size_actual = len(chunk.token_indices)
            
            # Compute time: proportional to chunk size
            chunk.compute_time = chunk_size_actual * compute_time_per_token
            
            # Communication time: AllReduce for sparse activations
            # Data size = chunk_size * hidden_dim * activation_sparsity * 2 bytes (fp16)
            data_size_bytes = chunk_size_actual * self.hidden_dim * activation_sparsity * 2
            chunk.comm_time = self._estimate_allreduce_time(data_size_bytes)
        
        # Calculate total time with overlap
        # Timeline:
        #   Chunk 0: [Compute] [AllReduce]
        #   Chunk 1:          [Compute] [AllReduce]
        
        # With perfect overlap:
        # Total time = chunk0_compute + max(chunk0_comm, chunk1_compute) + chunk1_comm
        overlap_time = max(chunks[0].comm_time, chunks[1].compute_time)
        total_time = chunks[0].compute_time + overlap_time + chunks[1].comm_time
        
        # Overlap efficiency
        sequential_time = sum(c.compute_time + c.comm_time for c in chunks)
        overlap_efficiency = (sequential_time - total_time) / sequential_time
        
        # Build timeline
        timeline = self._build_timeline(chunks)
        
        return PipelineSchedule(
            chunks=chunks,
            total_time=total_time,
            overlap_efficiency=overlap_efficiency,
            timeline=timeline
        )
    
    def _adaptive_schedule(
        self,
        sequence_length: int,
        activation_sparsity: float,
        compute_time_per_token: float,
    ) -> PipelineSchedule:
        """
        Adaptive schedule with variable chunk sizes.
        Adjusts chunk sizes to balance compute and communication times.
        """
        # Calculate optimal chunk sizes to balance compute and comm
        total_compute_time = sequence_length * compute_time_per_token
        total_data_size = sequence_length * self.hidden_dim * activation_sparsity * 2
        total_comm_time = self._estimate_allreduce_time(total_data_size)
        
        # If compute >> comm, use fewer chunks
        # If comm >> compute, use more chunks to hide latency
        if total_compute_time > total_comm_time * 2:
            # Compute-bound: use 2 chunks
            num_chunks = 2
        elif total_comm_time > total_compute_time * 2:
            # Communication-bound: use more chunks
            num_chunks = min(self.max_chunks, 4)
        else:
            # Balanced: find optimal number
            # Ideal: chunk_compute ≈ chunk_comm
            ratio = total_compute_time / total_comm_time
            num_chunks = min(self.max_chunks, max(2, int(2 / ratio)))
        
        # Create chunks
        chunks = []
        chunk_size = sequence_length // num_chunks
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = sequence_length if i == num_chunks - 1 else (i + 1) * chunk_size
            
            chunk = Chunk(
                chunk_id=i,
                token_indices=np.arange(start, end),
                start_pos=start,
                end_pos=end
            )
            
            # Estimate times
            actual_size = len(chunk.token_indices)
            chunk.compute_time = actual_size * compute_time_per_token
            data_size = actual_size * self.hidden_dim * activation_sparsity * 2
            chunk.comm_time = self._estimate_allreduce_time(data_size)
            
            chunks.append(chunk)
        
        # Calculate total time with overlap
        total_time = self._calculate_pipelined_time(chunks)
        
        sequential_time = sum(c.compute_time + c.comm_time for c in chunks)
        overlap_efficiency = (sequential_time - total_time) / sequential_time
        
        timeline = self._build_timeline(chunks)
        
        return PipelineSchedule(
            chunks=chunks,
            total_time=total_time,
            overlap_efficiency=overlap_efficiency,
            timeline=timeline
        )
    
    def _optimal_schedule(
        self,
        sequence_length: int,
        activation_sparsity: float,
        compute_time_per_token: float,
    ) -> PipelineSchedule:
        """
        Find optimal chunk configuration through search.
        """
        best_schedule = None
        best_time = float('inf')
        
        # Try different numbers of chunks
        for num_chunks in range(2, self.max_chunks + 1):
            # Try different chunk size distributions
            # Simple approach: try even split
            chunks = []
            base_size = sequence_length // num_chunks
            
            for i in range(num_chunks):
                start = i * base_size
                end = sequence_length if i == num_chunks - 1 else (i + 1) * base_size
                
                chunk = Chunk(
                    chunk_id=i,
                    token_indices=np.arange(start, end),
                    start_pos=start,
                    end_pos=end
                )
                
                actual_size = len(chunk.token_indices)
                chunk.compute_time = actual_size * compute_time_per_token
                data_size = actual_size * self.hidden_dim * activation_sparsity * 2
                chunk.comm_time = self._estimate_allreduce_time(data_size)
                
                chunks.append(chunk)
            
            total_time = self._calculate_pipelined_time(chunks)
            
            if total_time < best_time:
                best_time = total_time
                sequential_time = sum(c.compute_time + c.comm_time for c in chunks)
                overlap_efficiency = (sequential_time - total_time) / sequential_time
                timeline = self._build_timeline(chunks)
                
                best_schedule = PipelineSchedule(
                    chunks=chunks,
                    total_time=total_time,
                    overlap_efficiency=overlap_efficiency,
                    timeline=timeline
                )
        
        return best_schedule
    
    def _estimate_allreduce_time(
        self,
        data_size_bytes: float,
    ) -> float:
        """
        Estimate AllReduce time for given data size.
        
        Uses Ring AllReduce model:
        Time = 2 * (n-1)/n * size / bandwidth + 2 * (n-1) * latency
        """
        if data_size_bytes == 0:
            return 0.0
        
        n = self.num_gpus
        bandwidth_bps = self.bandwidth_gbps * 1e9
        latency_s = self.latency_us * 1e-6
        
        # Ring AllReduce time
        # For small messages, latency dominates
        # For large messages, bandwidth dominates
        if data_size_bytes < 1e6:  # < 1MB
            # Latency-dominated: use tree algorithm
            time = np.log2(n) * (latency_s + data_size_bytes / bandwidth_bps)
        else:
            # Bandwidth-dominated: use ring algorithm
            time = 2 * (n - 1) / n * data_size_bytes / bandwidth_bps + 2 * (n - 1) * latency_s
        
        return time * 1000  # Convert to ms
    
    def _calculate_pipelined_time(
        self,
        chunks: List[Chunk],
    ) -> float:
        """
        Calculate total time for pipelined execution.
        
        Simulates the timeline:
        Chunk i: Compute starts after chunk i-1 compute starts
        Chunk i: AllReduce starts after chunk i compute and chunk i-1 AllReduce complete
        """
        if len(chunks) == 0:
            return 0.0
        
        # Simulate timeline
        compute_end_times = [0.0] * len(chunks)
        allreduce_end_times = [0.0] * len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Compute can start after previous chunk's compute starts
            # (but typically we wait for previous to finish for memory)
            if i == 0:
                compute_start = 0.0
            else:
                compute_start = compute_end_times[i - 1]
            
            compute_end_times[i] = compute_start + chunk.compute_time
            
            # AllReduce can start after:
            # 1. Current chunk compute is done
            # 2. Previous chunk AllReduce is done (communication resource)
            if i == 0:
                allreduce_start = compute_end_times[i]
            else:
                allreduce_start = max(compute_end_times[i], allreduce_end_times[i - 1])
            
            allreduce_end_times[i] = allreduce_start + chunk.comm_time
        
        return allreduce_end_times[-1]
    
    def _build_timeline(
        self,
        chunks: List[Chunk],
    ) -> List[Dict]:
        """
        Build timeline for visualization.
        """
        timeline = []
        current_time = 0.0
        
        for i, chunk in enumerate(chunks):
            # Compute event
            timeline.append({
                'chunk_id': chunk.chunk_id,
                'event': 'compute',
                'start_time': current_time,
                'end_time': current_time + chunk.compute_time,
                'duration': chunk.compute_time,
            })
            
            compute_end = current_time + chunk.compute_time
            
            # AllReduce event
            if i > 0:
                # Wait for previous AllReduce
                prev_reduce_end = timeline[-2]['end_time'] if len(timeline) >= 2 else 0
                reduce_start = max(compute_end, prev_reduce_end)
            else:
                reduce_start = compute_end
            
            timeline.append({
                'chunk_id': chunk.chunk_id,
                'event': 'allreduce',
                'start_time': reduce_start,
                'end_time': reduce_start + chunk.comm_time,
                'duration': chunk.comm_time,
            })
            
            current_time = compute_end
        
        return timeline
    
    def optimize_chunk_sizes(
        self,
        sequence_length: int,
        activation_sparsity: float,
        compute_time_per_token: float,
        num_chunks: int = 2,
    ) -> List[int]:
        """
        Optimize chunk sizes to minimize total time.
        
        Returns:
            List of chunk sizes
        """
        # For two chunks, optimal split is when:
        # chunk0_comm_time ≈ chunk1_compute_time
        # This maximizes overlap
        
        if num_chunks == 2:
            # Binary search for optimal split
            left, right = 1, sequence_length - 1
            best_split = sequence_length // 2
            best_time = float('inf')
            
            for _ in range(20):  # Max iterations
                mid = (left + right) // 2
                
                # Calculate times for this split
                chunk0_size = mid
                chunk1_size = sequence_length - mid
                
                chunk0_compute = chunk0_size * compute_time_per_token
                chunk0_data = chunk0_size * self.hidden_dim * activation_sparsity * 2
                chunk0_comm = self._estimate_allreduce_time(chunk0_data)
                
                chunk1_compute = chunk1_size * compute_time_per_token
                
                # Overlap time
                overlap = max(chunk0_comm, chunk1_compute)
                total = chunk0_compute + overlap + self._estimate_allreduce_time(
                    chunk1_size * self.hidden_dim * activation_sparsity * 2
                )
                
                if total < best_time:
                    best_time = total
                    best_split = mid
                
                # Adjust search direction
                if chunk0_comm > chunk1_compute:
                    # chunk0 comm too long, reduce chunk0 size
                    right = mid - 1
                else:
                    # chunk0 comm too short, increase chunk0 size
                    left = mid + 1
            
            return [best_split, sequence_length - best_split]
        
        else:
            # For more chunks, use even split
            chunk_size = sequence_length // num_chunks
            sizes = [chunk_size] * (num_chunks - 1)
            sizes.append(sequence_length - sum(sizes))
            return sizes
    
    def get_schedule_info(
        self,
        schedule: PipelineSchedule,
    ) -> Dict:
        """
        Get detailed information about the schedule.
        """
        return {
            'num_chunks': len(schedule.chunks),
            'total_time_ms': schedule.total_time,
            'overlap_efficiency': schedule.overlap_efficiency,
            'chunk_details': [
                {
                    'chunk_id': c.chunk_id,
                    'num_tokens': len(c.token_indices),
                    'compute_time_ms': c.compute_time,
                    'comm_time_ms': c.comm_time,
                }
                for c in schedule.chunks
            ]
        }
