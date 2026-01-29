"""
Multi-GPU collective communication implementation selection.

This module provides strategies and utilities for selecting the best implementation
of collective communication operations based on GPU architecture and hardware capabilities.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Dict
import warnings

from . import cuda_utils
from .persistent_kernel import TBGraph, CyTBGraph


class GPUArchitecture:
    """GPU architecture types."""
    AMPERE = 80      # SM 80 (A100, etc.)
    HOPPER = 90      # SM 90 (H100, H200, GH200)
    BLACKWELL = 100  # SM 100 (B100, GB200)


class CollectiveCapabilities:
    """
    Stores hardware capabilities relevant to collective communication selection.
    """
    def __init__(self, num_devices: int, device_id: int = 0):
        self.num_devices = num_devices
        self.device_id = device_id
        self.target_cc = None
        self.peer_access_supported = False
        self.vmm_supported = False
        self.multicast_supported = False
        self.posix_handle_supported = False
        
        self._query_capabilities()
    
    def _query_capabilities(self):
        """Query GPU capabilities using cuda_utils."""
        # Query target compute capability
        (major_cc, minor_cc) = cuda_utils.queryComputeCapability(self.device_id)
        self.target_cc = major_cc * 10 + minor_cc

        # Query peer access support
        full_peer_access_supported = True
        for target_id in range(self.num_devices):
            if target_id == self.device_id:
                continue
            supported = cuda_utils.queryPeerAccessSupported(self.device_id, target_id)
            if not supported:
                full_peer_access_supported = False
                break
        self.peer_access_supported = full_peer_access_supported

        # Query virtual memory management
        self.vmm_supported = cuda_utils.queryVMMsupport(self.device_id)
        
        # Query multicast support
        self.multicast_supported = cuda_utils.queryMulticastSupport(self.device_id)
        
        # Query POSIX handle support
        self.posix_handle_supported = cuda_utils.queryHandleTypePosixFileDescriptorSupported(self.device_id)

collective_capabilities = None

def get_collective_capabilities(num_devices: int, device_id: int = 0) -> CollectiveCapabilities:
    """Get or create the CollectiveCapabilities instance."""
    global collective_capabilities
    if collective_capabilities is None:
        collective_capabilities = CollectiveCapabilities(num_devices, device_id)
    return collective_capabilities

# ============================================================================
# Strategy Pattern: Base Classes for Collective Implementations
# ============================================================================

class CollectiveStrategy(ABC):
    """Base class for all collective communication strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def register_tasks(self, mpk, tensors: List, grid_dim: Tuple, 
                      block_dim: Tuple, params: List[int]) -> None:
        """
        Register tasks to the kernel graph.
        
        Args:
            mpk: The persistent kernel instance
            tensors: List of tensors involved in the collective
            grid_dim: Grid dimensions for kernel launch
            block_dim: Block dimensions for kernel launch
            params: Additional parameters (typically [world_size, rank])
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class AllReduceStrategy(CollectiveStrategy):
    """Base class for AllReduce implementations."""
    pass


class AllGatherStrategy(CollectiveStrategy):
    """Base class for AllGather implementations."""
    pass


class AllToAllStrategy(CollectiveStrategy):
    """Base class for AllToAll implementations."""
    pass


class BroadcastStrategy(CollectiveStrategy):
    """Base class for Broadcast implementations."""
    pass


class ReduceScatterStrategy(CollectiveStrategy):
    """Base class for ReduceScatter implementations."""
    pass


# ============================================================================
# Concrete AllReduce Implementations
# ============================================================================

class AllReduceStrategy_AllgatherReduce(AllReduceStrategy):
    """AllReduce using NVSHMEM allgather + local reduction."""
    
    def __init__(self):
        super().__init__("allgather + reduce")
    
    def register_tasks(self, mpk, tensors: Dict, grid_dim: Tuple,
                      block_dim: Tuple, params: List[int]) -> None:
        """Register allgather + reduction tasks."""
        
        assert len(params) == 2, "params should contain [world_size, rank]"

        input_tensor = tensors.pop("input")
        output_tensor = tensors.pop("output")
        buffer_tensor = tensors.pop("buffer")

        if len(tensors) > 0:
            print(f"{self} Unused tensors: {tensors.keys()}")

        # Create allgather task
        allgather_tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        allgather_tb_graph.new_input(input_tensor, (1, -1, -1), -1, True)
        allgather_tb_graph.new_input(buffer_tensor, (2, -1, -1), -1, True)
        mpk.kn_graph.customized([input_tensor, buffer_tensor], allgather_tb_graph)
        mpk.kn_graph.register_task(allgather_tb_graph, "nvshmem_allgather_strided_put", params)
        
        # Create reduction task
        reduction_tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        reduction_tb_graph.new_input(input_tensor, (1, -1, -1), -1, True)
        reduction_tb_graph.new_input(buffer_tensor, (2, -1, -1), -1, True)
        reduction_tb_graph.new_input(output_tensor, (1, -1, -1), -1, True)
        mpk.kn_graph.customized([input_tensor, buffer_tensor, output_tensor], reduction_tb_graph)
        mpk.kn_graph.register_task(reduction_tb_graph, "reduction", params)


class NvshmemTileAllReduceStrategy(AllReduceStrategy):
    """AllReduce using NVSHMEM tile-based operations (for SM >= 90)."""
    
    def __init__(self):
        super().__init__("nvshmem_tile_allreduce")
    
    def register_tasks(self, mpk, tensors: List, grid_dim: Tuple,
                      block_dim: Tuple, params: List[int]) -> None:
        """Register tile-based allreduce tasks."""
        # TODO: Implement when tile-based allreduce is available
        raise NotImplementedError(
            "Tile-based allreduce is not yet implemented for SM90 and above.")

# ============================================================================
# Concrete AllGather Implementations
# ============================================================================


# ============================================================================
# Concrete AllToAll Implementations
# ============================================================================


# ============================================================================
# Selection Functions
# ============================================================================

def auto_select_allreduce_implementation(
    num_gpus: int,
    device_id: int = 0,
) -> AllReduceStrategy:
    """
    Automatically select the best AllReduce implementation.
    
    Args:
        num_gpus: Number of GPUs involved in the collective
        device_id: GPU device ID to query capabilities
        
    Returns:
        An AllReduceStrategy instance ready to register tasks
    """
    capabilities = get_collective_capabilities(num_gpus, device_id)
    
    # For SM >= 90, prefer tile-based allreduce if available
    if capabilities.target_cc >= 90:
        if (capabilities.vmm_supported and capabilities.multicast_supported 
            and capabilities.peer_access_supported):
            return NvshmemTileAllReduceStrategy()

    # Default to allgather + reduction
    return AllReduceStrategy_AllgatherReduce()


def auto_select_allgather_implementation(
    num_gpus: int,
    device_id: int = 0,
) -> AllGatherStrategy:
    """
    Automatically select the best AllGather implementation.
    
    Args:
        num_gpus: Number of GPUs involved in the collective
        device_id: GPU device ID to query capabilities
        
    Returns:
        An AllGatherStrategy instance ready to register tasks
    """
    raise NotImplementedError("AllGather strategies are not yet implemented.")


def auto_select_broadcast_implementation(
    num_gpus: int,
    device_id: int = 0,
) -> BroadcastStrategy:
    """
    Automatically select the best Broadcast implementation.
    
    Args:
        num_gpus: Number of GPUs involved in the collective
        device_id: GPU device ID to query capabilities
        
    Returns:
        A BroadcastStrategy instance ready to register tasks
    """
    raise NotImplementedError("Broadcast strategies are not yet implemented.")


def auto_select_reduce_scatter_implementation(
    num_gpus: int,
    device_id: int = 0,
) -> ReduceScatterStrategy:
    """
    Automatically select the best ReduceScatter implementation.
    
    Args:
        num_gpus: Number of GPUs involved in the collective
        device_id: GPU device ID to query capabilities
        
    Returns:
        A ReduceScatterStrategy instance ready to register tasks
    """
    raise NotImplementedError("ReduceScatter strategies are not yet implemented.")