from __future__ import annotations
from cs336_systems.fsdp import FullyShardedDataParallel

import torch



def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    from cs336_systems.flash_attention import TritonFlashAttention 
    return TritonFlashAttention


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    from cs336_systems.flash_attention import TritonFlashAttention 
    return TritonFlashAttention


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDP(module)
    from cs336_systems.naive_ddp import DDP
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    from cs336_systems.naive_ddp import ddp_on_after_backward
    return ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    return FullyShardedDataParallel(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return fsdp_model.gather_full_state_dict()


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    from cs336_systems.optimizer_sharding import ShardedOptimzer
    return ShardedOptimzer(params,optimizer_cls,**kwargs)
