import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.multiprocessing as mp
import torch.nn as nn

class DDP(nn.Module):
    def __init__(self, module : nn.Module):
        super().__init__()
        self.module = module
        self._handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for buf in self.module.buffers():
            dist.broadcast(buf.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self,param):
        def hook(*_):
            handle = dist.all_reduce(param.grad, op = dist.ReduceOp.SUM, async_op=True)
            self._handles.append((handle,param))
        return hook

    def forward(self,*args,**kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for handle,param in self._handles:
            handle.wait()
            param.grad /= world_size
        self._handles.clear()
        
        


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization() #type:ignore
