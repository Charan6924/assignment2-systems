import torch.distributed as dist
from typing import Any, Type
from torch.optim import Optimizer
import torch

class ShardedOptimzer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls : Type[Optimizer], **kwargs):
        self.optimizer_cls = optimizer_cls
        self.rank = dist.get_rank()
        self.kwargs = kwargs
        self.world_size = dist.get_world_size()

        self.param_to_rank = {}
        self.local_optimizer = None
        self._all_params = []

        super().__init__(params, kwargs)
        self._build_local_optimizer()


    def _build_local_optimizer(self):
        local_params = []
        for group in self.param_groups:
            for param in group["params"]:
                if param not in self.param_to_rank:
                    owner = len(self.param_to_rank) % self.world_size
                    self.param_to_rank[param] = owner
                self._all_params.append(param)
                if self.param_to_rank[param] == self.rank:
                    local_params.append(param)

        self.local_optimizer = self.optimizer_cls(local_params, **self.kwargs)

    def step(self,closure=None,**kwargs):
        loss = self.local_optimizer.step(closure,**kwargs)

        for param in self._all_params:
            owner = self.param_to_rank[param]
            dist.broadcast(param.data,src=owner)


    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        self._build_local_optimizer()
