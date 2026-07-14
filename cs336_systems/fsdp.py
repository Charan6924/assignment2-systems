import torch
import torch.nn as nn
import torch.distributed as dist


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.compute_dtype = compute_dtype

        self.sharded_layers = {}          
        self.pending_work = []            
        self.replicated_grad_work = []  

        sharded_module_names = set()
        for name, submodule in self.module.named_modules():
            if isinstance(submodule, (torch.nn.Linear, torch.nn.Embedding)):
                sharded_module_names.add(name)

                w_shard = nn.Parameter(self._shard_tensor(submodule.weight.data), requires_grad=True)
                self.sharded_layers[name] = {
                    "module": submodule,
                    "weight_shard": w_shard,
                    "weight_shape": submodule.weight.shape,
                    "bias_shard": None,
                    "bias_shape": None,
                    "gather_work": None,
                    "gathered_chunks": None,
                }
                self.register_parameter(f"{name.replace('.', '_')}_weight_shard", w_shard)

                if getattr(submodule, "bias", None) is not None:
                    b_shard = nn.Parameter(self._shard_tensor(submodule.bias.data), requires_grad=True)
                    self.sharded_layers[name]["bias_shard"] = b_shard
                    self.sharded_layers[name]["bias_shape"] = submodule.bias.shape
                    self.register_parameter(f"{name.replace('.', '_')}_bias_shard", b_shard)

        for pname, param in self.module.named_parameters():
            parent_name = pname.rsplit(".", 1)[0] if "." in pname else ""
            if parent_name in sharded_module_names:
                continue  # handled by shard reduce-scatter instead
            param.register_hook(self._make_replicated_grad_hook(pname))

    def _shard_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(-1)
        numel = flat.numel()
        pad_size = (self.world_size - numel % self.world_size) % self.world_size
        if pad_size > 0:
            flat = torch.cat([flat, flat.new_zeros(pad_size)])
        shard_size = flat.numel() // self.world_size
        return flat[self.rank * shard_size:(self.rank + 1) * shard_size].clone()

    def _start_gather(self, name):
        info = self.sharded_layers[name]
        if info["gather_work"] is not None:
            return  # already in flight

        w_shard = info["weight_shard"].data
        w_gathered = [torch.empty_like(w_shard) for _ in range(self.world_size)]
        w_work = dist.all_gather(w_gathered, w_shard.contiguous(), async_op=True)

        b_gathered, b_work = None, None
        if info["bias_shard"] is not None:
            b_shard = info["bias_shard"].data
            b_gathered = [torch.empty_like(b_shard) for _ in range(self.world_size)]
            b_work = dist.all_gather(b_gathered, b_shard.contiguous(), async_op=True)

        info["gathered_chunks"] = (w_gathered, b_gathered)
        info["gather_work"] = (w_work, b_work)

    def _wait_gather(self, name):
        info = self.sharded_layers[name]
        w_work, b_work = info["gather_work"]
        w_gathered, b_gathered = info["gathered_chunks"]

        w_work.wait()
        flat_full_w = torch.cat(w_gathered)
        numel_w = info["weight_shape"].numel()
        full_w = flat_full_w[:numel_w].view(info["weight_shape"])
        if self.compute_dtype is not None:
            full_w = full_w.to(self.compute_dtype)

        full_b = None
        if b_work is not None:
            b_work.wait()
            flat_full_b = torch.cat(b_gathered)
            numel_b = info["bias_shape"].numel()
            full_b = flat_full_b[:numel_b].view(info["bias_shape"])
            if self.compute_dtype is not None:
                full_b = full_b.to(self.compute_dtype)

        info["gather_work"] = None
        info["gathered_chunks"] = None
        return full_w, full_b

    def _start_reduce_scatter(self, name, full_grad: torch.Tensor, is_bias: bool):
        flat = full_grad.reshape(-1)
        pad_size = (self.world_size - flat.numel() % self.world_size) % self.world_size
        if pad_size > 0:
            flat = torch.cat([flat, flat.new_zeros(pad_size)])
        shard_size = flat.numel() // self.world_size
        chunks = list(flat.chunk(self.world_size))
        output = torch.empty(shard_size, device=flat.device, dtype=flat.dtype)
        work = dist.reduce_scatter(output, chunks, async_op=True)
        self.pending_work.append((work, name, output, is_bias))

    def _register_grad_hook(self, name, param, is_bias):
        def grad_hook(grad):
            self._start_reduce_scatter(name, grad, is_bias)
            return grad
        param.register_hook(grad_hook)

    def _make_replicated_grad_hook(self, pname):
        def hook(grad):
            work = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
            self.replicated_grad_work.append((work, grad))
            return grad
        return hook

    def forward(self, *args, **kwargs):
        names = list(self.sharded_layers.keys())

        # prime the pipeline
        if len(names) > 0:
            self._start_gather(names[0])
        if len(names) > 1:
            self._start_gather(names[1])

        for i, name in enumerate(names):
            info = self.sharded_layers[name]
            mod = info["module"]

            full_w, full_b = self._wait_gather(name)

            if i + 2 < len(names):
                self._start_gather(names[i + 2])

            mod.weight = nn.Parameter(full_w, requires_grad=True)
            self._register_grad_hook(name, mod.weight, is_bias=False)

            if full_b is not None:
                mod.bias = nn.Parameter(full_b, requires_grad=True)
                self._register_grad_hook(name, mod.bias, is_bias=True)

        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for work, name, sharded_grad, is_bias in self.pending_work:
            work.wait()
            info = self.sharded_layers[name]
            target = info["bias_shard"] if is_bias else info["weight_shard"]
            if target.grad is None:
                target.grad = sharded_grad
            else:
                target.grad += sharded_grad
        self.pending_work.clear()

        for work, grad in self.replicated_grad_work:
            work.wait()
            grad /= self.world_size
        self.replicated_grad_work.clear()


    def gather_full_state_dict(self) -> dict[str, torch.Tensor]:
        state_dict = {}
        sharded_names = set(self.sharded_layers.keys())

        for name, info in self.sharded_layers.items():
            w_shard = info["weight_shard"].data
            w_gathered = [torch.empty_like(w_shard) for _ in range(self.world_size)]
            dist.all_gather(w_gathered, w_shard.contiguous())
            flat_w = torch.cat(w_gathered)
            full_w = flat_w[:info["weight_shape"].numel()].view(info["weight_shape"])
            state_dict[f"{name}.weight"] = full_w

            if info["bias_shard"] is not None:
                b_shard = info["bias_shard"].data
                b_gathered = [torch.empty_like(b_shard) for _ in range(self.world_size)]
                dist.all_gather(b_gathered, b_shard.contiguous())
                flat_b = torch.cat(b_gathered)
                full_b = flat_b[:info["bias_shape"].numel()].view(info["bias_shape"])
                state_dict[f"{name}.bias"] = full_b

        for pname, param in self.module.named_parameters():
            parent_name = pname.rsplit(".", 1)[0] if "." in pname else ""
            if parent_name in sharded_names:
                continue
            state_dict[pname] = param.data.clone()

        return state_dict
