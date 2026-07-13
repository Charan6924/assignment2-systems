import torch
import os
import time
import torch.distributed as dist
import torch.multiprocessing as mp

warmup_steps = 5
timed_steps = 10

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def make_tensor(size_mb, dtype=torch.float32):
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(size_mb * 1024 * 1024 / bytes_per_elem)
    return torch.rand(num_elements, dtype=dtype)

def run_benchmark(rank, world_size):
    setup(rank, world_size)
    sizes = [1, 10, 100, 1024]
    results = {}

    for size in sizes:
        t = make_tensor(size).to(rank)

        # warmup
        for _ in range(warmup_steps):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # timed runs
        dist.barrier()  
        start = time.perf_counter()
        for _ in range(timed_steps):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / timed_steps * 1000
        results[size] = avg_time_ms

        if rank == 0:
            print(f"[world_size={world_size}] size={size}MB  avg_time={avg_time_ms:.3f} ms")

    cleanup()
    return results

if __name__ == "__main__":
    world_size = 2
    mp.spawn(fn=run_benchmark, args=(world_size,), nprocs=world_size, join=True)

