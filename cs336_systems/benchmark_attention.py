import timeit
import torch
from cs336_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]          # d_k (head embedding dimension)
SEQ_LENS = [256, 1024, 4096, 8192, 16384]  # sequence lengths
WARMUP = 5
MEASURE = 100
DEVICE = "cuda"
compiled_attention = torch.compile(scaled_dot_product_attention)
results = []

def benchmark():
    for d_k in HEAD_DIMS:
        for seq_len in SEQ_LENS:
            print(f"d_k={d_k}, seq_len={seq_len}", end=" ... ", flush=True)

            Q = torch.randn(BATCH_SIZE, seq_len, d_k, device=DEVICE, requires_grad=True)
            K = torch.randn(BATCH_SIZE, seq_len, d_k, device=DEVICE, requires_grad=True)
            V = torch.randn(BATCH_SIZE, seq_len, d_k, device=DEVICE, requires_grad=True)

            try:
                # warmup
                for _ in range(WARMUP):
                    out = compiled_attention(Q, K, V)
                    torch.cuda.synchronize()

                # benchmark forward
                fwd_times = []
                for _ in range(MEASURE):
                    start = timeit.default_timer()
                    out = compiled_attention(Q, K, V)
                    torch.cuda.synchronize()
                    fwd_times.append(timeit.default_timer() - start)

                fwd_mean = sum(fwd_times) / len(fwd_times)

                # memory before backward
                mem_before_bwd = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)  # MiB

                # benchmark backward
                bwd_times = []
                for _ in range(MEASURE):
                    out = compiled_attention(Q, K, V)
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    out.sum().backward()
                    torch.cuda.synchronize()
                    bwd_times.append(timeit.default_timer() - start)

                bwd_mean = sum(bwd_times) / len(bwd_times)

                print(f"fwd={fwd_mean*1e3:.2f}ms, bwd={bwd_mean*1e3:.2f}ms, mem_before_bwd={mem_before_bwd:.1f}MiB")
                results.append((d_k, seq_len, fwd_mean, bwd_mean, mem_before_bwd, None))

            except torch.cuda.OutOfMemoryError:
                print("OOM")
                results.append((d_k, seq_len, None, None, None, "OOM"))
                torch.cuda.empty_cache()

if __name__ == "__main__":
    benchmark()
    print("\nd_k | seq_len | fwd(ms) | bwd(ms) | mem_before_bwd(MiB)")
    for d_k, seq_len, fwd, bwd, mem, err in results:
        if err:
            print(f"{d_k:4d} | {seq_len:7d} | OOM")
        else:
            print(f"{d_k:4d} | {seq_len:7d} | {fwd*1e3:7.2f} | {bwd*1e3:7.2f} | {mem:8.1f}")
