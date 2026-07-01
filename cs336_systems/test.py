import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import TransformerBlock, RotaryEmbedding

d_model, d_ff, num_heads, context_length = 2560, 10240, 32, 2048
num_layers = 32

blocks = torch.nn.ModuleList([
    TransformerBlock(
        d_model=d_model, d_ff=d_ff, num_heads=num_heads,
        positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length)
    )
    for _ in range(num_layers)
]).to('cuda')

def one_block(x, block):
    return block(x)

def checkpoint_every_block(x):
    for block in blocks:
        x = checkpoint(one_block, x, block, use_reentrant=False)
    return x

x = torch.randn((4, context_length, d_model), device='cuda', requires_grad=True)

torch.cuda.reset_peak_memory_stats()
y = checkpoint_every_block(x)
loss = y.sum()
loss.backward()
peak = torch.cuda.max_memory_allocated() / (1024**2)
print(f"group_size=1: peak memory = {peak:.2f} MiB")
