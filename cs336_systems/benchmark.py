import argparse
from dataclasses import dataclass
from cs336_basics.model import BasicsTransformerLM
import torch
import timeit

@dataclass
class ModelParams:
    vocab_size: int = 10_000
    context_length: int = 512
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float | None = 10_000.0

MODEL_SIZES: dict[str, dict] = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS336 A2 benchmarking script")
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=list(MODEL_SIZES.keys()),
        help="Model size preset from Table 1",
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--memory_profiling", type=bool, default = True)
    parser.add_argument(
        "--mode",
        type=str,
        default="forward_backward",
        choices=["forward", "forward_backward", "full_step"],
        help="forward = forward only; forward_backward = + backward; full_step = + optimizer step",
    )
    parser.add_argument("--mixed-precision", action="store_true", help="Use BF16 autocast")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> ModelParams:
    size_cfg = MODEL_SIZES[args.size]
    return ModelParams(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **size_cfg,
    )


def benchmark():
    args = parse_args()
    params = build_params(args)

    model = BasicsTransformerLM(
        params.vocab_size,
        params.context_length,
        params.d_model,
        params.num_layers,
        params.num_heads,
        params.d_ff,
        params.rope_theta,
    )
    model.to(args.device)
    torch.compile(model)
    data = torch.randint(
        0, params.vocab_size,
        (args.batch_size, params.context_length),
        device=args.device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def run_step():
        if args.mode == "forward":
            #forward pass only
            with torch.no_grad():
                model(data)
        
        elif args.mode == "forward_backward":
            #forward and backward pass
            optimizer.zero_grad()
            logits = model(data)
            loss = logits.sum()
            loss.backward()

        else:
            # + optimizer step
            optimizer.zero_grad()
            logits = model(data)
            loss = logits.sum()
            loss.backward()
            optimizer.step()

        if args.device == "cuda":
            torch.cuda.synchronize()
    if args.memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(args.warmup_steps):
        run_step()
    if args.memory_profiling:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    times = []
    for _ in range(args.warmup_steps):
        start = timeit.default_timer()
        run_step()
        end = timeit.default_timer()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Mode: {args.mode}, Size: {args.size}, Context: {params.context_length}")
    print(f"Avg time per step: {avg_time*1000:.2f} ms")
    print(f"All step times: {[round(t*1000, 2) for t in times]}")


if __name__ == "__main__":
    benchmark()
