import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias = False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias = False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f"input dtype: {x.dtype}")
        print(f"fc1 weight dtype: {self.fc1.weight.dtype}")

        x = self.relu(self.fc1(x))
        print(f"fc1 output dtype: {x.dtype}")

        x = self.ln(x)
        print(f"layernorm output dtype: {x.dtype}")
        print(f"layernorm weight dtype: {self.ln.weight.dtype}")

        x = self.fc2(x)
        print(f"fc2 output (logits) dtype: {x.dtype}")
        return x

def main():
    device = "cuda"
    model = ToyModel(in_features=16, out_features=5).to(device)

    x = torch.randn(8, 16, device=device)
    target = torch.randn(8, 5, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = nn.functional.mse_loss(logits, target)
        print(f"loss dtype: {loss.dtype}")

    loss.backward()

    print(f"fc1 weight grad dtype: {model.fc1.weight.grad.dtype}")
    print(f"ln weight grad dtype: {model.ln.weight.grad.dtype}")
    print(f"fc2 weight grad dtype: {model.fc2.weight.grad.dtype}")

    print(f"\nParameter storage dtypes (outside autocast):")
    for name, p in model.named_parameters():
        print(f"  {name}: {p.dtype}")


if __name__ == "__main__":
    main()
