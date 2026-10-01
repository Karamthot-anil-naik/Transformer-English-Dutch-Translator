import torch


def init_torch(seed: int = 42) -> torch.device:
    """Initialize PyTorch with a stable random seed and a device."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def main() -> None:
    device = init_torch()
    print(f"PyTorch initialized successfully on {device}.")


__all__ = ["init_torch", "main"]
