from rich import print
import torch
import numpy as np
import random


def print_color(content: str, color: str = "green"):
    print(f"[{color}]{content}[/{color}]")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
