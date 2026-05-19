import torch
from torch import nn
from collections.abc import Callable, Iterable
from typing import Optional
import math
from numpy import typing as npt
from cs336_basics.tokenizer.tokenizer import Tokenizer
from contextlib import nullcontext
from dataclasses import dataclass


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size = inputs.shape[0]

    # 1. 寻找每行的最大值（用于数值稳定）
    x_max = torch.max(inputs, dim=-1, keepdim=True).values
    # 2. 计算分母的 Log-Sum-Exp 部分
    # inputs - x_max 顺便做了广播机制（Broadcast）

    log_sum_exp = x_max.squeeze(-1) + torch.log(
        torch.sum(torch.exp(inputs - x_max), dim=-1)
    )

    # 3. 提取正确标签的 logit
    target_logits = inputs[torch.arange(batch_size), targets]

    # 4. 计算最终的平均交叉熵
    loss = (-target_logits + log_sum_exp).mean()

    return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"beta must be a tuple of length 2, got: {betas}")

        beta1, beta2 = betas
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                step_size = lr / bias_correction1

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    t,
    a_max,
    a_min,
    t_w,
    t_c,
):
    a_t = 0.0
    if t < t_w:
        a_t = t / t_w * a_max
    elif t_w <= t <= t_c:
        a_t = a_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (
            a_max - a_min
        )
    elif t > t_c:
        a_t = a_min

    return a_t


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5

    clip_coef = max_l2_norm / (eps + total_norm)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
):
    n = dataset.shape[0]
    data = torch.as_tensor(dataset)

    starts = torch.randint(0, n - context_length, (batch_size,), dtype=torch.long)
    offsets = torch.arange(context_length, dtype=torch.long).unsqueeze(0)

    idx = starts.unsqueeze(-1) + offsets

    inputs = data[idx]
    targets = data[idx + 1]

    inputs = inputs.to(device)
    targets = targets.to(device)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str,
):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(state, out)


def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state = torch.load(
        src,
    )
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state["iteration"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_ctx(use_mixed: bool, device: torch.device, verbose: bool = True):
    if use_mixed and device.type == "cuda":

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        return nullcontext()


@dataclass
class BatchState:
    pos: int = 0


def get_batch_sequential(
    data: npt.NDArray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    state: BatchState,
    *,
    stride: int | None = None,
):
    if stride is None:
        stride = context_length

    n = data.numel()
    max_start = n - context_length - 1
    last_start = state.pos + (batch_size - 1) * stride
    end = last_start + context_length + 1

    if end > n:
        state.pos = 0
        last_start = (batch_size - 1) * stride
        end = last_start + context_length + 1

    base = data[state.pos : end]

    inputs = base.as_strided(size=(batch_size, context_length), stride=(stride, 1))
    targets = base[1:].as_strided(size=(batch_size, context_length), stride=(stride, 1))

    state.pos += stride * batch_size

    if (isinstance(device, torch.device) and device.type == "cuda") or (
        isinstance(device, str) and "cuda" in device.lower()
    ):
        inputs = inputs.to(device, non_blocking=True).long()
        targets = targets.to(device, non_blocking=True).long()
    else:
        inputs = inputs.long().to(device)
        targets = targets.long().to(device)

    return inputs, targets


def data_loading_sequential(
    data: npt.NDArray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    state: BatchState,
    *,
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch_sequential(
        data, batch_size, context_length, device, state, stride=stride
    )


if __name__ == "__main__":
    weights = nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.

        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
