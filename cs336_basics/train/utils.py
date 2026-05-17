import torch
from torch import nn
from collections.abc import Callable, Iterable
from typing import Optional
import math


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size = inputs.shape[0]

    # 1. 寻找每行的最大值（用于数值稳定）
    x_max = torch.max(inputs, dim=-1, keepdim=True).values
    print("x_max", x_max)
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


if __name__ == "__main__":
    weights = nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.

        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
