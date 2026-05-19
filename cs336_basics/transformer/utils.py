import einops
import torch
from torch import nn
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
            )
            if bias
            else None
        )

        self._init_weight()

    def _init_weight(self):

        mean = 0.0
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        return x @ self.weight.T


class Emebedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weights = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

    def _init_weight(
        self,
    ):

        mean = 0.0
        std = 1
        nn.init.trunc_normal_(self.weights, mean=mean, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        print(
            "config:",
            self.num_embeddings,
            self.embedding_dim,
        )
        print(x.shape)
        try:
            o = self.weights
        except Exception as e:
            print(
                "config:",
                self.num_embeddings,
                self.embedding_dim,
            )
            print(x.shape)
            raise ValueError("1!!!")
        return self.weights[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def _rms(self, x: torch.Tensor):
        return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x = x.to(torch.float32)

        rms = self._rms(x)
        x_normed = x / rms
        return (x_normed * self.weights).to(input_dtype)


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):

        return self.w2(SiLU(self.w1(x)) * self.w3(x))

    def load_weights(self, w1, w2, w3):
        with torch.no_grad():
            self.w1.weight.copy_(w1)
            self.w2.weight.copy_(w2)
            self.w3.weight.copy_(w3)


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: torch.Tensor):
        x = einops.rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return einops.rearrange(x, "... d j -> ... (d j)")

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0)

        theta = torch.einsum("... i, j -> ... i j", token_positions, self.inv_freq)

        cos = torch.cos(theta).repeat_interleave(2, dim=-1)
        sin = torch.sin(theta).repeat_interleave(2, dim=-1)

        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        return x_rotated


def soft_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x - x_max) / torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
):
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / d_k**0.5

    if mask is not None and isinstance(scores, torch.Tensor):
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weight = soft_max(scores, -1)

    output = attention_weight @ V
    return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 1024,
        device: torch.device = None,
        with_rope: bool = False,
        theta: float = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.out_proj = Linear(d_model, d_model, device=device)

        self.with_rope = with_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        if self.with_rope:
            self.rope = RoPE(theta, self.head_dim, max_seq_len, device)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool)
            ),
        )

    def forward(self, x: torch.Tensor, token_positons: torch.Tensor = None):
        B, S, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 在分头之后添加位置编码
        if self.with_rope:
            Q = self.rope(Q, token_positons)
            K = self.rope(K, token_positons)

        mask = self.mask[:S, :S]
        mask = mask.unsqueeze(0).unsqueeze(0)
        out = scaled_dot_product_attention(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        out = self.out_proj(out)

        return out

    def load_weights(self, q_w, k_w, v_w, o_w):
        """核心：由 Block.load_custom_state_dict 内部调用"""
        with torch.no_grad():
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)
            self.out_proj.weight.copy_(o_w)
        return self
