from torch import nn
from cs336_basics.transformer.utils import MultiHeadAttention, RMSNorm, SwiGLU
import torch


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = None,
        max_seq_len: int = 1024,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.atten = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            device=device,
            with_rope=True,
            theta=theta,
        )
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.swi_glu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        h = x + self.atten(self.ln1(x))

        normed_h = self.ln2(h)

        return self.swi_glu(normed_h) + h

    def load_custom_state_dict(self, weights: dict[str, torch.Tensor]):
        """
        接传入 weights 字典，全自动对齐并加载所有权重
        """
        with torch.no_grad():
            self.atten.load_weights(
                q_w=weights["attn.q_proj.weight"],
                k_w=weights["attn.k_proj.weight"],
                v_w=weights["attn.v_proj.weight"],
                o_w=weights["attn.output_proj.weight"],
            )

            self.ln1.weights.copy_(weights["ln1.weight"])
            self.ln2.weights.copy_(weights["ln2.weight"])
            self.swi_glu.load_weights(
                weights["ffn.w1.weight"],
                weights["ffn.w2.weight"],
                weights["ffn.w3.weight"],
            )
        return self  # 支持链式调用
