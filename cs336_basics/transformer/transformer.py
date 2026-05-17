from torch import nn
from cs336_basics.transformer.block import Block
from cs336_basics.transformer.utils import Emebedding, RMSNorm, Linear
import torch


class transformer_llm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = None,
        max_seq_len: int = 1024,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.embed = Emebedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    theta=theta,
                    max_seq_len=max_seq_len,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        embed = self.embed(x)

        for layer in self.layers:
            embed = layer(embed)

        out = self.rms_norm(embed)
        out = self.out_proj(out)

        return out

    def load_weight_from_dict(self, weights: dict[str, torch.Tensor]):
        with torch.no_grad():
            self.embed.weights.copy_(weights["token_embeddings.weight"])
            for i, layer in enumerate(self.layers):
                s = f"layers.{i}."
                layer.atten.load_weights(
                    q_w=weights[f"{s}attn.q_proj.weight"],
                    k_w=weights[f"{s}attn.k_proj.weight"],
                    v_w=weights[f"{s}attn.v_proj.weight"],
                    o_w=weights[f"{s}attn.output_proj.weight"],
                )

                layer.ln1.weights.copy_(weights[f"{s}ln1.weight"])
                layer.ln2.weights.copy_(weights[f"{s}ln2.weight"])
                layer.swi_glu.load_weights(
                    weights[f"{s}ffn.w1.weight"],
                    weights[f"{s}ffn.w2.weight"],
                    weights[f"{s}ffn.w3.weight"],
                )

            self.rms_norm.weights.copy_(weights["ln_final.weight"])
            self.out_proj.weight.copy_(weights["lm_head.weight"])
        return self
