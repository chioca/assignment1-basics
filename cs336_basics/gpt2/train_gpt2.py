import torch
from einops import rearrange
from transformers import pipeline, set_seed
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GptConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


# class GPT(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.transfomer = nn.ModuleDict(dict(
#             wte = ,
#         ))


embedding = nn.Embedding(num_embeddings=32000, embedding_dim=768)

token_ids = torch.tensor([101, 2769, 4263])
vecs = embedding(token_ids)
print(vecs[0].shape)
