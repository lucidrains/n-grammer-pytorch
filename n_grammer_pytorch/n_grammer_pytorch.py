import torch
from torch import nn, einsum
from einops import rearrange

class VQNgrammer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
