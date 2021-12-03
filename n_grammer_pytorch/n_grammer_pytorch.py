import torch
from torch import nn, einsum
from einops import rearrange
import sympy

# helper functions

def exists(val):
    return val is not None

def get_bigram_ids(ids, vocab_size, segment_pos = None):
    assert vocab_size > 0
    batch_size = ids.shape[0]

    ids = ids.long()
    pad = torch.zeros((batch_size, 1), dtype = torch.long)

    ids_0 = torch.cat((ids, pad), dim = -1)
    ids_1 = torch.cat((pad, ids), dim = -1)

    if exists(segment_pos):
        mask = (segment_pos == 0).long()
        mask = 1 - mask
        mask = torch.cat((mask, pad), dim = 1)
        ids_1 *= mask

    ngram_ids = ids_0 + ids_1 * vocab_size
    ngram_ids = ngram_ids[:, :-1]
    return ngram_ids

# classes

class VectorQuantization(nn.Module):
    def __init__(
        self,
        *,
        num_clusters,
        num_heads,
        dim_per_head,
        decay = 0.999,
        epsilon = 1e-6
    ):
        super().__init__()

    def forward(self, x):
        return x

class Ngrammer(nn.Module):
    def __init__(
        self,
        *,
        unigram_vocab_size,
        num_heads,
        dim_per_head,
        ngram_emb_dim = 8,
        ngram_vocab_size = 768 * 256,
        concat_ngrams = True
    ):
        super().__init__()

    def forward(
        self,
        x,
        cluster_ids,
        mask = None,
        segment_pos = None
    ):
        return x

# main class

class VQNgrammer(nn.Module):
    def __init__(
        self,
        *,
        num_clusters,
        num_heads,
        dim_per_head,
        ngram_vocab_size = 768 * 256,
        ngram_emb_dim = 8,
        concat_ngrams = True,
        decay = 0.999,
        epsilon = 1e-6
    ):
        super().__init__()

        self.vq = VectorQuantization(
            num_clusters = num_clusters,
            num_heads = num_heads,
            dim_per_head = dim_per_head,
            decay = decay,
            epsilon = epsilon
        )

        self.ngram = Ngrammer(
            unigram_vocab_size = num_clusters,
            ngram_vocab_size = ngram_vocab_size,
            ngram_emb_dim = ngram_emb_dim,
            concat_ngrams = concat_ngrams,
            num_heads = num_heads,
            dim_per_head = dim_per_head
        )

    def forward(
        self,
        x,
        mask = None,
        segment_pos = None
    ):

        return x
