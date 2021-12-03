# based off the jax code
# https://github.com/tensorflow/lingvo/blob/master/lingvo/jax/layers/ngrammer.py

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
import sympy

# helper functions

def exists(val):
    return val is not None

def sum_squares(t, dim = -1):
    return (t ** 2).sum(dim = dim)


def multi_way_hash_ids(x, a, b, prime, buckets):
    return ((x * a + b) % prime) % buckets

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
        self.decay = decay
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.register_buffer('means', torch.randn(num_heads, num_clusters, dim_per_head))

    def forward(
        self,
        x,
        mask = None
    ):
        h, dim_head, means = self.num_heads, self.dim_per_head, self.means
        assert x.shape[-1] == (h * dim_head), f'input embedding feature dimension must be {h * dim_head}'

        # split heads from input

        x = rearrange(x, 'b n (h d) -> b n h d', h = h)

        # get distance of input embeddings from means

        dists = (
            rearrange(sum_squares(x), 'b n h -> b n h 1') -
            2 * einsum('b n h d, h k d -> b n h k', x, means) +
            rearrange(sum_squares(means), 'h k -> 1 1 h k')
        )

        # get cluster ids

        cluster_ids = dists.argmin(dim = -1)
        return cluster_ids

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
        self.num_heads = num_heads
        self.ngram_vocab_size = ngram_vocab_size
        self.unigram_vocab_size = unigram_vocab_size
        self.concat_ngrams = concat_ngrams

        self.embeddings = nn.ModuleList([])
        self.ngram_norm = nn.ModuleList([])
        self.input_norm = nn.ModuleList([])

        for _ in range(num_heads):
            self.ngram_norm.append(nn.LayerNorm(ngram_emb_dim))
            self.input_norm.append(nn.LayerNorm(dim_per_head))

            head_embed = nn.Embedding(ngram_vocab_size, ngram_emb_dim)
            self.embeddings.append(head_embed)

        primes = list(sympy.primerange(ngram_vocab_size + 1, 2 * ngram_vocab_size))[:num_heads]
        self.primes = primes

    def forward(
        self,
        embeds,
        cluster_ids,
        mask = None,
        segment_pos = None
    ):
        num_heads, vocab_size, unigram_vocab_size = self.num_heads, self.ngram_vocab_size, self.unigram_vocab_size

        normed_embeds = []
        normed_ngram_embeds = []

        embeds = rearrange(embeds, 'b n (h d) -> b n h d', h = num_heads)

        for head_num, prime, input_ids_per_head, embed, ngram_emb, embed_norm, ngram_norm in zip(range(num_heads), self.primes, cluster_ids.unbind(dim = -1), embeds.unbind(dim = -2), self.embeddings, self.input_norm, self.ngram_norm):
            ngram_ids = get_bigram_ids(input_ids_per_head, unigram_vocab_size, segment_pos)
            ngram_ids_for_head = multi_way_hash_ids(ngram_ids, head_num + 1, head_num + 1, prime, vocab_size)

            ngram_embed = ngram_emb(ngram_ids_for_head)
            normed_ngram_embed = ngram_norm(ngram_embed)
            normed_ngram_embeds.append(normed_ngram_embed)

            normed_embed = embed_norm(embed)
            normed_embeds.append(normed_embed)

        normed_embeds = torch.stack(normed_embeds, dim = -2)
        normed_ngram_embeds = torch.stack(normed_ngram_embeds, dim = -2)

        if self.concat_ngrams:
            input_sliced_dim = normed_embeds.shape[-1] - normed_ngram_embeds.shape[-1]

            out = torch.cat((
                normed_embeds[..., :input_sliced_dim],
                normed_ngram_embeds
            ), dim = -1)
        else:
            out = normed_embeds + normed_ngram_embeds

        out = rearrange(out, 'b n ... -> b n (...)')

        if exists(mask):
            out = out * rearrange(mask, 'b n -> b n 1').float()

        return out

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

        cluster_ids = self.vq(x, mask = mask)

        out = self.ngram(
            x,
            cluster_ids = cluster_ids,
            mask = mask,
            segment_pos = segment_pos
        )

        return out
