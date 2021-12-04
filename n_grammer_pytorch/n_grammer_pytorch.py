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

# bigram related functions

def multi_way_hash_ids(x, a, b, prime, buckets):
    return ((x * a + b) % prime) % buckets

def get_bigram_ids(ids, vocab_size, segment_pos = None):
    # ids are in shape (batch, seq, heads)

    ids = ids.long()
    ids_0 = F.pad(ids, (0, 0, 0, 1))
    ids_1 = F.pad(ids, (0, 0, 1, 0))

    if exists(segment_pos):
        segment_pos = rearrange(segment_pos, 'b n -> b n 1')
        mask = (segment_pos == 0).long()
        mask = 1 - mask
        mask = F.pad(mask, (0, 0, 0, 1))
        ids_1 *= mask

    ngram_ids = ids_0 + ids_1 * vocab_size
    ngram_ids = ngram_ids[:, :-1]
    return ngram_ids

# optimizer related functions

def get_ngrammer_parameters(module):
    params = set()
    for m in module.modules():
        if isinstance(m, Ngrammer):
            params.update(m.parameters())
    rest = set(module.parameters()) - params
    return list(params), list(rest)

def get_ngrammer_param_groups(module, ngrammer_learning_rate = 1e-2):
    ngrammer_params, rest = get_ngrammer_parameters(module)
    return [{'params': rest}, {'params': ngrammer_params, 'lr': ngrammer_learning_rate}]

# layernorm

class MultiheadLayerNorm(nn.Module):
    def __init__(self, dim, heads = 1, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(heads, dim))
        self.b = nn.Parameter(torch.zeros(heads, dim))

    def forward(self, x):
        std = torch.var(x, dim = -1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

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
        self.num_clusters = num_clusters

        self.register_buffer('means', torch.randn(num_heads, num_clusters, dim_per_head))

    def forward(
        self,
        x,
        mask = None
    ):
        h, dim_head, num_clusters, eps, decay, means = self.num_heads, self.dim_per_head, self.num_clusters, self.epsilon, self.decay, self.means
        assert x.shape[-1] == (h * dim_head), f'input embedding feature dimension must be {h * dim_head}'

        # split heads from input

        x = rearrange(x, 'b n (h d) -> b n h d', h = h)

        # get distance of input embeddings from means

        dists = (
            rearrange(sum_squares(x), 'b n h -> b n h 1')
            - 2 * einsum('b n h d, h k d -> b n h k', x, means)
            + rearrange(sum_squares(means), 'h k -> 1 1 h k')
        )

        # get cluster ids

        cluster_ids = dists.argmin(dim = -1)

        if self.training:
            # get one hot, for calculating number of matches per mean

            nearest_one_hot = F.one_hot(cluster_ids, num_classes = num_clusters)
            per_cluster_count = nearest_one_hot.sum(dim = (0, 1))

            # sum of the input per each closest centroid.

            sum_x = einsum('b n h k, b n h d -> h k d', nearest_one_hot.float(), x)

            # calculate new means

            new_means = sum_x / (eps + rearrange(per_cluster_count, '... -> ... 1'))

            # exponential moving average

            updated_means = (1. - decay) * new_means + decay * means

            self.means.copy_(updated_means)

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

        self.ngram_layernorm = MultiheadLayerNorm(ngram_emb_dim, heads = num_heads)
        self.embeds_layernorm = MultiheadLayerNorm(dim_per_head, heads = num_heads)

        self.ngram_embeds = nn.Embedding(ngram_vocab_size * num_heads, ngram_emb_dim)

        primes = list(sympy.primerange(ngram_vocab_size + 1, 2 * ngram_vocab_size))[:num_heads]
        self.register_buffer('primes', torch.tensor(primes), persistent = False)

    def forward(
        self,
        embeds,
        cluster_ids,
        mask = None,
        segment_pos = None
    ):
        num_heads, vocab_size, unigram_vocab_size, device = self.num_heads, self.ngram_vocab_size, self.unigram_vocab_size, embeds.device

        ngram_cluster_ids = get_bigram_ids(cluster_ids, unigram_vocab_size, segment_pos)

        # prepare arange of heads for parallel computation of multi-way hash ids

        head_range = torch.arange(num_heads, device = device)
        head_range = rearrange(head_range, 'h -> 1 1 h')
        primes = rearrange(self.primes, 'h -> 1 1 h')

        # multi-way hash ids, using https://arxiv.org/abs/1504.06804

        ngram_ids = multi_way_hash_ids(ngram_cluster_ids, head_range + 1, head_range + 1, primes, vocab_size)

        # shift vocab range for each head appropriately by the head number

        ngram_ids = ngram_ids + (vocab_size * head_range)

        # get all n-gram embeddings in one go, and multi-head layernorm

        ngram_embeds = self.ngram_embeds(ngram_ids)
        normed_ngram_embeds = self.ngram_layernorm(ngram_embeds)

        # multi-head layernorm inputs

        embeds = rearrange(embeds, 'b n (h d) -> b n h d', h = num_heads)
        normed_embeds = self.embeds_layernorm(embeds)

        # concat original unigram embeds with bigram

        if self.concat_ngrams:
            input_sliced_dim = normed_embeds.shape[-1] - normed_ngram_embeds.shape[-1]

            out = torch.cat((
                normed_embeds[..., :input_sliced_dim],
                normed_ngram_embeds
            ), dim = -1)
        else:
            out = normed_embeds + normed_ngram_embeds

        # flatten

        out = rearrange(out, 'b n ... -> b n (...)')

        # mask if needed

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
