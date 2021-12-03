<img src="./n-grammer.png" width="400px"></img>

## N-Grammer - Pytorch

Implementation of <a href="https://openreview.net/forum?id=GxjCYmQAody">N-Grammer</a>, augmenting Transformers with latent n-grams, in Pytorch

## Install

```bash
$ pip install n-grammer-pytorch
````

## Usage

```python
import torch
from n_grammer_pytorch import VQNgrammer

vq_ngram = VQNgrammer(
    num_clusters = 1024,             # number of clusters
    dim_per_head = 32,               # dimension per head
    num_heads = 16,                  # number of heads
    ngram_vocab_size = 768 * 256,    # ngram vocab size
    ngram_emb_dim = 16,              # ngram embedding dimension
    decay = 0.999                    # exponential moving decay value
)

x = torch.randn(1, 1024, 32 * 16)
vq_ngram(x) # (1, 1024, 32 * 16)
```

## Learning Rates

Like product key memories, Ngrammer parameters need to have a higher learning rate (`1e-2` was recommended in the paper). The repository offers an easy way to generate the parameter groups.


```python
from torch.optim import Adam
from n_grammer_pytorch import get_ngrammer_parameters

# this helper function, for your root model, finds all the VQNgrammer models and the embedding parameters
ngrammer_parameters, other_parameters = get_ngrammer_parameters(transformer)

optim = Adam([
    {'params': other_parameters},
    {'params': ngrammer_parameters, 'lr': 1e-2}
], lr = 3e-4)
```

Or, even more simply

```python
from torch.optim import Adam
from n_grammer_pytorch import get_ngrammer_param_groups

param_groups = get_ngrammer_param_groups(model) # automatically creates array of parameter settings with learning rate set at 1e-2 for ngrammer parameter values
optim = Adam(param_groups, lr = 3e-4)
```

## Citations

```bibtex
@inproceedings{thai2020using,
    title   = {N-grammer: Augmenting Transformers with latent n-grams},
    author  = {Anonymous},
    year    = {2021},
    url     = {https://openreview.net/forum?id=GxjCYmQAody}
}
```
