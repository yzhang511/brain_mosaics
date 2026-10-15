"""Visual-prefix captioning using the upstream nanoMosaics memory blocks."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nanoMosaics.mosaic_model import MemoryMosaic, MemoryMosaicConfig


@dataclass
class CaptionConfig:
    vocab_size: int = 50304  # tiktoken gpt2 BPE (50257), padded to a multiple of 64
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    pmem_size: int = 256
    grid_size: int = 14
    feature_dim: int = 128
    max_caption_tokens: int = 64  # Includes BOS; targets include EOS.
    dropout: float = 0.0
    bias: bool = False  # matches nanoMosaics pretraining default, so checkpoints load cleanly
    pad_id: int = 50258
    bos_id: int = 50257
    eos_id: int = 50256


def spatial_positions(grid_size, dim):
    """Fixed sinusoidal row/column encoding, flattened in raster order."""
    if dim % 4:
        raise ValueError('n_embd must be divisible by four')
    frequency = 10000 ** (-torch.arange(dim // 4).float() / (dim // 4))
    rows, cols = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    parts = []
    for coordinate in (rows, cols):
        angle = coordinate.flatten().float()[:, None] * frequency
        parts.extend((angle.sin(), angle.cos()))
    return torch.cat(parts, dim=-1)


class Captioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if len({config.pad_id, config.bos_id, config.eos_id}) != 3:
            raise ValueError('PAD, BOS and EOS must be distinct')
        if not all(0 <= i < config.vocab_size for i in (config.pad_id, config.bos_id, config.eos_id)):
            raise ValueError('Special token IDs must be in the vocabulary')
        self.core = MemoryMosaic(MemoryMosaicConfig(
            vocab_size=config.vocab_size,
            block_size=config.grid_size ** 2 + config.max_caption_tokens,
            n_layer=config.n_layer, n_embd=config.n_embd, n_head=config.n_head,
            pmem_size=config.pmem_size, pmem_count=1, dropout=config.dropout, bias=config.bias,
        ))
        self.adapter = nn.Linear(config.feature_dim, config.n_embd)
        self.image_type = nn.Parameter(torch.zeros(config.n_embd))
        self.register_buffer('positions', spatial_positions(config.grid_size, config.n_embd))
        self.adapter.apply(self.core._init_weights)
        nn.init.normal_(self.image_type, std=0.02)

    def forward(self, features, tokens, targets=None, return_states=False):
        """Features: B,C,H,W; tokens: BOS + words; targets: words + EOS.

        Right-pad tokens only. Target PAD and -1 positions are ignored. No loss
        is calculated at visual positions. Each call constructs fresh context.
        """
        c = self.config
        if features.ndim != 4 or tuple(features.shape[1:]) != (c.feature_dim, c.grid_size, c.grid_size):
            raise ValueError('Expected features shaped B,feature_dim,grid_size,grid_size')
        if tokens.ndim != 2 or tokens.shape[0] != features.shape[0] or not 1 <= tokens.shape[1] <= c.max_caption_tokens:
            raise ValueError('Invalid caption batch or sequence length')
        visual = self.adapter(features.flatten(2).transpose(1, 2))
        visual = visual + self.positions + self.image_type
        x = self.core.mosaic.drop(torch.cat((visual, self.core.mosaic.wte(tokens)), dim=1))
        for block in self.core.mosaic.h:
            x = block(x)
        x = self.core.mosaic.ln_f(x)
        logits = self.core.lm_head(x[:, c.grid_size ** 2:])
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError('Targets must match token shape')
            labels = targets.masked_fill(targets == c.pad_id, -1)
            if not (labels != -1).any():
                raise ValueError('At least one caption target is required')
            loss = F.cross_entropy(logits.reshape(-1, c.vocab_size), labels.reshape(-1), ignore_index=-1)
        if return_states:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, features, max_new_tokens=None):
        """Greedy decoding; recompute full prefix, stop each example at EOS."""
        c = self.config
        limit = c.max_caption_tokens if max_new_tokens is None else max_new_tokens
        if not 1 <= limit <= c.max_caption_tokens:
            raise ValueError('Generation limit exceeds configured caption context')
        was_training = self.training
        self.eval()
        try:
            tokens = torch.full((features.shape[0], 1), c.bos_id, dtype=torch.long, device=features.device)
            finished = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
            for _ in range(limit):
                logits, _ = self(features, tokens)
                scores = logits[:, -1].clone()
                scores[:, [c.pad_id, c.bos_id]] = -torch.inf
                next_token = scores.argmax(-1).masked_fill(finished, c.pad_id)
                tokens = torch.cat((tokens, next_token[:, None]), dim=1)
                finished |= next_token == c.eos_id
                if finished.all():
                    break
            return tokens
        finally:
            self.train(was_training)
