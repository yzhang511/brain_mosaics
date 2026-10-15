"""Deterministic word vocabulary and fixed-caption pilot batches."""
import re
from collections import Counter

import torch

SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", text.lower())


def build_vocabulary(captions, max_size=4000):
    counts = Counter(word for caption in captions for word in words(caption))
    ordered = sorted(counts, key=lambda word: (-counts[word], word))
    return SPECIALS + ordered[:max_size - len(SPECIALS)]


def encode(text, vocabulary, max_tokens=64):
    lookup = {word: i for i, word in enumerate(vocabulary)}
    pieces = words(text)
    if len(pieces) + 1 > max_tokens:
        raise ValueError('Caption exceeds context limit; review lengths before truncating')
    return [1] + [lookup.get(word, 3) for word in pieces] + [2]


def decode(ids, vocabulary):
    result = []
    for token in ids:
        if token == 2:
            break
        if token not in (0, 1):
            result.append(vocabulary[token])
    return ' '.join(result)


def caption_batch(records, vocabulary, max_tokens=64):
    encoded = [encode(record['caption'], vocabulary, max_tokens) for record in records]
    size = max(len(row) - 1 for row in encoded)
    inputs = torch.zeros(len(encoded), size, dtype=torch.long)
    targets = torch.zeros_like(inputs)
    for i, row in enumerate(encoded):
        inputs[i, :len(row) - 1] = torch.tensor(row[:-1])
        targets[i, :len(row) - 1] = torch.tensor(row[1:])
    return inputs, targets
