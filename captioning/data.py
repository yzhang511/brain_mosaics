"""GPT-2 BPE captions with reserved BOS/PAD, shared across pilots and any
nanoMosaics text pretraining (same tokenizer, same vocab_size)."""
import tiktoken
import torch

_ENC = tiktoken.get_encoding('gpt2')

VOCAB_SIZE = 50304  # GPT-2's 50257 BPE ids padded to a multiple of 64 for efficiency
EOS_ID = _ENC.eot_token  # 50256; tiktoken's own <|endoftext|>, meaningful to any nanoMosaics pretraining
BOS_ID = 50257  # unused reserved slot; marks the image-to-caption transition
PAD_ID = 50258  # unused reserved slot
assert EOS_ID < BOS_ID < PAD_ID < VOCAB_SIZE


def encoded_length(text):
    """Token count including BOS and EOS, i.e. what max_caption_tokens caps."""
    return len(_ENC.encode_ordinary(text)) + 2


def encode(text, max_tokens=64):
    pieces = _ENC.encode_ordinary(text)
    if len(pieces) + 2 > max_tokens:
        raise ValueError('Caption exceeds context limit; review lengths before truncating')
    return [BOS_ID] + pieces + [EOS_ID]


def decode(ids):
    result = []
    for token in ids:
        if token == EOS_ID:
            break
        if token not in (PAD_ID, BOS_ID):
            result.append(token)
    return _ENC.decode(result)


def select_captions(records, rng):
    """One caption per record, drawn uniformly from that record's all_captions."""
    return [rng.choice(record['all_captions']) for record in records]


def caption_batch(records, max_tokens=64, captions=None):
    texts = captions if captions is not None else [record['caption'] for record in records]
    encoded = [encode(text, max_tokens) for text in texts]
    size = max(len(row) - 1 for row in encoded)
    inputs = torch.full((len(encoded), size), PAD_ID, dtype=torch.long)
    targets = torch.full((len(encoded), size), PAD_ID, dtype=torch.long)
    for i, row in enumerate(encoded):
        inputs[i, :len(row) - 1] = torch.tensor(row[:-1])
        targets[i, :len(row) - 1] = torch.tensor(row[1:])
    return inputs, targets
