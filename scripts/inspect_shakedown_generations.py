"""One-off: generate captions from the core-freeze-shakedown-90min checkpoint (job 48056531,
step 5600) on train images, since that job was cancelled before writing its own results.json."""
import json
from pathlib import Path

import torch

from captioning.data import caption_batch
from captioning.model import CaptionConfig, Captioner
from captioning.train import generations

RUN_DIR = Path('/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k')
CKPT = RUN_DIR / 'core_freeze_shakedown_90min' / 'checkpoint.pt'
MANIFEST = Path('reports/caption_pilot_10k/manifest.json')
FEATURES = RUN_DIR / 'features.pt'
PANEL_SIZE = 8
BATCH_SIZE = 16

saved = torch.load(CKPT, map_location='cpu', weights_only=False)
print('checkpoint step:', saved['step'])

manifest = json.loads(MANIFEST.read_text())
cache = torch.load(FEATURES, map_location='cpu', weights_only=True)

train_records = manifest['train']
assert cache['train_ids'] == [r['image_id'] for r in train_records]
inputs, targets = caption_batch(train_records)
features = cache['train']

config = CaptionConfig(**saved['model_config'])
model = Captioner(config)
model.load_state_dict(saved['model'])
model.eval()

panel_size = min(PANEL_SIZE, len(features))
for condition, modified in [('correct', features), ('shuffled', features.roll(1, 0)),
                             ('zero_content', torch.zeros_like(features))]:
    samples = generations(model, modified[:panel_size], train_records[:panel_size], BATCH_SIZE)
    print(f'\n=== train/{condition} ===')
    for s in samples:
        print(f"image_id={s['image_id']} exact_match={s['exact_token_match']}")
        print(f"  reference: {s['reference']!r}")
        print(f"  generated: {s['generated']!r}")
