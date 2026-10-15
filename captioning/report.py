"""Produce a standalone pilot report from completed local run artifacts."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.run_dir / 'results.json').read_text())
    history = json.loads((args.run_dir / 'history.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ('train', 'val'):
        points = [row for row in history if split in row]
        axes[0].plot([p['step'] for p in points], [p[split]['nll'] for p in points], label=split)
        axes[1].plot([p['step'] for p in points], [p[split]['token_accuracy'] for p in points], label=split)
    axes[0].set_ylabel('Next-token negative log-likelihood (nats)')
    axes[1].set_ylabel('Teacher-forced token accuracy')
    axes[1].set_ylim(0, 1.02)
    for axis in axes:
        axis.set_xlabel('Optimizer step')
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'learning_curve.png', dpi=160)
    plt.close(fig)
    rows, samples = [], []
    for split in ('train', 'val'):
        for condition in ('correct', 'shuffled', 'zero_content'):
            metrics = result['results'][split][condition]
            rows.append(f'<tr><td>{split}</td><td>{condition}</td><td>{metrics["nll"]:.4f}</td>'
                        f'<td>{metrics["token_accuracy"]:.1%}</td><td>{metrics["exact_match_rate"]:.1%}</td></tr>')
        conditions = result['results'][split]
        for index, sample in enumerate(conditions['correct']['generations']):
            lines = [f'<h3>{split}: COCO {sample["image_id"]}</h3>',
                     f'<p><b>Reference:</b> {html.escape(sample["reference"])}</p>']
            for condition in ('correct', 'shuffled', 'zero_content'):
                generation = conditions[condition]['generations'][index]
                lines.append(f'<p><b>{condition}:</b> {html.escape(generation["generated"])}</p>')
            samples.append('<article>' + ''.join(lines) + '</article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>COCO caption pilot</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.5 system-ui;color:#202630}
img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px 16px;border-bottom:1px solid #ddd}
article{padding:12px 0;border-top:1px solid #ddd}p{margin:8px 0}</style>
<h1>32-image fixed-caption overfit pilot</h1>
<p>This is a pipeline and memorization check. Eight validation images are too few to
establish generalization. Validation uses the pilot's training-only word vocabulary;
unknown words limit interpretation. No reserved test images were evaluated.</p>'''
    gpu_memory = result['peak_gpu_memory_bytes']
    memory_text = 'not measured (CPU run)' if gpu_memory is None else f'{gpu_memory / 2**20:.1f} MiB'
    page += f'<p>Final step: {result["step"]}. Median step time: {result["median_step_seconds"]:.3f} s. Peak GPU memory: {memory_text}.</p>'
    page += '<img src="learning_curve.png" alt="Training and validation loss and teacher-forced accuracy">'
    page += '<table><tr><th>Split</th><th>Image condition</th><th>NLL</th><th>Token accuracy</th><th>Exact generation</th></tr>' + ''.join(rows) + '</table>'
    page += '<p>Shuffled images use a fixed one-position rotation with no self-matches. Zero-content replaces raw cached features with zeros and retains adapter bias, spatial positions, and image-type embedding. Exact match includes EOS and uses tokenized captions.</p>'
    page += '<p>Validation unknown-token rate: ' + f'{result["results"]["val"]["unknown_token_rate"]:.1%}' + '.</p>'
    page += '<h2>Free generations</h2>' + ''.join(samples) + '</html>'
    (args.output_dir / 'index.html').write_text(page)


if __name__ == '__main__':
    main()
