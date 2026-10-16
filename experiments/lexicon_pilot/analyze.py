"""Analyze the prespecified lexical pilot; bootstrap noun pairs, not prompts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITIONS = ['neutral', 'consistent', 'conflicting', 'other']


def interval(values, pairs, seed=1729):
    values = np.asarray(values, dtype=float)
    pairs = np.asarray(pairs)
    if not len(values):
        return None
    groups = np.unique(pairs)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(5000):
        sampled = rng.choice(groups, len(groups), replace=True)
        estimates.append(np.concatenate([values[pairs == group] for group in sampled]).mean(axis=0))
    low, high = np.percentile(estimates, [2.5, 97.5], axis=0)
    return dict(mean=values.mean(axis=0).tolist(), low=low.tolist(), high=high.tolist(), clusters=len(groups))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.input / 'scores.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['D'] = float(r['D'])
        for key in ['pair_id', 'frame_id', 'variant']:
            r[key] = int(r[key])
    assert len(rows) == 20*8*2*4
    assert len({(r['noun'], r['frame_id'], r['variant'], r['condition']) for r in rows}) == len(rows)
    nouns = list(dict.fromkeys(r['noun'] for r in rows))
    selected = []
    screening = []
    means, pairs = [], []
    for noun in nouns:
        subset = [r for r in rows if r['noun'] == noun]
        neutral = [r for r in subset if r['condition'] == 'neutral']
        frames = [np.mean([r['D'] for r in neutral if r['frame_id'] == frame]) for frame in range(8)]
        keep = bool(all(v > 0 for v in frames[:4]))
        confirmed = bool(np.mean(frames[4:]) > 0 and sum(v > 0 for v in frames[4:]) >= 3)
        if keep:
            selected.append(noun)
        cond = [float(np.mean([r['D'] for r in subset if r['split'] == 'evaluation' and r['condition'] == c])) for c in CONDITIONS]
        means.append(cond)
        pairs.append(subset[0]['pair_id'])
        screening.append(dict(noun=noun, pair_id=pairs[-1], default=subset[0]['default'], alternative=subset[0]['alternative'],
                              discovery_mean=float(np.mean(frames[:4])), discovery_positive_frames=sum(v > 0 for v in frames[:4]),
                              selected=keep, evaluation_mean=float(np.mean(frames[4:])),
                              evaluation_positive_frames=sum(v > 0 for v in frames[4:]), confirmed=confirmed,
                              **dict(zip(CONDITIONS, cond)),
                              override=cond[2]-cond[0], other_shift=cond[3]-cond[0], specificity=cond[2]-cond[3]))
    means, pairs = np.array(means), np.array(pairs)
    mask = np.array([noun in selected for noun in nouns])
    summary = dict(selected_nouns=selected, selected_count=len(selected), total_nouns=len(nouns),
                   selected_confirmed_count=sum(r['selected'] and r['confirmed'] for r in screening),
                   confirmation_rule='Held-out neutral mean > 0 and at least 3/4 positive template means; not used to reselect.',
                   conditions=CONDITIONS, bootstrap='5000 resamples of noun pairs; fixed templates; seed 1729')
    for label, use in [('all_nouns', np.ones(len(nouns), dtype=bool)), ('selected_nouns_results', mask)]:
        m, p = means[use], pairs[use]
        contrasts = np.column_stack([m[:, 2]-m[:, 0], m[:, 3]-m[:, 0], m[:, 2]-m[:, 3]])
        eval_rows = [r for r in rows if r['split'] == 'evaluation' and (label == 'all_nouns' or r['noun'] in selected)]
        summary[label] = dict(condition_scores=interval(m, p),
                              contrasts=interval(contrasts, p),
                              contrast_names=['target_override', 'other_shift', 'entity_specificity'],
                              fraction_default_preferred={c: float(np.mean([r['D'] > 0 for r in eval_rows if r['condition'] == c])) if eval_rows else None for c in CONDITIONS})
    pair_controls = []
    for pair in np.unique(pairs):
        indices = np.flatnonzero(pairs == pair)
        pair_controls.append(dict(pair_id=int(pair), nouns=[nouns[i] for i in indices],
                                  neutral_evaluation_mean=float(means[indices, 0].mean())))
    summary['pair_controls'] = pair_controls
    summary['pair_controls_positive'] = sum(r['neutral_evaluation_mean'] > 0 for r in pair_controls)
    # Post hoc diagnostic prompted by the unexpectedly weak consistent condition.
    summary['presentation_order_diagnostic'] = {}
    for label, chosen in [('all_nouns', nouns), ('selected_nouns', selected)]:
        variants = {}
        for variant in range(2):
            data = [r for r in rows if r['split'] == 'evaluation' and r['noun'] in chosen and r['variant'] == variant]
            scores = {c: float(np.mean([r['D'] for r in data if r['condition'] == c])) if data else None for c in CONDITIONS}
            variants[str(variant)] = dict(condition_means=scores,
                entity_specificity=scores['conflicting']-scores['other'] if data else None)
        summary['presentation_order_diagnostic'][label] = variants
    # Same candidates reverse orientation within each pair: generic candidate bias cancels in pair mean.
    with (args.out / 'screening.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(screening[0]))
        writer.writeheader()
        writer.writerows(screening)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ['#65788b', '#3c9276', '#c65a57', '#a17b36']
    for ax, label, title in zip(axes, ['all_nouns', 'selected_nouns_results'], ['All 20 nouns', f'Discovery-selected nouns (n={len(selected)})']):
        stat = summary[label]['condition_scores']
        if stat is None:
            ax.text(.5, .5, 'No nouns pass discovery screen', ha='center', transform=ax.transAxes)
            continue
        center, low, high = [np.asarray(stat[k]) for k in ['mean', 'low', 'high']]
        ax.bar(range(4), center, color=colors, alpha=.85)
        ax.errorbar(range(4), center, yerr=[center-low, high-center], fmt='none', color='black', capsize=4)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(range(4), ['Neutral', 'Consistent', 'Conflicting', 'Other entity'], rotation=20)
        ax.set_title(title)
        ax.set_ylabel('log P(default) - log P(alternative) [nats]')
    fig.suptitle('Held-out lexical preference scores')
    fig.text(.5, .01, '95% noun-pair bootstrap intervals; four fixed held-out templates; two presentation variants', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .95])
    fig.savefig(args.out / 'conditions.png', dpi=180)
    fig.savefig(args.out / 'conditions.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8))
    limit = max(abs(means.min()), abs(means.max()))
    im = ax.imshow(means, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_yticks(range(len(nouns)), [n + (' *' if n in selected else '') for n in nouns])
    ax.set_xticks(range(4), ['Neutral', 'Consistent', 'Conflicting', 'Other entity'])
    ax.set_title('Held-out scores by noun (* discovery-selected)')
    for i in range(len(nouns)):
        for j in range(4):
            ax.text(j, i, f'{means[i,j]:.1f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(means[i,j]) > limit*.6 else 'black')
    fig.colorbar(im, ax=ax, label='Default minus alternative log probability [nats]')
    fig.tight_layout()
    fig.savefig(args.out / 'per_noun.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
