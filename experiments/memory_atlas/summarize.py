"""Export figures, descriptive summary and a self-contained report."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args();out=args.input
    A=json.loads((out/'atlas.json').read_text());H=A['heads'];P=A['prompts'];M=A['metadata']
    assert len(H)==144 and len(P)==24
    metrics=[('context_half_life','Contextual key half-life','tokens'),('persistent_half_life','Persistent key half-life','tokens'),('key_nearest_cos_mean','Nearest stored-key similarity','cosine'),('value_cos_among_key_neighbors','Value similarity among key neighbors','cosine'),('mean_effective_slots','Effective retrieved slots','exp(entropy)'),('mean_conflict_other_TV','Contradictory–irrelevant retrieval distance','total variation')]
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for ax,(key,title,label) in zip(axes.flat,metrics):
        v=np.array([h['metrics'][key] for h in H]).reshape(12,12)
        im=ax.imshow(v,aspect='auto',cmap='viridis');ax.set(title=title,xlabel='Head (zero-based)',ylabel='Layer (zero-based)',xticks=range(12),yticks=range(12));fig.colorbar(im,ax=ax,label=label,shrink=.85)
    fig.suptitle('Example-guided memory atlas · BabiStories step 23,000\nWeight geometry and retrieval across 24 prompts; descriptive, not causal',fontsize=16)
    for ext in ['png','svg']:fig.savefig(out/f'overview.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    summary=[]
    for c,ax in enumerate(axes.flat):
        tv=np.array([h['tv'][c] for h in H]).reshape(12,12);p=P[c*4]
        ax.plot(range(12),tv.mean(1),'o-',color='#217a9b',label='Mean across heads');ax.fill_between(range(12),tv.min(1),tv.max(1),alpha=.15,color='#217a9b',label='Head range')
        ax.set(title=f"{p['animal']} · frame {p['frame']} · variant {p['variant']}",xlabel='Layer (zero-based)',ylabel='Contradictory–irrelevant TV',ylim=(0,1),xticks=range(12));ax.grid(alpha=.2)
        row=dict(animal=p['animal'],frame=p['frame'],variant=p['variant'],mean_TV=float(tv.mean()),max_TV=float(tv.max()),max_layer=int(np.unravel_index(tv.argmax(),tv.shape)[0]),max_head=int(np.unravel_index(tv.argmax(),tv.shape)[1]));summary.append(row)
    axes[0,0].legend(fontsize=8);fig.suptitle('Persistent retrieval changes between target and companion preference statements\nFull 2,688-slot distributions at the final prompt position',fontsize=15)
    for ext in ['png','svg']:fig.savefig(out/f'per_example.{ext}',dpi=180)
    plt.close(fig)
    with (out/'example_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    top=sorted(H,key=lambda h:h['metrics']['mean_conflict_other_TV'],reverse=True)[:10]
    cosine=np.array([h['output_cosine'] for h in H]);tv=np.array([h['tv'] for h in H]);early=tv.reshape(12,12,6)[:4].mean();late=tv.reshape(12,12,6)[8:].mean()
    def span(key):
        vals=[h['metrics'][key] for h in H];return f'{min(vals):.3f}–{max(vals):.3f}'
    report=f'''# Example-guided memory atlas

Built from the unchanged 23,000-step `babistories-12L768d` checkpoint. This is a descriptive atlas, not a causal intervention experiment.

Open [the standalone interactive atlas](atlas.html). Static exports: [overview](overview.png), [vector overview](overview.svg), [per-example retrieval changes](per_example.png), and [vector per-example figure](per_example.svg).

## Coverage and validation

- All six selected examples, all four conditions: 24 prompts spanning five animals.
- All 12 layers and 12 heads: 144 heads and 387,072 persistent slots.
- Full persistent retrieval distributions and contextual source-position weights at the final prompt token.
- Full slot geometry; detailed profiles for {sum(len(h['profiles']) for h in H):,} slots selected by a fixed descriptive rule.
- Maximum memory-output reconstruction error: {M['module_reconstruction_max_error']:.3g}.
- Maximum complete-continuation D discrepancy from saved pilot scores: {M['baseline_max_error']:.3g} nats.
- Maximum independent reference-versus-batch scoring discrepancy: {M['reference_scoring_max_error']:.3g} nats.
- Maximum synthetic head-projection reconstruction error: {M['synthetic_projection_max_error']:.3g}.
- CPU float32, one thread; extraction took {M['elapsed_seconds']:.1f} seconds. All numerical checks passed their declared thresholds (1e-4 for output/scoring checks, 1e-5 for synthetic projection).

## Descriptive findings

Across the 864 example/head comparisons, contradictory-versus-irrelevant persistent retrieval total variation averages {tv.mean():.4f}, with range {tv.min():.4f}–{tv.max():.4f}. The first four layers average {early:.4f}; the last four average {late:.4f}. These changes show sensitivity to which animal receives the preference, not that any particular head causes the behavioral distinction.

The corresponding retrieved projected-head-output cosine averages {cosine.mean():.4f}, with range {cosine.min():.4f}–{cosine.max():.4f}. Changes in slot routing need not imply equally large changes in the resulting vector, because values can be similar or cancel.

Two-dimensional PCA explains {span('key_pca_variance')} of key variance and {span('value_pca_variance')} of projected-value variance across heads. The scatterplots are incomplete views: nearest neighbors are computed using full-space cosine similarity, not plotted distance.

| Example | Mean retrieval TV | Largest TV | Layer / head of maximum |
| --- | ---: | ---: | --- |
'''
    for r in summary:report+=f"| {r['animal']} / frame {r['frame']} / variant {r['variant']} | {r['mean_TV']:.4f} | {r['max_TV']:.4f} | {r['max_layer']} / {r['max_head']} |\n"
    report+='''
Largest mean contradictory–irrelevant retrieval differences (exploratory candidates, not causal rankings):

| Layer | Head | Mean TV | Effective slots | Value similarity among key neighbors |
| ---: | ---: | ---: | ---: | ---: |
'''
    for h in top:
        m=h['metrics'];report+=f"| {h['layer']} | {h['head']} | {m['mean_conflict_other_TV']:.4f} | {m['mean_effective_slots']:.1f} | {m['value_cos_among_key_neighbors']:.4f} |\n"
    report+='''
## How to read the atlas

1. Select an overview metric and click a layer/head cell.
2. Choose one of the six examples. The four prompts and complete-candidate D scores remain visible.
3. Compare the linked key and projected-value scatterplots. A marker's size reflects retrieval, while red/blue marks the direction of contradictory-minus-irrelevant retrieval change. Gray slots are the weight-only background.
4. Compare persistent retrieval across conditions and contextual source-position maps.
5. Select a profiled slot to inspect original-space neighbors, token alignments, and weighted contribution norms.

Interactive geometry overlays use the union of each example's four top-32 retrieval lists, the head's top-32 absolute contradictory–irrelevant changes across examples, and its profiled slots. We preserve exact weights for this shared displayed set in all four conditions. The full distributions are in `arrays.npz`. Profiles cover the union of the eight slots with highest retrieval across 24 prompts and eight with largest absolute contradictory–irrelevant change, separately per head. No clustering or semantic labeling is performed.

## Definitions and computation

For each head, persistent retrieval is `softmax(q @ P_k.T)`. The query includes the model's history aggregation, normalization and learned scale. Slot contribution is retrieval weight times its stored value, head output scale, and the appropriate slice of the output projection. A weighted contribution norm is the norm of that vector; it is not additive causal importance.

Key PCA uses centered unit-normalized keys. Projected-value PCA uses centered, unnormalized projected values. A thin QR factorization of the head output projection gives a 64-dimensional isometric representation of those vectors, preserving their norms, pairwise distances, and PCA variance. The displayed key/value maps have separate axes for every head. Neighbors use cosine similarity in their original spaces. Key norm is reported separately because normalized-key PCA does not show its effect on retrieval logits.

The contextual and persistent key half-life is `log(2)/(10*abs(leaky_key_beta))`, in token offsets. This describes exponential aggregation of input features before key normalization, not memory retention in training or a hard attention window. Contextual values mix source and next-position representations with the learned `coef`; the code does not constrain that coefficient to a convex mixture. Therefore token highlights do not isolate individual words. Text-region labels are display annotations only.

Vocabulary profiles rank cosine alignment of projected values with the model's output embedding rows, using all 50,257 valid GPT-2 token IDs (including special/byte tokens, excluding padded rows). Raw dot products accompany the rankings. Positive and negative alignment are geometric directions, not model probabilities. Output biases are not assigned to individual slots. Final layer normalization and downstream transformations are not included.

Candidate-direction panels use the first candidate token at the final prompt position. In contrast, all reported behavioral D values score complete candidate continuations, including multi-token candidates. The diagnostics are intentionally distinct; no full-candidate causal effect is inferred from a first-token direction.

Total variation is half the L1 distance between complete retrieval distributions, with range 0 to 1. Effective retrieved slots is exp(entropy), not a count of slots storing a fact. Contextual retrieval weights are reconstructed with the strictly earlier-position mask and the model's shifted value extractor. Both contextual and persistent reconstructed final-token module outputs are checked against the original forward outputs for all prompts and layers. A separate random normalized-weight test checks decomposition through head scales and the output projection.

## Limits and next experiment

The six examples were selected for success and share animals/templates; they are a discovery set. Routing differences can reflect lexical identity, syntax or presentation effects as well as entity-sensitive use. Similar values, nonlinear downstream computation, and residual-stream interactions prevent reading causal function directly from this atlas. Stable retrieval is not proof of stable knowledge storage. A slot may support multiple functions, and knowledge may be distributed.

Use the atlas to nominate modules or slots, then apply the bidirectional contradictory/irrelevant patches in [the mechanism plan](../../../lexicon_pilot/MECHANISM_PLAN.md). Validate discoveries with new controlled prompts and matched failures. No intervention has yet been executed.

## Files and provenance

- `atlas.html`: self-contained interactive view, no network dependencies.
- `atlas.json`: prompts, exact candidate scores/token IDs, contextual weights, sparse display retrievals, geometry, profiles, and metadata.
- `arrays.npz`: `persistent_retrieval` shape [24,12,12,2688]; `geometry` shape [144,2688,7] (key PC1/2, value PC1/2, key norm, value norm, projected norm); `context_i` shape [12,12,prompt_length] for prompt i. Heads are layer-major. Values use float32.
- `heads.csv`, `baselines.csv`, `example_summary.csv`, `validation.csv`: tabular summaries and checks.
- `metadata.json`: checkpoint path/stat identity, iteration, model configuration, source hashes, runtime and validation results. Checkpoint identity uses path, size and modification time, not a full-file cryptographic checksum.
- `build.py`, `summarize.py`, `template.html` in the experiment directory reproduce these outputs.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8');print(report[:3800])
if __name__=='__main__':main()
