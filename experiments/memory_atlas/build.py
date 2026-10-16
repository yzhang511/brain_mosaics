"""Example-guided memory atlas; descriptive retrievals, not causal attribution."""
import argparse, csv, gc, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nanoMosaics'))
from mosaic_model import MemoryMosaic, MemoryMosaicConfig
sys.path.insert(0, str(ROOT / 'experiments/lexicon_pilot'))
from run import Scorer
CONDITIONS = ['neutral', 'consistent', 'conflicting', 'other']
CASES = [('cat',5,0), ('squirrel',6,1), ('squirrel',7,1), ('lion',6,0), ('dog',5,1), ('elephant',6,0)]

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=True, separators=(',', ':'), allow_nan=False), encoding='utf-8')

def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def pca(x):
    z=x.double()-x.double().mean(0)
    vals, vecs=torch.linalg.eigh(z.T @ z)
    order=vals.argsort(descending=True); vals=vals[order].clamp_min(0); vecs=vecs[:,order[:2]]
    # Canonical signs for reproducible plotting.
    for j in range(2):
        if vecs[vecs[:,j].abs().argmax(),j]<0: vecs[:,j]*=-1
    return (z @ vecs).float(), (vals[:2]/vals.sum().clamp_min(1e-30)).tolist()

def unit(x): return F.normalize(x,dim=-1)
def rnd(x, n=5): return np.asarray(x,dtype=np.float64).round(n).tolist()

def token_info(enc, prompt, noun, other):
    ids=enc.encode_ordinary(prompt); toks=[enc.decode([i]) for i in ids]
    # Character intervals are used only for display annotations, never causal labels.
    spans=[]; start=0
    for token in toks:
        end=start+len(token); spans.append((start,end)); start=end
    intro_end=prompt.find('.')+1; statement_end=prompt.find('.',intro_end)+1
    query_start=statement_end
    subj_start=prompt.find('The ',intro_end)
    subj_end=(subj_start+4+len(noun if prompt[subj_start+4:].startswith(noun) else other)) if subj_start>=0 else -1
    roles=[]
    for a,b in spans:
        role='introduction' if a<intro_end else ('statement' if a<statement_end else 'query')
        if subj_start>=0 and a<subj_end and b>subj_start: role='preference subject'
        if a>=query_start and noun in prompt[max(query_start,a-len(noun)):b+len(noun)]: role='query / animal vicinity'
        roles.append(role)
    assert ''.join(toks)==prompt
    return ids,toks,roles

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--benchmark',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(1729); started=time.monotonic(); args.out.mkdir(parents=True,exist_ok=True)
    stat=args.checkpoint.stat()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=ck['model_args']; iteration=ck.get('iter_num')
    model=MemoryMosaic(MemoryMosaicConfig(**config)).eval()
    state={k.removeprefix('_orig_mod.'):v for k,v in ck['model'].items()}; model.load_state_dict(state,strict=True)
    del ck,state; gc.collect()
    enc=tiktoken.get_encoding('gpt2'); scorer=Scorer(model,enc,'cpu')
    assert config['pmem_count']==1, 'This atlas currently requires one persistent bank.'
    L,H,S=config['n_layer'],config['n_head'],config['pmem_size']; d=config['n_embd']//H
    source=ROOT/'experiments/lexicon_pilot/results/2026-10-15/scores.csv'
    rows=list(csv.DictReader(source.open())); selected=[]
    for noun,frame,variant in CASES:
        for cond in CONDITIONS:
            selected.append(next(r for r in rows if r['noun']==noun and int(r['frame_id'])==frame and int(r['variant'])==variant and r['condition']==cond and r['split']=='evaluation'))
    if args.benchmark: selected=selected[:1]
    retrieval=np.zeros((len(selected),L,H,S),np.float32); contexts=[]; prompts=[]; validation=[]; hooks=[]; current=[0]
    def hook(layer,kind):
        def collect(mod, inp, output):
            x=inp[0]; q=mod.k_featurizer(x,scale_pow=2 if kind=='persistent' else 1)
            if kind=='persistent':
                weights=(q[:,:,-1:] @ mod.P_k[0].transpose(-1,-2)).softmax(-1)
                y=(weights @ mod.P_v[0])*(mod.exp_scaling*mod.out_scale).exp()
                retrieval[current[0],layer]=weights[0,:,0].numpy()
            else:
                v=mod.v_featurizer(x)
                logits=q[:,:,-1:] @ q.transpose(-1,-2)
                logits[:,:,:,-1]=float('-inf')
                weights=logits.softmax(-1); y=weights @ v
                contexts[current[0]][layer]=weights[0,:,0].numpy().tolist()
            recon=mod.c_proj(y.transpose(1,2).contiguous().view(1,1,-1))
            err=(recon-output[:,-1:]).abs().max().item()
            validation.append(dict(prompt_id=current[0],layer=layer,kind=kind,max_abs_error=err,weights_sum_error=(weights.sum(-1)-1).abs().max().item()))
        return collect
    for l,block in enumerate(model.mosaic.h):
        hooks.extend([block.attn.register_forward_hook(hook(l,'contextual')),block.mlp.register_forward_hook(hook(l,'persistent'))])
    for i,r in enumerate(selected):
        current[0]=i; ids,toks,roles=token_info(enc,r['prompt'],r['noun'],r['other']); contexts.append([None]*L)
        for ti,t in enumerate(toks):
            if t.strip() in [r['default'],r['alternative']]: roles[ti] += ' / food'
        logits,_=model(torch.tensor([ids])); lp=logits[0,-1].log_softmax(-1)
        cand=[]
        for name in ['default','alternative']:
            whole,n=scorer.encode(r['prompt'],r[name])
            cand.append(dict(text=r[name],token_id=whole[n],token_ids=whole[n:],logp=lp[whole[n]].item()))
        D=cand[0]['logp']-cand[1]['logp']
        prompts.append(dict(id=i,case=i//4,animal=r['noun'],frame=int(r['frame_id']),variant=int(r['variant']),condition=r['condition'],prompt=r['prompt'],tokens=toks,token_ids=ids,roles=roles,candidates=cand,D=D,saved_D=float(r['D']),baseline_error=abs(D-float(r['D']))))
        print('prompt retrieval',i+1,len(selected),flush=True)
    for h in hooks: h.remove()
    score_validation=[]
    for p in prompts:
        for candidate in p['candidates']:
            example=(p['prompt'],candidate['text'])
            reference=scorer.reference(example)
            score_validation.append(abs(reference-scorer.batch([example])[0]))
            candidate['logp']=reference
        p['D']=p['candidates'][0]['logp']-p['candidates'][1]['logp']
        p['baseline_error']=abs(p['D']-p['saved_D'])
    print('baseline max error',max(p['baseline_error'] for p in prompts),'scoring validation',max(score_validation),flush=True)
    assert max(score_validation)<1e-4
    assert max(v['max_abs_error'] for v in validation)<1e-4, 'Memory reconstruction failed'
    assert max(p['baseline_error'] for p in prompts)<1e-4, 'Baseline reproduction failed'
    # Independent synthetic retrieval check includes all heads, output scaling and projection.
    synthetic=[]
    for l,block in enumerate(model.mosaic.h):
        m=block.mlp; a=torch.rand(H,S); a=a/a.sum(-1,keepdim=True)
        raw=(a[:,:,None]*m.P_v[0,0]).sum(1)*m.out_scale.flatten().mul(10).exp()[:,None]
        direct=m.c_proj(raw.flatten())
        parts=sum(raw[h] @ m.c_proj.weight[:,h*d:(h+1)*d].T for h in range(H))
        if m.c_proj.bias is not None: parts+=m.c_proj.bias
        synthetic.append((direct-parts).abs().max().item())
    assert max(synthetic)<1e-5
    heads=[]; metrics=[]; profiles=[]; geometry=[]
    vocab=model.lm_head.weight[:enc.n_vocab].detach(); vocab_norm=vocab.norm(dim=-1).clamp_min(1e-12)
    count=1 if args.benchmark else L*H
    for index in range(count):
        tick=time.monotonic(); l,h=divmod(index,H); block=model.mosaic.h[l]; m=block.mlp
        k=m.P_k[0,0,h]; v=m.P_v[0,0,h]; proj=m.c_proj.weight[:,h*d:(h+1)*d]
        scale=m.out_scale.flatten()[h].mul(10).exp(); vscaled=v*scale
        # Thin QR gives exact projected-value distances/PCA via a 64-D isometry.
        Q,R=torch.linalg.qr(proj,mode='reduced'); compact=vscaled @ R.T
        kn=unit(k); vn=unit(compact)
        kp,kev=pca(kn); vp,vev=pca(compact)
        geom=np.column_stack([kp.numpy(),vp.numpy(),k.norm(dim=-1).numpy(),v.norm(dim=-1).numpy(),compact.norm(dim=-1).numpy()]); geometry.append(geom)
        # Full-space nearest neighbors, never neighbors inferred from PCA.
        ks=kn @ kn.T; ks.fill_diagonal_(-float('inf')); knear,kid=ks.topk(5,dim=-1); del ks
        vs=vn @ vn.T; vs.fill_diagonal_(-float('inf')); vnear,vid=vs.topk(5,dim=-1); del vs
        cross=(vn[:,None,:]*vn[kid]).sum(-1).mean().item()
        a=torch.from_numpy(retrieval[:,l,h]); peak=a.max(0).values
        if len(selected)>1:
            delta=a[2::4]-a[3::4]; change=delta.abs().max(0).values
        else: change=peak*0
        # Fixed descriptive shortlist: top 8 peak retrieval + top 8 absolute conflict-other changes.
        chosen=sorted(set(peak.topk(8).indices.tolist()+change.topk(8).indices.tolist()))
        # Raw dot products and cosines are diagnostic alignments, not log-probability effects.
        output_basis=vocab @ Q
        alignment=compact[chosen] @ output_basis.T
        cos=alignment/compact[chosen].norm(dim=-1)[:,None].clamp_min(1e-12)/vocab_norm[None,:]
        headprofiles=[]
        for j,s in enumerate(chosen):
            pos=cos[j].topk(8); neg=(-cos[j]).topk(8)
            def words(res,sign): return [[int(t),enc.decode([int(t)]),float(sign*c),float(alignment[j,t])] for c,t in zip(res.values,res.indices)]
            profile=dict(slot=s,key_norm=float(k[s].norm()),value_norm=float(v[s].norm()),projected_norm=float(compact[s].norm()),key_neighbors=[[int(t),float(c)] for t,c in zip(kid[s],knear[s])],value_neighbors=[[int(t),float(c)] for t,c in zip(vid[s],vnear[s])],positive=words(pos,1),negative=words(neg,-1),retrieval=a[:,s].tolist(),contribution_norm=(a[:,s]*compact[s].norm()).tolist())
            headprofiles.append(profile); profiles.append(dict(layer=l,head=h,**profile))
        sparse=[]
        for pi in range(len(selected)):
            group=range((pi//4)*4, min((pi//4)*4+4, len(selected)))
            top=[s for pj in group for s in a[pj].topk(32).indices.tolist()]
            if len(selected)>1: top+=change.topk(32).indices.tolist()
            ids=sorted(set(top+chosen))
            sparse.append([[s,float(a[pi,s])] for s in ids])
        betas=[float(b.k_featurizer.leaky_avg.leaky_key_beta.flatten()[h].abs()*10) for b in [block.attn,block.mlp]]
        half=[math.log(2)/max(b,1e-30) for b in betas]
        entropy=-(a*a.clamp_min(1e-30).log()).sum(-1)
        tv=(.5*(a[2::4]-a[3::4]).abs().sum(-1)).tolist() if len(selected)>1 else []
        cosine=(F.cosine_similarity(a[2::4] @ compact,a[3::4] @ compact,dim=-1)).tolist() if len(selected)>1 else []
        candidate_alignment=[]; head_scores=[]
        for pi,p in enumerate(prompts):
            contrast=vocab[p['candidates'][0]['token_id']]-vocab[p['candidates'][1]['token_id']]
            direction=compact @ (Q.T @ contrast)
            head_scores.append(float(a[pi] @ direction))
            candidate_alignment.append([[s,float(a[pi,s]*direction[s])] for s in chosen])
        metric=dict(layer=l,head=h,context_half_life=half[0],persistent_half_life=half[1],context_value_mix=float(block.attn.v_featurizer.coef.flatten()[h]),key_norm_mean=float(k.norm(dim=-1).mean()),projected_value_norm_mean=float(compact.norm(dim=-1).mean()),key_nearest_cos_mean=float(knear[:,0].mean()),value_nearest_cos_mean=float(vnear[:,0].mean()),value_cos_among_key_neighbors=cross,key_pca_variance=sum(kev),value_pca_variance=sum(vev),mean_effective_slots=float(entropy.exp().mean()),mean_conflict_other_TV=float(np.mean(tv)) if tv else 0)
        metrics.append(metric)
        heads.append(dict(layer=l,head=h,geometry=rnd(geom,4),key_pca_variance=kev,value_pca_variance=vev,profiles=headprofiles,sparse=sparse,entropy=entropy.tolist(),tv=tv,output_cosine=cosine,head_alignment=head_scores,slot_alignment=candidate_alignment,metrics=metric))
        print('head',index+1,count,'seconds',round(time.monotonic()-tick,2),flush=True)
    meta=dict(checkpoint=str(args.checkpoint),checkpoint_size=stat.st_size,checkpoint_mtime_ns=stat.st_mtime_ns,iteration=iteration,model_args=config,torch_version=str(torch.__version__),threads=1,seed=1729,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),template_sha256=hashlib.sha256((Path(__file__).parent/'template.html').read_bytes()).hexdigest(),model_source_sha256=hashlib.sha256((ROOT/'nanoMosaics/mosaic_model.py').read_bytes()).hexdigest(),scores_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),module_reconstruction_max_error=max(v['max_abs_error'] for v in validation),baseline_max_error=max(p['baseline_error'] for p in prompts),synthetic_projection_max_error=max(synthetic),reference_scoring_max_error=max(score_validation),benchmark=args.benchmark,elapsed_seconds=time.monotonic()-started)
    assert args.checkpoint.stat().st_mtime_ns==stat.st_mtime_ns
    dump(args.out/'metadata.json',meta); table(args.out/'validation.csv',validation); table(args.out/'heads.csv',metrics)
    table(args.out/'baselines.csv',[{k:p[k] for k in ['id','case','animal','frame','variant','condition','D','saved_D','baseline_error','prompt']} for p in prompts])
    np.savez_compressed(args.out/'arrays.npz',persistent_retrieval=retrieval,geometry=np.stack(geometry),**{f'context_{i}':np.array(x,dtype=np.float32) for i,x in enumerate(contexts)})
    atlas=dict(prompts=prompts,heads=heads,contexts=contexts,metadata=meta,conditions=CONDITIONS)
    dump(args.out/'atlas.json',atlas)
    template=(Path(__file__).parent/'template.html').read_text(encoding='utf-8')
    (args.out/'atlas.html').write_text(template.replace('__ATLAS_DATA__',json.dumps(atlas,ensure_ascii=True,separators=(',',':'),allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    print('DONE',json.dumps(meta),flush=True)
if __name__=='__main__': main()
