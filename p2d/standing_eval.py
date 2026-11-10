# Cadence Flash standing evaluation + calibration harness. CPU-safe, deterministic, same call for every retrain:
#   python p2d/standing_eval.py freeze                      # one-time: build p2d/frozen + MANIFEST.json (refuses to overwrite)
#   python p2d/standing_eval.py run --ckpt outputs/<ckpt> --out p2d/runs/<name> [--baseline p2d/baseline_p2c.json] [--write-baseline p2d/baseline_p2c.json]
# Run with CUDA_VISIBLE_DEVICES=-1 for CPU. trust_remote_code is always False.
import os, sys, re, json, hashlib, argparse, time, math, collections
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); FZ=os.path.join(HERE,'frozen')
TYPES=['SPOTREP','SALUTE','9-line MEDEVAC','CCIR trigger','SITREP','FRAGO/order','RFI','routine admin','other']; PRECS=['Flash','Immediate','Priority','Routine']
ABR=re.compile(r"\b(TIC|WIA|KIA|IDF|CAS|QRF|EN|FRD|pax|vics?|vhcls?|w/o?|xroads|rpt|req|ASAP|NLT|cas|L\d|S:|A:|FRAGORD|WARNO|BP|PL|LD|NAI|PACE|REDCON|NMC|BUB|tmrw|vic)(?=\W|$)")
sha=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
rj=lambda f:[json.loads(x) for x in open(f,encoding='utf-8') if x.strip()]
nw=lambda t:len(t.split())
# ---- expectations for named probes (label policy: P2D_LABEL_POLICY.md) ----
PROBE_EXPECT={'PRIORITY SITREP: all quiet, PACE green':dict(type='SITREP',prec='Priority'),
 'ROUTINE TIC grid NV 4512 6789':dict(type='SPOTREP',prec='Flash',flash_p_min=0.5),
 'TIC':dict(prec='Flash',flash_p_min=0.5),'FLASH TIC grid NV 4512 6789':dict(prec='Flash'),
 'troops in contact':dict(prec_not='Routine'),'Troops in contact at grid 31U NB 1234 5678.':dict(prec_not='Routine')}
# ---- gates. absolute floors = the p2c release targets (phase-1 floor re-based to the frozen 300-per-source public subsample: p2c scores 0.905 there; the old 0.862 was on the full p1 test incl. typed_decisions); "targets" = goals for new slices (reported, gate only with --strict-targets) ----
ABS=[('military_test','type_acc','>=',0.98),('military_test','prec_acc','>=',0.95),('military_test','flash_recall','>=',0.95),
     ('military_test','ece_type','<=',0.08),('military_test','ece_prec','<=',0.08),('phase1_public','acc','>=',0.88),('phase1_public','ece','<=',0.08),
     ('typed_decisions','acc','>=',0.65),('src:p2c_short_test','prec_acc','>=',0.85),('src:p2c_short_test','flash_recall','>=',0.90),('no_prec_word','flash_recall','>=',0.95),('p2c_decoy','prec_acc','>=',0.90)]
TARGETS=[('frago','both_acc','>=',0.95),('fm50_p2d','both_acc','>=',0.90),('fm50_p2d','type_acc','>=',0.97),('typecall','type_acc','>=',0.95),('typecall','prec_acc','>=',0.90),
     ('decoy','prec_acc','>=',0.90),('stated_conflict_1','prec_acc','>=',0.85),('short','flash_recall','>=',0.95),('abbreviation','prec_acc','>=',0.90),('probes','pass_rate','>=',1.0)]
# regression vs baseline: accuracy-type metrics may not drop more than max(0.01, 2*SE_baseline); ECE may not rise more than 0.02.
def freeze():
    if os.path.exists(os.path.join(FZ,'MANIFEST.json')): sys.exit('frozen set exists; delete p2d/frozen deliberately to rebuild')
    os.makedirs(FZ,exist_ok=True); import pandas as pd
    P2=os.path.join(ROOT,'p2','data'); P2C=os.path.join(ROOT,'p2c','data'); P2D=os.path.join(ROOT,'p2d','data')
    mil=[]
    def add(x,src,extra):
        s=set(extra); t=x['text']
        if nw(t)<=12: s.add('short')
        if nw(t)<=8: s.add('short_le8')
        if ABR.search(t) or x.get('kind')=='shorthand': s.add('abbreviation')
        mil.append(dict(text=t,type=x['type'],precedence=x['precedence'],eval_source=src,source_doc=x.get('source_doc'),kind=x.get('kind'),slices=sorted(s)))
    for x in rj(os.path.join(P2,'test.jsonl')):
        e=['military_test']+(['fm50','fm50_p2orig'] if x['source_doc']=='FM 5-0' else [])+(['frago'] if x['type']=='FRAGO/order' and x['source_doc']!='FM 5-0' else []); add(x,'p2_test',e)
    for x in rj(os.path.join(P2C,'short_test.jsonl')):
        add(x,'p2c_short_test',['p2c_decoy','decoy'] if x['kind']=='decoy' else ['p2c_short'])
    for x in rj(os.path.join(P2C,'noword_test.jsonl')): add(x,'p2c_noword_test',['no_prec_word'])
    for x in rj(os.path.join(P2D,'p2d_test.jsonl')):
        e=['p2d_test']; k=x['kind']
        if k=='frago': e.append('frago')
        if k in('warnord','opord','re','brief'): e+=['fm50','fm50_p2d','fm50_'+k]
        if k=='typecall':
            e.append('typecall'); g=x.get('stated_gap')
            if g is not None and g>=2: e+=['decoy','stated_conflict_2plus']
            if g==1: e.append('stated_conflict_1')
        add(x,'p2d_test',e)
    p1=pd.read_parquet(os.path.join(ROOT,'mix','test.parquet')); p1['h']=p1.text.map(lambda t:hashlib.sha1(t.encode('utf-8')).hexdigest())
    p1=p1.sort_values(['source','h']); td=p1[p1.source=='typed_decisions']; pub=p1[p1.source!='typed_decisions'].groupby('source').head(300)
    p1rows=[dict(text=r.text,group=r.group,label=r.label,source=r.source,slices=['typed_decisions' if r.source=='typed_decisions' else 'phase1_public','phase1_all']) for r in pd.concat([pub,td]).itertuples()]
    probes=json.load(open(os.path.join(ROOT,'p2c','probes.json')))+json.load(open(os.path.join(ROOT,'p2c','probe_inputs.json')))
    probes+=['FLASH FLASH. Troops in contact grid NV 4512 6789, enemy platoon with RPGs, request immediate fire support. Over.']
    pr=[dict(text=t,expect=PROBE_EXPECT.get(t)) for t in dict.fromkeys(probes)]
    files={'military.jsonl':mil,'phase1.jsonl':p1rows,'probes.jsonl':pr}
    for f,L in files.items():
        with open(os.path.join(FZ,f),'w',encoding='utf-8',newline='\n') as fh:
            for x in L: fh.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
    sc=collections.Counter(s for x in mil+p1rows for s in x['slices'])
    man=dict(created=time.strftime('%Y-%m-%d %H:%M:%S %z'),files={f:dict(sha256=sha(os.path.join(FZ,f)),rows=len(L)) for f,L in files.items()},slice_counts=dict(sorted(sc.items())),
      sources={os.path.relpath(p,ROOT):sha(p) for p in [os.path.join(P2,'test.jsonl'),os.path.join(P2C,'short_test.jsonl'),os.path.join(P2C,'noword_test.jsonl'),os.path.join(P2D,'p2d_test.jsonl'),os.path.join(ROOT,'mix','test.parquet')]},
      note='FM 5-0 p2 test rows (fm50_p2orig) keep their original labels; their precedence is not recoverable from content (same timing phrase labeled Immediate and Priority). Use fm50_p2d for the policy-consistent FM 5-0 score.')
    json.dump(man,open(os.path.join(FZ,'MANIFEST.json'),'w'),indent=1); print(json.dumps(man,indent=1))
def verify():
    man=json.load(open(os.path.join(FZ,'MANIFEST.json')))
    for f,v in man['files'].items():
        h=sha(os.path.join(FZ,f))
        if h!=v['sha256']: sys.exit(f'MANIFEST MISMATCH {f}: {h} != {v["sha256"]}')
    return man
def ece(conf,ok,n=15):
    import numpy as np
    conf=np.asarray(conf,dtype=float); ok=np.asarray(ok,dtype=float)
    if not len(conf): return None
    b=np.minimum((conf*n).astype(int),n-1); return float(sum((b==k).mean()*abs(ok[b==k].mean()-conf[b==k].mean()) for k in range(n) if (b==k).any()))
def run(a):
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    man=verify(); torch.manual_seed(0); torch.use_deterministic_algorithms(True); torch.set_num_threads(a.threads)
    dev='cpu' if a.device=='cpu' else 'cuda'
    ck=os.path.abspath(a.ckpt); labels=json.load(open(os.path.join(ck,'labels.json'))); L2I={l:i for i,l in enumerate(labels)}
    T=json.load(open(os.path.join(ck,'temperature.json'))); Tt,Tp=T.get('mil_type',1.0),T.get('mil_prec',1.0)
    tok=AutoTokenizer.from_pretrained(ck,trust_remote_code=False,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(ck,torch_dtype=torch.float32,trust_remote_code=False,local_files_only=True,use_safetensors=True,attn_implementation='sdpa').to(dev).eval()
    t0=time.time()
    @torch.no_grad()
    def logits(texts):
        order=sorted(range(len(texts)),key=lambda i:(len(texts[i]),i)); out=[None]*len(texts)
        for s in range(0,len(order),a.bs):
            idx=order[s:s+a.bs]; e=tok([texts[i] for i in idx],truncation=True,max_length=256,padding=True,return_tensors='pt').to(dev)
            z=model(**e).logits.float().cpu().numpy()
            for j,i in enumerate(idx): out[i]=z[j]
        return np.stack(out)
    sm=lambda z:(lambda e:e/e.sum(-1,keepdims=True))(np.exp(z-z.max(-1,keepdims=True)))
    ti=[L2I['mil_type::'+t] for t in TYPES]; pi=[L2I['mil_prec::'+p] for p in PRECS]
    mil=rj(os.path.join(FZ,'military.jsonl')); p1=rj(os.path.join(FZ,'phase1.jsonl')); pr=rj(os.path.join(FZ,'probes.jsonl'))
    zm=logits([x['text'] for x in mil]); qt=sm(zm[:,ti]/Tt); qp=sm(zm[:,pi]/Tp); qtr=sm(zm[:,ti]); qpr=sm(zm[:,pi])
    yt=np.array([TYPES.index(x['type']) for x in mil]); yp=np.array([PRECS.index(x['precedence']) for x in mil]); pt=qt.argmax(1); pp=qp.argmax(1)
    for x in mil: x['slices']=x['slices']+['src:'+x['eval_source']]
    S={}
    for s in sorted(set(s for x in mil for s in x['slices'])):
        m=np.array([s in x['slices'] for x in mil]); okt=pt[m]==yt[m]; okp=pp[m]==yp[m]; fl=yp[m]==0
        S[s]=dict(n=int(m.sum()),type_acc=float(okt.mean()),prec_acc=float(okp.mean()),both_acc=float((okt&okp).mean()),flash_recall=float((pp[m][fl]==0).mean()) if fl.any() else None,
          flash_precision=float((yp[m][pp[m]==0]==0).mean()) if (pp[m]==0).any() else None,
          ece_type=ece(qt[m].max(1),okt),ece_prec=ece(qp[m].max(1),okp),ece_type_raw=ece(qtr[m].max(1),okt),ece_prec_raw=ece(qpr[m].max(1),okp),
          prec_recall={p:float((pp[m][yp[m]==i]==i).mean()) for i,p in enumerate(PRECS) if (yp[m]==i).any()},
          prec_confusion=[[int(((yp[m]==i)&(pp[m]==j)).sum()) for j in range(4)] for i in range(4)])
    # phase-1 heads: group-masked softmax over that group's labels, no temperature (as in train_p2c)
    zp1=logits([x['text'] for x in p1]); recs=[]
    gi={}
    for x,z in zip(p1,zp1):
        g=x['group']; idx=gi.setdefault(g,[i for i,l in enumerate(labels) if l.split('::')[0]==g]); q=sm(z[idx]); j=int(q.argmax())
        recs.append((x['source'],x['slices'],idx[j]==L2I[x['label']],float(q[j])))
    for s in ['phase1_public','typed_decisions','phase1_all']:
        r=[q for q in recs if s in q[1]]; S[s]=dict(n=len(r),acc=float(np.mean([q[2] for q in r])),ece=ece([q[3] for q in r],[q[2] for q in r]))
    S['phase1_by_source']={src:dict(n=len(r),acc=float(np.mean([q[2] for q in r])),ece=ece([q[3] for q in r],[q[2] for q in r])) for src in sorted(set(q[0] for q in recs)) for r in [[q for q in recs if q[0]==src]]}
    zpr=logits([x['text'] for x in pr]); P=[]
    for x,z in zip(pr,zpr):
        a_=sm(z[ti]/Tt); b_=sm(z[pi]/Tp); d=dict(text=x['text'],type=TYPES[int(a_.argmax())],type_p=round(float(a_.max()),4),prec=PRECS[int(b_.argmax())],prec_p=round(float(b_.max()),4),flash_p=round(float(b_[0]),4))
        e=x.get('expect')
        if e:
            ok=True
            if 'type' in e: ok&=d['type']==e['type']
            if 'prec' in e: ok&=d['prec']==e['prec']
            if 'prec_not' in e: ok&=d['prec']!=e['prec_not']
            if 'flash_p_min' in e: ok&=d['flash_p']>=e['flash_p_min']
            d['expect']=e; d['pass']=bool(ok)
        P.append(d)
    chk=[d for d in P if 'pass' in d]; S['probes']=dict(n=len(chk),pass_rate=sum(d['pass'] for d in chk)/len(chk))
    def g(sl,met): return S.get(sl,{}).get(met)
    def cmp(v,op,th): return v is not None and (v>=th if op=='>=' else v<=th)
    gates=[dict(slice=s,metric=m,op=o,threshold=t,value=g(s,m),passed=cmp(g(s,m),o,t)) for s,m,o,t in ABS]
    tg=[dict(slice=s,metric=m,op=o,threshold=t,value=g(s,m),passed=cmp(g(s,m),o,t)) for s,m,o,t in TARGETS]
    reg=[]
    if a.baseline:
        B=json.load(open(a.baseline))['slices']
        for s,v in B.items():
            if s in('probes','phase1_by_source') or s not in S: continue
            for m,bv in v.items():
                if not isinstance(bv,(int,float)) or m in('n',) or m.endswith('_raw') or bv is None: continue
                cv=S[s].get(m)
                if cv is None: continue
                if m.startswith('ece'): lim=bv+0.02; ok=cv<=lim
                else: tol=max(0.01,2*math.sqrt(max(bv*(1-bv),1e-9)/max(v['n'],1))); lim=bv-tol; ok=cv>=lim
                reg.append(dict(slice=s,metric=m,baseline=bv,value=cv,limit=round(lim,4),passed=bool(ok)))
    res=dict(checkpoint=ck,checkpoint_sha256={f:sha(os.path.join(ck,f)) for f in ['model.safetensors','labels.json','temperature.json']},temperature=T,device=dev,threads=a.threads,
      frozen_manifest=man['files'],elapsed_s=round(time.time()-t0,1),slices=S,probes=P,absolute_gates=gates,targets=tg,regression=reg,
      passed_absolute=all(x['passed'] for x in gates),passed_regression=all(x['passed'] for x in reg),targets_met=all(x['passed'] for x in tg),
      regression_rule='acc-type metrics: value >= baseline - max(0.01, 2*sqrt(p(1-p)/n)); ECE: value <= baseline + 0.02')
    res['overall_pass']=res['passed_absolute'] and res['passed_regression'] and (res['targets_met'] or not a.strict_targets)
    os.makedirs(a.out,exist_ok=True)
    with open(os.path.join(a.out,'predictions_military.jsonl'),'w',encoding='utf-8',newline='\n') as fh:
        for x,i,j,u,v in zip(mil,pt,pp,qt.max(1),qp.max(1)): fh.write(json.dumps(dict(text=x['text'],type=x['type'],precedence=x['precedence'],pred_type=TYPES[int(i)],pred_prec=PRECS[int(j)],type_p=round(float(u),5),prec_p=round(float(v),5),slices=x['slices']),ensure_ascii=False)+'\n')
    json.dump(res,open(os.path.join(a.out,'standing_eval.json'),'w'),indent=1)
    f3=lambda v:'-' if v is None else f'{v:.3f}'
    md=[f'# Cadence Flash standing eval: {os.path.basename(ck)}','',f'Overall: **{"PASS" if res["overall_pass"] else "FAIL"}** (absolute {res["passed_absolute"]}, regression {res["passed_regression"] if reg else "n/a"}, new-slice targets {res["targets_met"]})','',
        '| slice | n | type acc | prec acc | both | Flash recall | ECE type | ECE prec |','|---|---|---|---|---|---|---|---|']
    for s,v in S.items():
        if 'type_acc' in v: md.append(f'| {s} | {v["n"]} | {f3(v["type_acc"])} | {f3(v["prec_acc"])} | {f3(v["both_acc"])} | {f3(v["flash_recall"])} | {f3(v["ece_type"])} | {f3(v["ece_prec"])} |')
    md+=['','| phase-1 slice | n | acc | ECE |','|---|---|---|---|']+[f'| {s} | {S[s]["n"]} | {f3(S[s]["acc"])} | {f3(S[s]["ece"])} |' for s in ['phase1_public','typed_decisions','phase1_all']]
    md+=['','## Gates','| kind | slice | metric | rule | value | pass |','|---|---|---|---|---|---|']+[f'| {k} | {x["slice"]} | {x["metric"]} | {x["op"]} {x["threshold"]} | {f3(x["value"])} | {"yes" if x["passed"] else "NO"} |' for k,L in [('absolute',gates),('target',tg)] for x in L]
    if reg:
        bad=[x for x in reg if not x['passed']]; md+=['',f'## Regression vs baseline: {len(reg)-len(bad)}/{len(reg)} pass ({res["regression_rule"]})']+[f'- FAIL {x["slice"]}.{x["metric"]}: {x["value"]:.4f} vs baseline {x["baseline"]:.4f} (limit {x["limit"]})' for x in bad]
    md+=['','## Probes','| text | type (p) | prec (p) | Flash p | expected | pass |','|---|---|---|---|---|---|']+[f'| {d["text"][:70]} | {d["type"]} ({d["type_p"]}) | {d["prec"]} ({d["prec_p"]}) | {d["flash_p"]} | {json.dumps(d.get("expect")) if d.get("expect") else ""} | {"" if "pass" not in d else ("yes" if d["pass"] else "NO")} |' for d in P]
    open(os.path.join(a.out,'standing_eval.md'),'w',encoding='utf-8').write('\n'.join(md)+'\n')
    if a.write_baseline: json.dump(res,open(a.write_baseline,'w'),indent=1)
    print(json.dumps(dict(overall_pass=res['overall_pass'],absolute=res['passed_absolute'],regression=res['passed_regression'],targets=res['targets_met'],elapsed_s=res['elapsed_s'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True); sp.add_parser('freeze'); sp.add_parser('verify')
    r=sp.add_parser('run'); r.add_argument('--ckpt',required=True); r.add_argument('--out',required=True); r.add_argument('--baseline'); r.add_argument('--write-baseline')
    r.add_argument('--device',default='cpu',choices=['cpu','cuda']); r.add_argument('--threads',type=int,default=6); r.add_argument('--bs',type=int,default=32); r.add_argument('--strict-targets',action='store_true')
    a=ap.parse_args()
    if a.cmd=='freeze': freeze()
    elif a.cmd=='verify': print(json.dumps(verify()['files'],indent=1))
    else: run(a)
