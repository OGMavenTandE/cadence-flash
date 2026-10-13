# Cadence Flash phase 1: ModernBERT-large decision classifier, group-masked softmax over a shared label space.
import os, sys, json, time, math, random, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
ap=argparse.ArgumentParser()
ap.add_argument('--epochs',type=float,default=2); ap.add_argument('--bs',type=int,default=16); ap.add_argument('--lr',type=float,default=3e-5)
ap.add_argument('--max_len',type=int,default=256); ap.add_argument('--max_minutes',type=float,default=225); ap.add_argument('--smoke_seconds',type=float,default=0)
ap.add_argument('--live',default='live'); ap.add_argument('--out',default='outputs/cadence-flash-p1'); ap.add_argument('--eval_cap',type=int,default=0)
a=ap.parse_args()
R=os.path.dirname(os.path.abspath(__file__)); LIVE=os.path.join(R,a.live); OUT=os.path.join(R,a.out); os.makedirs(LIVE,exist_ok=True); os.makedirs(OUT,exist_ok=True)
STOP=os.path.join(R,'STOP')
def log(m):
    s=time.strftime('%Y-%m-%d %H:%M:%S')+' '+m; print(s,flush=True); open(os.path.join(LIVE,'progress.log'),'a',encoding='utf-8').write(s+'\n')
torch.manual_seed(13); random.seed(13)
labels=json.load(open(os.path.join(R,'mix','labels.json'))); L2I={l:i for i,l in enumerate(labels)}
groups=sorted(set(l.split('::')[0] for l in labels)); G2I={g:i for i,g in enumerate(groups)}
mask=torch.full((len(groups),len(labels)),float('-inf'))
for l,i in L2I.items(): mask[G2I[l.split('::')[0]],i]=0
tr=pd.read_parquet(os.path.join(R,'mix','train.parquet')); te=pd.read_parquet(os.path.join(R,'mix','test.parquet'))
if a.eval_cap: te=te.groupby('source').head(a.eval_cap)
W=os.path.join(R,'weights','ModernBERT-large')
tok=AutoTokenizer.from_pretrained(W,trust_remote_code=False)
model=AutoModelForSequenceClassification.from_pretrained(W,num_labels=len(labels),trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
mask=mask.cuda()
def batches(df,bs,shuffle):
    idx=list(range(len(df)));
    if shuffle: random.shuffle(idx)
    for i in range(0,len(idx),bs):
        b=df.iloc[idx[i:i+bs]]
        enc=tok(list(b.text),truncation=True,max_length=a.max_len,padding=True,return_tensors='pt')
        yield {k:v.cuda() for k,v in enc.items()}, torch.tensor([L2I[x] for x in b.label]).cuda(), torch.tensor([G2I[x] for x in b.group]).cuda(), b
steps_per_epoch=math.ceil(len(tr)/a.bs); total=int(steps_per_epoch*a.epochs)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01); sch=get_linear_schedule_with_warmup(opt,int(0.06*total),total)
log(f'start train n={len(tr)} test={len(te)} labels={len(labels)} groups={len(groups)} total_steps={total} bs={a.bs} max_len={a.max_len} epochs={a.epochs}')
torch.cuda.reset_peak_memory_stats(); t0=time.time(); step=0; stopped=None; ema=None
model.train()
while step<total and not stopped:
    for enc,y,g,_ in batches(tr,a.bs,True):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**enc).logits
        loss=F.cross_entropy(logits.float()+mask[g],y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step+=1
        ema=loss.item() if ema is None else 0.98*ema+0.02*loss.item()
        el=time.time()-t0
        if step%25==0 or step==5:
            sps=step/el; proj=total/sps
            m=dict(step=step,total=total,loss=round(ema,4),elapsed_s=round(el),sec_per_step=round(1/sps,3),projected_train_min=round(proj/60,1),peak_vram_gb=round(torch.cuda.max_memory_allocated()/2**30,2),reserved_gb=round(torch.cuda.max_memory_reserved()/2**30,2))
            log('progress '+json.dumps(m)); json.dump(m,open(os.path.join(LIVE,'train-status.json'),'w'))
        if os.path.exists(STOP): stopped='STOP file'; break
        if a.smoke_seconds and el>a.smoke_seconds: stopped='smoke time limit'; break
        if el>a.max_minutes*60: stopped='time budget'; break
        if step>=total: break
train_s=time.time()-t0; log(f'train end step={step}/{total} train_s={train_s:.0f} stopped={stopped}')
# ---- eval, offline ----
model.eval(); recs=[]; te0=time.time()
with torch.no_grad():
    for enc,y,g,b in batches(te,64,False):
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(**enc).logits
        p=F.softmax(logits.float()+mask[g],-1); conf,pred=p.max(-1)
        for s,yy,pp,cc in zip(b.source,y.tolist(),pred.tolist(),conf.tolist()): recs.append((s,yy==pp,cc))
ev=pd.DataFrame(recs,columns=['source','correct','conf'])
def ece(d,n=15):
    bins=np.minimum((d.conf.values*n).astype(int),n-1); e=0
    for k in range(n):
        m=bins==k
        if m.any(): e+=m.mean()*abs(d.correct.values[m].mean()-d.conf.values[m].mean())
    return float(e)
res=dict(train_s=round(train_s),steps=step,total_steps=total,stopped=stopped,eval_s=round(time.time()-te0),n_test=len(ev),
 accuracy=float(ev.correct.mean()),ece=ece(ev),macro_source_acc=float(ev.groupby('source').correct.mean().mean()),
 per_source={s:dict(n=len(d),acc=float(d.correct.mean()),ece=ece(d)) for s,d in ev.groupby('source')},
 peak_vram_gb=round(torch.cuda.max_memory_allocated()/2**30,2),reserved_gb=round(torch.cuda.max_memory_reserved()/2**30,2),args=vars(a))
if not a.smoke_seconds:
    model.save_pretrained(OUT); tok.save_pretrained(OUT); json.dump(labels,open(os.path.join(OUT,'labels.json'),'w')); res['checkpoint']=OUT
json.dump(res,open(os.path.join(LIVE,'metrics.json'),'w'),indent=1); log('DONE '+json.dumps({k:res[k] for k in ['accuracy','ece','train_s','stopped']}))