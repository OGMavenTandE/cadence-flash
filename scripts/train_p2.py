# Cadence Flash phase 2: continue ModernBERT-large from p1. Shared label space with group-masked softmax.
# Phase 1 groups/labels are kept unchanged (rows 0..N1-1), so p1 can be re-scored with its own heads.
# Two new heads are appended as new groups: mil_type::<9 types> and mil_prec::<Flash/Immediate/Priority/Routine>.
import os, sys, json, time, math, random, argparse
import numpy as np, pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument('--epochs',type=float,default=3); ap.add_argument('--bs',type=int,default=16); ap.add_argument('--lr',type=float,default=2e-5)
ap.add_argument('--max_len',type=int,default=256); ap.add_argument('--max_minutes',type=float,default=170); ap.add_argument('--smoke_seconds',type=float,default=0)
ap.add_argument('--live',default='p2/live'); ap.add_argument('--out',default='outputs/cadence-flash-p2'); ap.add_argument('--init',default='outputs/cadence-flash-p1')
ap.add_argument('--replay',type=int,default=2700); ap.add_argument('--calib_frac',type=float,default=0.05); ap.add_argument('--eval_cap',type=int,default=0)
ap.add_argument('--dry',action='store_true',help='build the mix and print counts only; no torch/GPU')
a=ap.parse_args()
R=os.path.dirname(os.path.abspath(__file__)); LIVE=os.path.join(R,a.live); OUT=os.path.join(R,a.out); INIT=os.path.join(R,a.init)
D=os.path.join(R,'p2','data'); STOP=os.path.join(R,'STOP'); os.makedirs(LIVE,exist_ok=True)
def log(m):
    s=time.strftime('%Y-%m-%d %H:%M:%S')+' '+m; print(s,flush=True); open(os.path.join(LIVE,'progress.log'),'a',encoding='utf-8').write(s+'\n')
random.seed(13); np.random.seed(13)
TYPES=['SPOTREP','SALUTE','9-line MEDEVAC','CCIR trigger','SITREP','FRAGO/order','RFI','routine admin','other']; PRECS=['Flash','Immediate','Priority','Routine']
p1_labels=json.load(open(os.path.join(INIT,'labels.json')))
labels=p1_labels+['mil_type::'+t for t in TYPES]+['mil_prec::'+p for p in PRECS]; L2I={l:i for i,l in enumerate(labels)}
gname=lambda l:l.split('::')[0]; groups=sorted(set(gname(l) for l in labels)); G2I={g:i for i,g in enumerate(groups)}
rj=lambda f:[json.loads(x) for x in open(os.path.join(D,f),encoding='utf-8') if x.strip()]
mtr=pd.DataFrame(rj('train.jsonl')); mte=pd.DataFrame(rj('test.jsonl'))
assert set(mtr.type)<=set(TYPES) and set(mtr.precedence)<=set(PRECS) and set(mte.type)<=set(TYPES) and set(mte.precedence)<=set(PRECS)
# held-out calibration slice from military TRAIN (stratified by type x precedence), never test
mtr['strat']=mtr.type+'|'+mtr.precedence
cal=mtr.groupby('strat').sample(frac=a.calib_frac,random_state=13); mfit=mtr.drop(cal.index)
p1tr=pd.read_parquet(os.path.join(R,'mix','train.parquet')); p1te=pd.read_parquet(os.path.join(R,'mix','test.parquet'))
# replay: stratified by source, proportional, from the p1 train mix
fr=a.replay/len(p1tr); rep=p1tr.groupby('source').sample(frac=fr,random_state=13)
rows=[dict(text=t,mt=L2I['mil_type::'+y],mp=L2I['mil_prec::'+p],l1=-1,g1=-1,src='military') for t,y,p in zip(mfit.text,mfit.type,mfit.precedence)]
rows+=[dict(text=t,mt=-1,mp=-1,l1=L2I[l],g1=G2I[g],src='replay:'+s) for t,l,g,s in zip(rep.text,rep.label,rep.group,rep.source)]
random.shuffle(rows)
info=dict(military_train=len(mfit),calib=len(cal),replay=len(rep),replay_by_source=rep.source.value_counts().to_dict(),total_train=len(rows),
 replay_share_of_total=round(len(rep)/len(rows),4),mil_test=len(mte),p1_test=len(p1te),labels=len(labels),groups=len(groups))
if a.eval_cap: mte=mte.groupby('type').head(a.eval_cap); p1te=p1te.groupby('source').head(a.eval_cap)
log('mix '+json.dumps(info))
if a.dry: json.dump(info,open(os.path.join(LIVE,'mix_info.json'),'w'),indent=1); sys.exit(0)
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
torch.manual_seed(13)
mask=torch.full((len(groups),len(labels)),float('-inf'))
for l,i in L2I.items(): mask[G2I[gname(l)],i]=0
GT,GP=G2I['mil_type'],G2I['mil_prec']
tok=AutoTokenizer.from_pretrained(INIT)
model=AutoModelForSequenceClassification.from_pretrained(INIT,torch_dtype=torch.float32,attn_implementation='sdpa')
assert model.config.num_labels==len(p1_labels)
old=model.classifier; new=torch.nn.Linear(old.in_features,len(labels))
with torch.no_grad():
    new.weight.normal_(0,0.02); new.bias.zero_(); new.weight[:len(p1_labels)]=old.weight; new.bias[:len(p1_labels)]=old.bias
model.classifier=new; model.num_labels=len(labels); model.config.num_labels=len(labels)
model.config.id2label={i:l for i,l in enumerate(labels)}; model.config.label2id=L2I
model=model.cuda(); mask=mask.cuda()
def enc(texts):
    e=tok(list(texts),truncation=True,max_length=a.max_len,padding=True,return_tensors='pt'); return {k:v.cuda() for k,v in e.items()}
total=int(math.ceil(len(rows)/a.bs)*a.epochs)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01); sch=get_linear_schedule_with_warmup(opt,int(0.06*total),total)
log(f'start train total_steps={total} bs={a.bs} max_len={a.max_len} epochs={a.epochs} lr={a.lr}')
torch.cuda.reset_peak_memory_stats(); t0=time.time(); step=0; stopped=None; ema=None; model.train()
while step<total and not stopped:
    random.shuffle(rows)
    for i in range(0,len(rows),a.bs):
        b=rows[i:i+a.bs]; e=enc([r['text'] for r in b])
        with torch.autocast('cuda',dtype=torch.bfloat16): lg=model(**e).logits.float()
        mil=torch.tensor([r['mt']>=0 for r in b]).cuda(); loss=0; parts=0
        if mil.any():
            x=lg[mil]; mt=torch.tensor([r['mt'] for r in b if r['mt']>=0]).cuda(); mp=torch.tensor([r['mp'] for r in b if r['mt']>=0]).cuda()
            loss=loss+F.cross_entropy(x+mask[GT],mt,reduction='sum')+F.cross_entropy(x+mask[GP],mp,reduction='sum'); parts+=2*int(mil.sum())
        if (~mil).any():
            x=lg[~mil]; y=torch.tensor([r['l1'] for r in b if r['mt']<0]).cuda(); g=torch.tensor([r['g1'] for r in b if r['mt']<0]).cuda()
            loss=loss+F.cross_entropy(x+mask[g],y,reduction='sum'); parts+=int((~mil).sum())
        loss=loss/parts
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step+=1
        ema=loss.item() if ema is None else 0.98*ema+0.02*loss.item(); el=time.time()-t0
        if step%25==0 or step==5:
            sps=step/el; m=dict(step=step,total=total,loss=round(ema,4),elapsed_s=round(el),sec_per_step=round(1/sps,3),projected_train_min=round(total/sps/60,1),
              peak_vram_gb=round(torch.cuda.max_memory_allocated()/2**30,2),reserved_gb=round(torch.cuda.max_memory_reserved()/2**30,2))
            log('progress '+json.dumps(m)); json.dump(m,open(os.path.join(LIVE,'train-status.json'),'w'))
        if os.path.exists(STOP): stopped='STOP file'; break
        if a.smoke_seconds and el>a.smoke_seconds: stopped='smoke time limit'; break
        if el>a.max_minutes*60: stopped='time budget'; break
        if step>=total: break
train_s=time.time()-t0; log(f'train end step={step}/{total} train_s={train_s:.0f} stopped={stopped}')
# ---------------- eval (offline) ----------------
model.eval()
@torch.no_grad()
def logits_of(texts):
    out=[]
    for i in range(0,len(texts),64):
        with torch.autocast('cuda',dtype=torch.bfloat16): out.append(model(**enc(texts[i:i+64])).logits.float())
    return torch.cat(out)
def ece(conf,correct,n=15):
    conf=np.asarray(conf); correct=np.asarray(correct,dtype=float); bins=np.minimum((conf*n).astype(int),n-1); e=0
    for k in range(n):
        m=bins==k
        if m.any(): e+=m.mean()*abs(correct[m].mean()-conf[m].mean())
    return float(e)
def head(lg,g,lab_prefix,vals):
    idx=[L2I[lab_prefix+v] for v in vals]; return lg[:,idx]  # restrict to this head's columns
def fit_T(z,y):
    lt=torch.zeros(1,device=z.device,requires_grad=True); o=torch.optim.LBFGS([lt],lr=0.1,max_iter=200)
    def c():
        o.zero_grad(); l=F.cross_entropy(z/lt.exp(),y); l.backward(); return l
    o.step(c); return float(lt.exp())
te0=time.time()
lc=logits_of(list(cal.text)); lm=logits_of(list(mte.text))
yct=torch.tensor([TYPES.index(t) for t in cal.type]).cuda(); ycp=torch.tensor([PRECS.index(p) for p in cal.precedence]).cuda()
yt=np.array([TYPES.index(t) for t in mte.type]); yp=np.array([PRECS.index(p) for p in mte.precedence])
zt_c,zp_c=head(lc,GT,'mil_type::',TYPES),head(lc,GP,'mil_prec::',PRECS); zt,zp=head(lm,GT,'mil_type::',TYPES),head(lm,GP,'mil_prec::',PRECS)
Tt,Tp=fit_T(zt_c,yct),fit_T(zp_c,ycp)
def summ(z,y,T,names):
    pr=F.softmax(z,-1); pc=F.softmax(z/T,-1); conf_r,pred=pr.max(-1); conf_c=pc.max(-1).values
    pred=pred.cpu().numpy(); ok=pred==y
    per={n:dict(n=int((y==i).sum()),acc=float(ok[y==i].mean()) if (y==i).any() else None,
               recall=float((pred[y==i]==i).mean()) if (y==i).any() else None,
               precision=float((y[pred==i]==i).mean()) if (pred==i).any() else None) for i,n in enumerate(names)}
    return dict(n=len(y),accuracy=float(ok.mean()),temperature=T,ece_raw=ece(conf_r.cpu().numpy(),ok),ece_temp_scaled=ece(conf_c.cpu().numpy(),ok),per_class=per),pred
mt_res,pt=summ(zt,yt,Tt,TYPES); mp_res,pp=summ(zp,yp,Tp,PRECS)
both=float(((pt==yt)&(pp==yp)).mean())
mt_res['per_class']['other']['note']='only n=83 in test; noisy'
per_src={s:float(((pt==yt)&(pp==yp))[(mte.source_doc==s).values].mean()) for s in mte.source_doc.unique()} if 'source_doc' in mte else {}
# p1 re-score with the original p1 group heads (masked over p1 labels only)
recs=[]
with torch.no_grad():
    for i in range(0,len(p1te),64):
        b=p1te.iloc[i:i+64]; 
        with torch.autocast('cuda',dtype=torch.bfloat16): lg=model(**enc(b.text)).logits.float()
        g=torch.tensor([G2I[x] for x in b.group]).cuda(); p=F.softmax(lg+mask[g],-1); c,pr=p.max(-1)
        y=[L2I[x] for x in b.label]
        for s,yy,q,cc in zip(b.source,y,pr.tolist(),c.tolist()): recs.append((s,yy==q,cc))
ev=pd.DataFrame(recs,columns=['source','correct','conf'])
p1res=dict(n=len(ev),accuracy=float(ev.correct.mean()),ece=ece(ev.conf,ev.correct),
 per_source={s:dict(n=len(d),acc=float(d.correct.mean()),ece=ece(d.conf,d.correct)) for s,d in ev.groupby('source')})
td=p1res['per_source'].get('typed_decisions',{}).get('acc')
flash=mp_res['per_class']['Flash']['recall']
tgt=dict(mil_type_acc=mt_res['accuracy']>=0.85,mil_prec_acc=mp_res['accuracy']>=0.85,flash_recall=flash>=0.95,
 ece_temp_scaled=max(mt_res['ece_temp_scaled'],mp_res['ece_temp_scaled'])<=0.08,p1_acc=p1res['accuracy']>=0.862,typed_decisions=(td or 0)>=0.66)
res=dict(label='AI Eval curated benchmark',smoke=bool(a.smoke_seconds),eval_capped=bool(a.eval_cap),train_s=round(train_s),steps=step,total_steps=total,stopped=stopped,
 eval_s=round(time.time()-te0),mix=info,calibration='temperature fit per head on held-out military train slice (n=%d), not test'%len(cal),
 military=dict(type=mt_res,precedence=mp_res,both_correct=both,flash_recall=flash,by_source_doc_both_correct=per_src),
 phase1_rescore=p1res,typed_decisions_acc=td,targets_met=tgt,
 peak_vram_gb=round(torch.cuda.max_memory_allocated()/2**30,2),reserved_gb=round(torch.cuda.max_memory_reserved()/2**30,2),args=vars(a))
if not a.smoke_seconds:
    os.makedirs(OUT,exist_ok=True); model.save_pretrained(OUT); tok.save_pretrained(OUT); json.dump(labels,open(os.path.join(OUT,'labels.json'),'w'))
    json.dump(dict(mil_type=Tt,mil_prec=Tp),open(os.path.join(OUT,'temperature.json'),'w')); res['checkpoint']=OUT
    json.dump(res,open(os.path.join(OUT,'metrics.json'),'w'),indent=1)
json.dump(res,open(os.path.join(LIVE,'metrics.json'),'w'),indent=1)
log('DONE '+json.dumps(dict(type=mt_res['accuracy'],prec=mp_res['accuracy'],flash=flash,p1=p1res['accuracy'],td=td,stopped=stopped)))
