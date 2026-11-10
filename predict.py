"""Cadence Flash inference. Usage: python predict.py "message text" [--model DIR]"""
import os, json, argparse, math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
HERE=os.path.dirname(os.path.abspath(__file__))
def load(d=HERE):
    if not os.path.isdir(d): raise FileNotFoundError(d)
    kw=dict(trust_remote_code=False,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(d,**kw)
    model=AutoModelForSequenceClassification.from_pretrained(d,use_safetensors=True,**kw).eval()
    with open(os.path.join(d,'label_map.json'),encoding='utf-8') as f: lm=json.load(f)
    with open(os.path.join(d,'temperature.json'),encoding='utf-8') as f: T=json.load(f)['temperatures']
    for h,t in T.items():
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<=0:
            raise ValueError('invalid temperature for '+str(h))
    return tok,model,lm,T
@torch.no_grad()
def predict(text,tok,model,lm,T,heads=('mil_type','mil_prec'),max_len=256):
    z=model(**tok([text],truncation=True,max_length=max_len,return_tensors='pt')).logits[0]
    out={}
    for h in heads:
        ent=lm['heads'][h]; idx=torch.tensor([e['index'] for e in ent])
        p=torch.softmax(z[idx]/T.get(h,1.0),-1); k=int(p.argmax())
        out[h]=dict(label=ent[k]['label'],prob=round(float(p[k]),4),probs={e['label']:round(float(q),4) for e,q in zip(ent,p)})
    return {'type':out.pop('mil_type',None),'precedence':out.pop('mil_prec',None),**out}
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('text'); ap.add_argument('--model',default=HERE); ap.add_argument('--head',action='append',default=[])
    a=ap.parse_args(); tok,model,lm,T=load(a.model)
    print(json.dumps(predict(a.text,tok,model,lm,T,heads=('mil_type','mil_prec',*a.head)),indent=1))
