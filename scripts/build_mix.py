# Builds the phase-1 public decision mix. Public data only; no emails / private data.
import os, json, random
import pandas as pd
R=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(R,'data'); rnd=random.Random(13)
MNLI_TRAIN=int(os.environ.get('CF_MNLI_TRAIN','40000')); MNLI_TEST=3000
rows={'train':[],'test':[]}
def add(split,src,group,text,label): rows[split].append({'source':src,'group':group,'text':text,'label':f'{group}::{label}'})
# typed-decisions (apache-2.0)
TD=os.environ.get('TYPED_DECISIONS_DIR',os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','typed-decisions'))
for split,f in [('train','train-00000-of-00001.parquet'),('test','test-00000-of-00001.parquet')]:
    for _,r in pd.read_parquet(os.path.join(TD,f)).iterrows():
        qs=json.loads(r['questions']); gold=json.loads(r['gold'])
        for q,spec in qs.items():
            if q not in gold: continue
            crit=spec.get('criteria',{'true':'The statement is true.','false':'The statement is false.'})
            opts='; '.join(f'{k}: {v}' for k,v in crit.items()) if isinstance(crit,dict) else '; '.join(f'{i}: {v}' for i,v in enumerate(crit))
            add(split,'typed_decisions',f"typed/{r['workflow']}/{q}",f"[decision] {spec['instructions']} Options: {opts} State: {r['state']}",gold[q]['label'])
# BANKING77 (cc-by-4.0)
for split in ['train','test']:
    for _,r in pd.read_csv(os.path.join(D,'banking77','raw',f'{split}.csv')).iterrows(): add(split,'banking77','banking77',f"[intent:banking] {r['text']}",r['category'])
# CLINC150 plus (cc-by-3.0)
for split in ['train','test']:
    for _,r in pd.read_parquet(os.path.join(D,'clinc150','raw','plus',f'{split}-00000-of-00001.parquet')).iterrows(): add(split,'clinc150','clinc150',f"[intent:assistant] {r['text']}",int(r['intent']))
# BoolQ (cc-by-sa-3.0); validation used as test
for split,f in [('train','train'),('test','validation')]:
    for _,r in pd.read_parquet(os.path.join(D,'boolq','raw','data',f'{f}-00000-of-00001.parquet')).iterrows(): add(split,'boolq','boolq',f"[yesno] Question: {r['question']} Passage: {r['passage']}",str(bool(r['answer'])).lower())
# MNLI (OANC / cc-by-3.0 / cc-by-sa-3.0); subsampled; validation_matched as test
for split,f,n in [('train','train',MNLI_TRAIN),('test','validation_matched',MNLI_TEST)]:
    df=pd.read_parquet(os.path.join(D,'mnli','raw','data',f'{f}-00000-of-00001.parquet')); df=df[df.label>=0].sample(n=min(n,len(df)),random_state=13)
    for _,r in df.iterrows(): add(split,'mnli','mnli',f"[entailment] Premise: {r['premise']} Hypothesis: {r['hypothesis']}",['entailment','neutral','contradiction'][int(r['label'])])
# deepset/prompt-injections (apache-2.0)
import glob
for split in ['train','test']:
    for _,r in pd.read_parquet(glob.glob(os.path.join(D,'pi_deepset','raw','data',f'{split}-*.parquet'))[0]).iterrows(): add(split,'pi_deepset','prompt_injection',f"[injection] {r['text']}",'injection' if int(r['label'])==1 else 'benign')
# jackhhao/jailbreak-classification (apache-2.0)
for split in ['train','test']:
    for _,r in pd.read_csv(os.path.join(D,'pi_jackhhao','raw','default',f'jailbreak_dataset_{split}.csv')).iterrows(): add(split,'pi_jackhhao','prompt_injection',f"[injection] {r['prompt']}",'injection' if r['type']=='jailbreak' else 'benign')
os.makedirs(os.path.join(R,'mix'),exist_ok=True)
for s in rows:
    rnd.shuffle(rows[s]); pd.DataFrame(rows[s]).to_parquet(os.path.join(R,'mix',f'{s}.parquet'))
labels=sorted(set(x['label'] for s in rows for x in rows[s]))
json.dump(labels,open(os.path.join(R,'mix','labels.json'),'w'))
counts={s:pd.DataFrame(rows[s]).source.value_counts().to_dict() for s in rows}
json.dump({'counts':counts,'n_labels':len(labels),'n_groups':len(set(l.split('::')[0] for l in labels))},open(os.path.join(R,'mix','summary.json'),'w'),indent=1)
print(json.dumps(counts),len(labels))