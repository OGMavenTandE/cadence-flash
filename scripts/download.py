import os, json, sys
for v in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN']: os.environ.pop(v, None)
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
R=os.path.dirname(os.path.abspath(__file__)); os.environ['HF_HOME']=os.path.join(R,'hf-home')
from huggingface_hub import snapshot_download, HfApi
api=HfApi()
SETS={'banking77':'PolyAI/banking77','clinc150':'clinc/clinc_oos','boolq':'google/boolq','mnli':'nyu-mll/multi_nli','pi_deepset':'deepset/prompt-injections','pi_jackhhao':'jackhhao/jailbreak-classification'}
lic={}
for k,repo in SETS.items():
    try:
        info=api.dataset_info(repo, token=False)
        card=(info.card_data.to_dict() if info.card_data else {})
        lic[k]={'repo':repo,'license':card.get('license'),'sha':info.sha}
        for rev,sub in [(None,'raw'),('refs/convert/parquet','rawpq')]:
            try:
                p=snapshot_download(repo,repo_type='dataset',revision=rev,token=False,allow_patterns=['*.parquet','*.csv','*.jsonl','README.md'],local_dir=os.path.join(R,'data',k,sub))
                if any(f.endswith(('.parquet','.csv','.jsonl')) for _,_,fs in os.walk(p) for f in fs):
                    lic[k]['path']=p; break
            except Exception as e: print(k,rev,'ERR',e)
        print(k,lic[k],flush=True)
    except Exception as e:
        lic[k]={'repo':repo,'error':str(e)}; print(k,'ERR',e,flush=True)
mi=api.model_info('answerdotai/ModernBERT-large',token=False)
lic['model']={'repo':'answerdotai/ModernBERT-large','license':(mi.card_data.to_dict().get('license') if mi.card_data else None),'sha':mi.sha}
snapshot_download('answerdotai/ModernBERT-large',token=False,local_dir=os.path.join(R,'weights','ModernBERT-large'),allow_patterns=['*.json','*.safetensors','*.txt'])
json.dump(lic,open(os.path.join(R,'licenses-raw.json'),'w'),indent=1); print('DONE',flush=True)