import importlib.util, json, os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location('cadence_predict',os.path.join(root,'predict.py'))
if spec is None or spec.loader is None: raise ImportError('predict.py')
P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
tok,m,lm,T=P.load(sys.argv[1])
with open(sys.argv[2],encoding='utf-8') as f: rows=json.load(f)
for t in rows:
    r=P.predict(t,tok,m,lm,T); ty,pr=r['type'],r['precedence']
    print(json.dumps([t,ty['label'],ty['prob'],pr['label'],pr['prob'],pr['probs'].get('Flash')]))
