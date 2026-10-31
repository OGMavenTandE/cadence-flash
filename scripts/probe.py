import sys,json; sys.path.insert(0,sys.argv[1]); import predict as P
tok,m,lm,T=P.load(sys.argv[1])
for t in json.load(open(sys.argv[2])):
    r=P.predict(t,tok,m,lm,T); ty,pr=r['type'],r['precedence']
    top2=sorted(pr['probs'].items(),key=lambda x:-x[1])[:2]
    print(json.dumps([t,ty['label'],ty['prob'],pr['label'],pr['prob'],pr['probs'].get('Flash')]))
