import random,json,re,sys,collections as C,concurrent.futures as cf,threading
from templates import BUILDERS,SRC,warno
from gen_lib import gem
random.seed(2026)
PRICE={"gemini-3.5-flash":(1.5,9.0),"gemini-3.1-pro-preview":(2.0,12.0)}
spend={"usd":0.0,"tok":C.Counter()};lock=threading.Lock()
def acct(m,u):
    i=u.get("promptTokenCount",0);o=u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0)
    with lock:
        spend["tok"][m+"_in"]+=i;spend["tok"][m+"_out"]+=o;spend["usd"]+=i/1e6*PRICE[m][0]+o/1e6*PRICE[m][1]
TYPES=list(BUILDERS)
def mk(t,split,k):
    if split=="test" and t=="FRAGO/order" and random.random()<0.5: txt,p,src,page,f=warno()
    else: txt,p,src,page,f=BUILDERS[t]()
    return dict(id=f"cf2-{split}-{k:05d}",split=split,type=t,precedence=p,text=txt,template_text=txt,method="template",source_doc=src,page=page,
        pub_date="; ".join(SRC[s.strip()][1] for s in src.split("/")),dist_statement="Approved for public release; distribution is unlimited",fields=f)
recs=[];seen=set();k=0
for split,n in (("train",1500),("test",230)):
    for t in TYPES:
        c=0
        while c<n:
            r=mk(t,split,k)
            if r["text"] in seen: continue
            seen.add(r["text"]);recs.append(r);c+=1;k+=1
PW=["FLASH","IMMEDIATE","PRIORITY","ROUTINE"]
TW={"SPOTREP":r"\bSPOT ?REP","SALUTE":r"\bSALUTE\b","9-line MEDEVAC":r"MEDEVAC|9-LINE","SITREP":r"\bSITREP","FRAGO/order":r"FRAGO|WARNO","RFI":r"\bRFI\b|REQUEST FOR INFO","CCIR trigger":r"\bCCIR|\bPIR\b|\bFFIR\b"}
def drift(r,new):
    a,b=r["template_text"].upper(),new.upper()
    for w in PW:
        if re.search(r"\b"+w+r"\b",b) and not re.search(r"\b"+w+r"\b",a): return f"added precedence word {w}"
    for t,pat in TW.items():
        if re.search(pat,b) and not re.search(pat,a): return f"added type cue {t}"
    for nums in re.findall(r"\b\d{4}\b",a):
        if nums not in b: return "dropped grid/number "+nums
    return None
RP={"train":"""You rewrite fictional, unclassified military training messages so they read like realistic radio, chat, or CP log traffic.
Rules: keep every fact, number, grid, the message type, and the urgency exactly; do not add facts, units, casualties, places, or precedence words; vary style (terse voice, typed chat, log entry); you may drop line numbers. Fictional data only.
Return a JSON list of {"id","text"} with the same ids.
Messages:
""","test":"""Rewrite each fictional, unclassified military message as a different operator would send it: informal, abbreviated, out-of-order fields allowed, but keep every fact, number, grid, the message type and urgency exactly. Do not add facts, units, casualties, places or precedence words.
Return a JSON list of {"id","text"} with the same ids.
Messages:
"""}
todo=[r for r in recs if r["split"]=="test" or random.random()<0.5]
CAP_REPH=30.0;dlog=open("full_rephrase_log.jsonl","w");stats=C.Counter()
def job(b):
    if spend["usd"]>CAP_REPH: return b,None
    sp=b[0]["split"]
    try: out,u=gem("gemini-3.5-flash",RP[sp]+json.dumps([{"id":r["id"],"text":r["text"]} for r in b]))
    except Exception as e: return b,None
    acct("gemini-3.5-flash",u);return b,out
batches=[];
for sp in ("train","test"):
    L=[r for r in todo if r["split"]==sp];batches+=[L[i:i+25] for i in range(0,len(L),25)]
with cf.ThreadPoolExecutor(8) as ex:
    for n,(b,out) in enumerate(ex.map(job,batches)):
        try: m={o["id"]:o["text"] for o in json.loads(out)} if out else {}
        except Exception: m={}
        for r in b:
            t=m.get(r["id"],"").strip()
            if not t: stats["rephrase_missing"]+=1;continue
            d=drift(r,t)
            if d: stats["drift_rejected"]+=1;dlog.write(json.dumps({"id":r["id"],"drift":d})+"\n");continue
            r["text"]=t;r["method"]="template+rephrase(gemini-3.5-flash)";stats["rephrased_ok"]+=1
        if n%40==0: print(n,len(batches),round(spend["usd"],2),dict(stats),flush=True)
# test must be rephrased; drop unrephrased test items
recs=[r for r in recs if r["split"]=="train" or "rephrase" in r["method"]]
norm=lambda s:re.sub(r"\W+"," ",s.lower()).strip()
mask=lambda s:re.sub(r"\d+","#",norm(s))
d={}
for r in recs:
    key=norm(r["text"])
    if key in d: stats["dedupe_dropped"]+=1;continue
    d[key]=r
recs=list(d.values())
trm={mask(r["text"]) for r in recs if r["split"]=="train"}|{mask(r["template_text"]) for r in recs if r["split"]=="train"}
keep=[]
for r in recs:
    if r["split"]=="test" and (mask(r["text"]) in trm): stats["test_leak_dropped"]+=1;continue
    keep.append(r)
recs=keep
json.dump({"recs":recs,"spend":spend,"stats":stats},open("full_stage1.json","w"))
print("DONE",len(recs),round(spend["usd"],2),dict(stats))
