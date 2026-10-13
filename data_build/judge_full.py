import json,random,re,collections as C,concurrent.futures as cf,os,csv
from gen_lib import gem
random.seed(7)
S=json.load(open("full_stage1.json"));recs=S["recs"];spend=S["spend"]
P="""You are auditing labels for a fictional, unclassified military message-triage training set (doctrine: FM 6-99, FM 6-0, FM 5-0, ATP 4-02.2).
Message types: SPOTREP, SALUTE, 9-line MEDEVAC, CCIR trigger, SITREP, FRAGO/order (incl. WARNO), RFI, routine admin, other.
Precedence rubric (owner-approved): Flash = immediate threat to life or mission (troops in contact, imminent attack, any URGENT or URGENT-SURGICAL casualty, critical PIR met); Immediate = affects current operations within hours (enemy moving toward friendlies, priority casualties with possible enemy, near-term order change); Priority = important, not time-critical (~24h), INCLUDING administrative items with a near-term suspense that holds up higher HQ; Routine = administrative or no operational impact.
For each item: your own type and precedence, and agree=true if the PROPOSED labels are both defensible; reason if not; sensitive=true if text looks real rather than fictional.
Return JSON list of {"id","type","precedence","agree":bool,"reason","sensitive":bool}.
Items:
"""
by=C.defaultdict(list)
for r in recs: by[(r["split"],r["type"])].append(r)
samp=[]
for (sp,t),L in by.items(): samp+=random.sample(L,min(len(L),70 if sp=="train" else 45))
def job(b):
    out,u=gem("gemini-3.1-pro-preview",P+json.dumps([{"id":r["id"],"text":r["text"],"proposed_type":r["type"],"proposed_precedence":r["precedence"]} for r in b]))
    return json.loads(out),u
res={};tok=C.Counter()
with cf.ThreadPoolExecutor(6) as ex:
    futs=[ex.submit(job,samp[i:i+20]) for i in range(0,len(samp),20)]
    for f in futs:
        try: out,u=f.result()
        except Exception as e: continue
        for o in out: res[o["id"]]=o
        tok["in"]+=u.get("promptTokenCount",0);tok["out"]+=u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0)
jusd=tok["in"]/1e6*2+tok["out"]/1e6*12
judged=[r for r in samp if r["id"] in res]
bad={r["id"] for r in judged if not res[r["id"]].get("agree") or res[r["id"]].get("sensitive")}
st=dict(judged=len(judged),agree=len(judged)-len(bad),rate=round(1-len(bad)/max(1,len(judged)),4),
 by_split={sp:f"{sum(1 for r in judged if r['split']==sp and r['id'] not in bad)}/{sum(1 for r in judged if r['split']==sp)}" for sp in("train","test")},
 disagree_by_type=C.Counter(f"{r['type']}|{r['precedence']}->{res[r['id']].get('type')}|{res[r['id']].get('precedence')}" for r in judged if r["id"] in bad).most_common(15),
 sensitive=sum(1 for r in judged if res[r["id"]].get("sensitive")),judge_usd=round(jusd,2))
with open("judge_full_flags.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow(["id","split","type","precedence","judge_type","judge_precedence","reason","text"])
    for r in judged:
        if r["id"] in bad: j=res[r["id"]];w.writerow([r["id"],r["split"],r["type"],r["precedence"],j.get("type"),j.get("precedence"),j.get("reason"),r["text"]])
for r in recs: r["judge"]=("flagged" if r["id"] in bad else "pass") if r["id"] in res else "not_screened"
final=[r for r in recs if r["judge"]!="flagged"]
out="../final";os.makedirs(out,exist_ok=True)
for sp in("train","test"):
    with open(f"{out}/{sp}.jsonl","w") as f:
        for r in final:
            if r["split"]==sp: f.write(json.dumps(r)+"\n")
summ=dict(stage1=S["stats"],judge=st,counts={sp:dict(n=sum(1 for r in final if r["split"]==sp),type=C.Counter(r["type"] for r in final if r["split"]==sp),prec=C.Counter(r["precedence"] for r in final if r["split"]==sp),rephrased=sum(1 for r in final if r["split"]==sp and "rephrase" in r["method"]),source=C.Counter(r["source_doc"] for r in final if r["split"]==sp)) for sp in("train","test")},
 cost=dict(rephrase_usd=round(spend["usd"],2),judge_usd=round(jusd,2),total_usd=round(spend["usd"]+jusd,2),tokens=dict(spend["tok"],judge_in=tok["in"],judge_out=tok["out"])))
json.dump(summ,open(f"{out}/dataset_stats.json","w"),indent=1,default=str);print(json.dumps(summ,indent=1,default=str))
