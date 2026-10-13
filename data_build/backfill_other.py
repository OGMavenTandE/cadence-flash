import random,json,re,collections as C,concurrent.futures as cf
from gen_lib import gem
from templates import grid,unit,dtg,pick
random.seed(1013)
def other2():
    p=pick("Flash","Immediate","Priority","Routine")
    if p=="Flash": t=pick(f"Flash flood warning: river rising fast, LSA at {grid()} must evacuate to high ground now.",
        f"Lightning strike within {pick('1','2')} nm of the fuel point at {grid()}; cease all refuel operations immediately.",
        f"Tornado sighted {pick('3','5','4')} km west of the {pick('LSA','motor pool','BSA')}, moving east; all personnel to shelters now.")
    elif p=="Immediate": t=pick(f"Weather advisory: {pick('dust storm','high winds 45 kt','dense fog')} expected within {pick('1','2','3')} hours at {unit()}; plan for near-zero visibility.",
        f"Host-nation power grid outage affecting the {pick('main CP','LSA','aid station')}; generators running, fuel for {pick('6','8','10')} hours.",
        f"Local authorities closed the {pick('north','east','river')} bridge to vehicles over {pick('10','20','30')} tons, effective now.")
    elif p=="Priority": t=pick(f"Heat category {pick('4','5')} forecast for tomorrow; adjust work-rest cycles for {unit()}.",
        f"Host-nation holiday tomorrow: expect road closures and heavy civilian traffic near {pick('the market','the stadium','downtown')}.",
        f"Manufacturer safety notice: {pick('generator','water heater','space heater')} model in use at {unit()} has a recall; inspect within 24 hours.")
    else: t=pick(f"{pick('Dining facility','Gym','Laundry point','Chapel')} hours change starting {pick('Monday','next week','the 1st')}.",
        f"Weather outlook: {pick('clear skies','light rain','mild temperatures')} for the next {pick('48','72','96')} hours.",
        f"MWR {pick('movie night','5K run','card tournament')} scheduled {pick('Friday','Saturday','Sunday')} at the {pick('LSA','gym','dining facility')}.")
    return t,p
P="""Rewrite each fictional, unclassified message as a soldier might pass it on radio, chat, or CP log. Keep every fact and the urgency; add nothing; no precedence words. Return JSON list of {"id","text"}.
Messages:
"""
old={json.loads(l)["template_text"] for sp in("train","test") for l in open(f"../final/{sp}.jsonl")}
cost=0.0;out={}
for sp,n,frac in (("train",1436,0.5),("test",230,1.0)):
    seen=set();R=[]
    while len(R)<n:
        t,p=other2()
        if t in seen or t in old: continue
        seen.add(t);R.append(dict(id=f"cf2-{sp}-o{len(R):04d}",split=sp,type="other",precedence=p,text=t,template_text=t,method="template",source_doc="FM 6-99",page="general (non-format traffic)",pub_date="2021-05-17",dist_statement="Approved for public release; distribution is unlimited",fields=["FREE TEXT"],judge="not_screened"))
    out[sp]=R
    if sp=="train": old|=seen
print({k:len(v) for k,v in out.items()})
todo=[r for r in out["train"] if random.random()<0.5]+out["test"]
def job(b):
    o,u=gem("gemini-3.5-flash",P+json.dumps([{"id":r["id"],"text":r["text"]} for r in b]));return b,o,u
bad=re.compile(r"\b(FLASH|IMMEDIATE|PRIORITY|ROUTINE|SPOTREP|SALUTE|MEDEVAC|SITREP|FRAGO|RFI|CCIR|PIR)\b",re.I)
with cf.ThreadPoolExecutor(8) as ex:
    for b,o,u in ex.map(job,[todo[i:i+25] for i in range(0,len(todo),25)]):
        cost+=u.get("promptTokenCount",0)/1e6*1.5+(u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0))/1e6*9
        try: m={x["id"]:x["text"] for x in json.loads(o)}
        except Exception: m={}
        for r in b:
            t=m.get(r["id"],"").strip()
            if t and not bad.search(t) and all(x in t for x in re.findall(r"\b\d{4}\b",r["template_text"])): r["text"]=t;r["method"]="template+rephrase(gemini-3.5-flash)"
out["test"]=[r for r in out["test"] if "rephrase" in r["method"]]
# judge screen ~150
J="""Audit labels for a fictional military message-triage set. Types: SPOTREP, SALUTE, 9-line MEDEVAC, CCIR trigger, SITREP, FRAGO/order, RFI, routine admin, other (non-tactical safety, weather, host-nation, quality-of-life notices). Precedence: Flash = immediate threat to life/mission; Immediate = affects ops within hours; Priority = ~24h; Routine = no ops impact. agree=true if both proposed labels are defensible. Return JSON list {"id","type","precedence","agree":bool,"reason"}.
Items:
"""
samp=random.sample(out["train"],90)+random.sample(out["test"],60);res={}
def jj(b):
    o,u=gem("gemini-3.1-pro-preview",J+json.dumps([{"id":r["id"],"text":r["text"],"proposed_type":"other","proposed_precedence":r["precedence"]} for r in b]));return o,u
with cf.ThreadPoolExecutor(6) as ex:
    for o,u in ex.map(jj,[samp[i:i+15] for i in range(0,len(samp),15)]):
        cost+=u.get("promptTokenCount",0)/1e6*2+(u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0))/1e6*12
        for x in json.loads(o): res[x["id"]]=x
flag={i for i,x in res.items() if not x.get("agree")}
print("judge",len(res)-len(flag),"/",len(res),C.Counter(f"{x.get('type')}|{x.get('precedence')}" for i,x in res.items() if i in flag))
for sp in out:
    for r in out[sp]:
        if r["id"] in res: r["judge"]="flagged" if r["id"] in flag else "pass"
    keep=[r for r in out[sp] if r["judge"]!="flagged"]
    with open(f"../final/{sp}.jsonl","a") as f: f.writelines(json.dumps(r)+"\n" for r in keep)
    print(sp,len(keep))
json.dump(dict(cost=round(cost,2),judged=len(res),agree=len(res)-len(flag)),open("backfill_other_stats.json","w"));print("cost",round(cost,2))
