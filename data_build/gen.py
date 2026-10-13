import random,json,os,re,time,urllib.request,concurrent.futures as cf
from templates import BUILDERS,SRC
random.seed(13)
KEY=os.environ["GEMINI_API_KEY"]
def gem(model,prompt,tries=3):
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}"
    body=json.dumps({"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"responseMimeType":"application/json","temperature":0.7 if 'flash' in model else 0}}).encode()
    for i in range(tries):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,body,{"Content-Type":"application/json"}),timeout=180))
            u=r.get("usageMetadata",{}); return r["candidates"][0]["content"]["parts"][0]["text"],u
        except Exception as e: time.sleep(5*(i+1)); err=e
    raise err
recs=[];seen=set();i=0
for t,b in BUILDERS.items():
    n=0
    while n<56:
        txt,p,src,page,fields=b()
        if txt in seen: continue
        seen.add(txt); n+=1; i+=1
        recs.append(dict(id=f"p2pilot-{i:04d}",type=t,precedence=p,text=txt,template_text=txt,method="template",source_doc=src,page=page,
            pub_date="; ".join(SRC[s.strip()][1] for s in src.split("/")),dist_statement="Approved for public release; distribution is unlimited",fields=fields))
random.shuffle(recs)
RP="""You rewrite fictional, unclassified military training messages so they read like realistic radio, chat, or CP log traffic.
Rules: keep every fact, the message type, and the urgency exactly; do not add new facts, units, casualties, or places; vary style (terse voice, typed chat, log entry); you may drop line numbers. Fictional data only.
Return a JSON list of objects {"id":..., "text":...}, same ids.
Messages:
"""
tok={"in":0,"out":0}
todo=[r for r in recs if random.random()<0.6]
def job(batch):
    out,u=gem("gemini-3.5-flash",RP+json.dumps([{"id":r["id"],"text":r["text"]} for r in batch]))
    return batch,out,u,RP
with cf.ThreadPoolExecutor(6) as ex:
    log=open("rephrase_log.jsonl","w")
    for batch,out,u,prompt in ex.map(job,[todo[k:k+15] for k in range(0,len(todo),15)]):
        tok["in"]+=u.get("promptTokenCount",0); tok["out"]+=u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0)
        try: m={o["id"]:o["text"] for o in json.loads(out)}
        except Exception: m={}
        for r in batch:
            if r["id"] in m and m[r["id"]].strip():
                r["text"]=m[r["id"]].strip(); r["method"]="template+rephrase(gemini-3.5-flash)"
        log.write(json.dumps({"model":"gemini-3.5-flash","prompt_ids":[r["id"] for r in batch],"prompt_header":prompt[:200],"usage":u})+"\n")
# dedupe on normalized text
norm=lambda s:re.sub(r"\W+"," ",s.lower()).strip(); d={};
for r in recs: d.setdefault(norm(r["text"]),r)
recs=list(d.values())
json.dump(recs,open("pilot.json","w"),indent=1)
with open("pilot.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
json.dump(tok,open("rephrase_tokens.json","w")); print(len(recs),tok)
