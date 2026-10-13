import json,random,collections as C,concurrent.futures as cf
from gen_lib import gem
random.seed(200)
recs=json.load(open("pilot.json"))
# stratified 200: ~22 per type
by=C.defaultdict(list)
for r in recs: by[r["type"]].append(r)
samp=[]
for t,l in by.items(): samp+=random.sample(l,22)
samp+=random.sample([r for r in recs if r not in samp],200-len(samp))
P="""You are auditing labels for a fictional, unclassified military message-triage training set (doctrine: FM 6-99, FM 6-0, ATP 4-02.2).
Message types: SPOTREP, SALUTE, 9-line MEDEVAC, CCIR trigger, SITREP, FRAGO/order, RFI, routine admin, other.
Precedence rubric: Flash = immediate threat to life or mission (troops in contact, imminent attack, urgent-surgical casualties under fire, critical PIR met); Immediate = affects current operations within hours (enemy moving toward friendlies, urgent casualties, near-term order change); Priority = important, not time-critical (within ~24h); Routine = administrative or no operational impact.
For each item give your own type and precedence, then say whether the PROPOSED labels are acceptable (agree=true if both are defensible), plus a short reason if not, and flag any text that looks real/sensitive rather than fictional.
Return JSON list of {"id","type","precedence","agree":bool,"reason","sensitive":bool}.
Items:
"""
def job(b):
    out,u=gem("gemini-3.1-pro-preview",P+json.dumps([{"id":r["id"],"text":r["text"],"proposed_type":r["type"],"proposed_precedence":r["precedence"]} for r in b]))
    return json.loads(out),u
res={};tok=[0,0]
with cf.ThreadPoolExecutor(5) as ex:
    for out,u in ex.map(job,[samp[k:k+20] for k in range(0,200,20)]):
        for o in out: res[o["id"]]=o
        tok[0]+=u.get("promptTokenCount",0); tok[1]+=u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0)
rows=[]
for r in samp:
    j=res.get(r["id"],{})
    rows.append(dict(r,judge_type=j.get("type"),judge_precedence=j.get("precedence"),judge_agree=j.get("agree"),judge_reason=j.get("reason",""),judge_sensitive=j.get("sensitive")))
json.dump(rows,open("judge_200.json","w"),indent=1)
n=len(rows); ok=lambda k:sum(1 for x in rows if k(x))
st=dict(n=n,missing=ok(lambda x:x["judge_agree"] is None),overall_agree=ok(lambda x:x["judge_agree"]),
 type_match=ok(lambda x:x["judge_type"]==x["type"]),prec_match=ok(lambda x:x["judge_precedence"]==x["precedence"]),
 both_exact=ok(lambda x:x["judge_type"]==x["type"] and x["judge_precedence"]==x["precedence"]),sensitive_flags=ok(lambda x:x["judge_sensitive"]),tokens=tok,
 disagree_by_type=C.Counter(x["type"] for x in rows if not x["judge_agree"]))
json.dump(st,open("judge_stats.json","w"),indent=1,default=str);print(st)
