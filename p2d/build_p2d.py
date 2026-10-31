# Cadence Flash p2d data build (rule-based templates only; no LLM rephrasing, no private or email data).
# Adds: FRAGOs, FM 5-0 planning messages (WARNORDs, OPORD paragraph excerpts, running estimates, COA/decision/MA briefs),
# and short type calls with stated precedence that may conflict with the content.
# Sources (all public, Distribution A: "Approved for public release; distribution is unlimited"):
#   FM 5-0 Planning and Orders Production (Army, 2022): MDMP products, WARNORD / OPORD / FRAGO formats, running estimates, COA & decision briefs.
#   FM 6-0 Commander and Staff Organization and Operations (Army, 2022): running estimates, staff briefings.
#   FM 6-99 U.S. Army Report and Message Formats (2021): FRAGO, SITREP, SPOTREP, SALUTE, RFI formats.
#   ATP 4-02.2 Medical Evacuation (2019): 9-line MEDEVAC categories (urgent / priority / routine).
#   FM 1-02.1 Operational Terms (2021): abbreviations (FRAGORD, WARNO, BP, PL, LD, NAI, CCIR/PIR/FFIR, PACE, REDCON).
# Label policy (documented in P2D_LABEL_POLICY.md):
#   content level = time-to-effect / threat rule; stated precedence honored if within ONE level of content,
#   content wins if the gap is >= 2 levels (this reproduces every p2c decoy label), and Flash content is never lowered.
import os, re, json, random, collections, hashlib, argparse
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--root',default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); a=ap.parse_args()
R=a.root; O=os.path.join(R,'p2d','data'); os.makedirs(O,exist_ok=True)
rng=random.Random(2027)
PRECS=['Flash','Immediate','Priority','Routine']; LV={p:i for i,p in enumerate(PRECS)}
def policy(content,stated):
    if stated is None: return content
    c,s=LV[content],LV[stated]
    if c==0: return 'Flash'                 # life/safety floor
    if abs(c-s)<=1: return stated           # originator's call within one level
    return content                          # >=2 levels: content wins (p2c decoy rule)
SRC=dict(FM50='FM 5-0',FM60='FM 6-0',FM699='FM 6-99',MED='FM 6-99 / ATP 4-02.2',ABBR='FM 1-02.1')
# ---------- pools (train / test disjoint names) ----------
POOL={'train':dict(tf=['ANVIL','GRANITE','HAWKEYE','VIKING','LANCER','IRONWOOD','TALON','COPPERHEAD'],co=['A','B','C'],pl=['RED','BLUE','GOLD','OAK'],obj=['ELM','PINE','HAMMER','SPUR'],
   msr=['TAMPA','DOVER','IOWA'],zones=['31U','32T'],calls=['Anvil 3','Talon 2-6','Granite 6','Hawkeye 1-6','Viking 2','Lancer 3-1'],town=['KIRBY','ALDEN'],dirs=['north','east','NE','SW']),
 'test':dict(tf=['SABER','RAPTOR','BULLDOG','OUTLAW','REAPER','MUSTANG'],co=['D','HHC'],pl=['SILVER','ASH','COPPER'],obj=['CEDAR','ANVIL WEST','FLINT'],
   msr=['BOSTON','RENO'],zones=['33T','34S'],calls=['Saber 5','Raptor 1-2','Bulldog 6','Outlaw 3','Reaper 2-1'],town=['MARLOW','ESTON'],dirs=['northwest','southeast','S','W'])}
def G(p): return f"{rng.choice(p['zones'])} {rng.choice(['NB','PC','QD','MV','LU'])} {rng.randint(1000,9999)} {rng.randint(1000,9999)}"
def dtg(): return f"{rng.randint(1,28):02d}{rng.randint(0,23):02d}{rng.choice(['00','15','30','45'])}Z{rng.choice(['JAN','MAR','MAY','JUL','SEP','NOV'])}{rng.choice(['25','26'])}"
def F(t,p):
    return t.format(tf=rng.choice(p['tf']),tf2=rng.choice(p['tf']),co=rng.choice(p['co']),pl=rng.choice(p['pl']),obj=rng.choice(p['obj']),msr=rng.choice(p['msr']),g=G(p),
      n=rng.randint(1,9),bp=rng.randint(1,12),c=rng.choice(p['calls']),town=rng.choice(p['town']),d=rng.choice(p['dirs']),dtg=dtg(),op=f"{rng.randint(1,12):02d}-26",
      fr=f"{rng.randint(1,60):03d}",wo=rng.randint(1,30),hh=rng.choice(['14','16','18','20','22']),pct=rng.choice([30,35,40,45]))
# timing phrases -> content precedence (time-to-effect rule, matches p2 FRAGO labels: immediately/imminent=Flash, NLT ..L today=Immediate,
# tomorrow=Priority, next week=Routine). Last item of each list is TEST-only.
TIMING={'Flash':['effective immediately; enemy attack imminent','execute now, enemy assault expected within 15 minutes','immediately, contact expected at any moment','NOW; EN armor crossing the PL'],
 'Immediate':['NLT {hh}00L today','within 4 hours of receipt','by {hh}30L tonight','NLT 3 hours from receipt'],
 'Priority':['NLT 0600 tomorrow','in 72 hours','within 48 hours','by 1200 the day after tomorrow'],
 'Routine':['next week','in 10 days','during the next refit cycle','NLT the 15th of next month']}
def timing(p,split): L=TIMING[p]; return L[-1] if split=='test' else rng.choice(L[:-1])
# tasks: every 4th is TEST-only
TASKS=['assumes the screen along PL {pl}','conducts route clearance of MSR {msr}','occupies BP {bp}','establishes OP {n} vic {g}','relieves {co} Co in place at OBJ {obj}',
 'displaces the CP to {g}','conducts a forward passage of lines through TF {tf2}','secures the bridge at {g}','provides overwatch of the {co} Co breach','conducts area security of the BSA',
 'shifts its northern boundary to grid line {n}0','emplaces obstacle group {n} vic {g}','conducts a raid on OBJ {obj}','establishes a TCP at {g}','conducts reconnaissance of NAI {n}',
 'escorts the CLP to FOB {town}','establishes a hasty defense on OBJ {obj}','clears the village of {town}','assumes the battalion QRF mission','conducts link-up with TF {tf2} at LU {n}']
def tasks(split): return [t for i,t in enumerate(TASKS) if (i%4==3)==(split=='test')]
SIT=['enemy reconnaissance active along PL {pl}','no significant change','enemy forces massing vic {g}','weather degrading, visibility under 1 km','enemy indirect fire increasing in sector']
def sit(p): return rng.choice(SIT) if p!='Flash' else rng.choice(['enemy attack imminent from the {d}','EN mech company advancing on BP {bp}','enemy assault underway vic {g}'])
# ---------- family generators: each returns (text,type,content_prec,doc,section,template_id) ----------
FRAGO_L=[ # FM 6-99 / FM 5-0 FRAGO format: references, time zone, situation changes, mission/tasks changes, ACKNOWLEDGE
 'FRAGORD {fr} to OPORD {op}. Time zone: L. 1. SITUATION: {S}. 2. MISSION: {co} Co, TF {tf} {T} {W}. 3. EXECUTION: no other change. ACKNOWLEDGE.',
 'FRAGO {fr} / OPORD {op} / DTG {dtg} / TO: {co} Co, TF {tf} / SIT: {S} / TASK: {T} {W} / ACK.',
 'FRAGO {fr}: {co} Co {T} {W}. Sit: {S}. All other paragraphs no change. Acknowledge.',
 'Change {fr} to OPORD {op}: TF {tf} {T} {W}. Coordinating instructions unchanged. ACKNOWLEDGE.',
 'FRAGORD {fr} (OPORD {op}) | {co} Co, TF {tf} | SITUATION: {S} | MISSION CHANGE: {T}, {W} | SUSTAINMENT: no change | ACKNOWLEDGE',  # test
 'TF {tf} FRAGO {fr}. {S}. {co} Co will {T2} {W}. Report when complete. ACK.']  # test
WARN_L=[ # FM 5-0 WARNORD format: situation, mission, execution (intent, tasks, time schedule), sustainment, command and signal
 'WARNORD {wo}. Ref OPORD {op}. 1. SITUATION: {S}. 2. MISSION: {co} Co, TF {tf} {T} {W}. 3. EXECUTION: Commander\'s intent: retain freedom of maneuver. Rehearsal at {hh}00. 4. SUSTAINMENT: {SU}. 5. COMMAND AND SIGNAL: CP locations unchanged. ACKNOWLEDGE.',
 'WARNO #{wo} / TF {tf} / Sit: {S} / Msn: {co} Co {T} {W} / Time schedule: MA brief {hh}00, COA brief {hh}30 / Sust: {SU} / ACK',
 'WARNING ORDER {wo}: {co} Co, TF {tf} prepares to {T2} {W}. Earliest movement time {dtg}. Initial planning guidance to follow. {SU}. Acknowledge.',
 'WARNORD {wo} (OPORD {op} to follow). Situation: {S}. Mission: be prepared to {T2} {W}. Tasks: {co} Co lead, priority of fires to lead element. Sustainment: {SU}. ACKNOWLEDGE.',  # test
 'TF {tf} WARNO {wo}. Expect mission to {T2} {W}. Begin TLPs now; confirmation brief at {hh}00. {SU}.']  # test
OPORD_L=[ # FM 5-0 five-paragraph OPORD excerpts
 'OPORD {op}, para 3.b, Tasks to Subordinate Units: {co} Co {T} {W}.',
 'OPORD {op} para 3.c Coordinating Instructions: all elements report REDCON 1 {W}; {co} Co {T}.',
 'OPORD {op} para 4 Sustainment: Class III and V push to {co} Co {W}; LRP at {g}.',
 'OPORD {op} para 5 Command and Signal: main CP displaces to {g} {W}; PACE primary FM, alternate JCR.',  # test
 'Extract, OPORD {op} Execution: scheme of maneuver changes; {co} Co {T} {W}.']  # test
SUST=['draw 2 DOS Class I','top off Class III','Class V basic load on hand','no change']
RE_SEC={'S4':{'Flash':None,'Immediate':'Class III at {pct}%, below the 60% minimum for LD in 4 hours; recommend emergency resupply',
              'Priority':'projected Class III shortfall in 48 hours; recommend push on tomorrow\'s LOGPAC','Routine':'all classes of supply within tolerance; no recommendations'},
        'S1':{'Flash':None,'Immediate':'{co} Co at 68% strength after IDF; recommend replacements before the attack tonight',
              'Priority':'personnel strength 91%; 12 replacements due in 72 hours','Routine':'strength 97%; no shortfalls; awards board next week'},
        'S6':{'Flash':None,'Immediate':'primary and alternate lost with {co} Co; recommend retrans team move within 3 hours',
              'Priority':'SATCOM degraded, PACE alternate working; fix expected tomorrow','Routine':'all systems green; COMSEC changeover next month'},
        'S2':{'Flash':'EN armor crossed PL {pl}, attack on BP {bp} expected within 30 minutes; recommend immediate shift to alternate BPs',
              'Immediate':'EN massing vic {g}; attack likely tonight; recommend increased R&S on NAI {n}','Priority':'EN recon activity up 20% over 72 hours; no change to MLCOA','Routine':'no change to enemy assessment; next update next week'},
        'FSO':{'Flash':None,'Immediate':'2 of 6 howitzers NMC; priority targets cannot be covered for tonight\'s attack','Priority':'Class V (155mm) at 70%; resupply in 48 hours','Routine':'all fire support systems FMC; no changes'}}
RE_L=[ # FM 5-0 / FM 6-0 running estimate: facts, assumptions, status, conclusions/recommendations
 '{sec} running estimate update, {dtg}. Facts: {X}. Assumptions unchanged. Conclusion: see recommendation.',
 '{sec} RUNNING ESTIMATE // DTG {dtg} // TF {tf} // STATUS: {X}',
 'Running estimate ({sec}), TF {tf}: {X}.',  # test
 '{sec} estimate change: {X}. Assumption review at next BUB.']  # test
BRIEF_L=[ # FM 5-0 MDMP briefings: mission analysis brief, COA brief, decision brief
 'COA brief, TF {tf}: COA 1 attacks along MSR {msr}; COA 2 envelops via OBJ {obj}. Staff recommends COA {k}. Commander decision required {W}.',
 'Mission analysis brief, TF {tf}: specified tasks {T2}; constraints: no movement before {dtg}. Approval of restated mission needed {W}.',
 'DECISION BRIEF / TF {tf} / Issue: {co} Co task organization / Recommendation: COA {k} / Decision needed {W}',
 'Wargame results, TF {tf}: COA {k} defeats EN MLCOA at PL {pl}; decision point 2 refined. Back-brief to the commander {W}.',  # test
 'COA decision brief: S3 recommends COA {k} for the move to OBJ {obj}; risk moderate. CDR approval {W}.']  # test
def lay(L,split,ntest): return L[-ntest:] if split=='test' else L[:-ntest]
def gen_order(fam,prec,split,p):
    if fam=='frago': L,doc,sec=lay(FRAGO_L,split,2),SRC['FM699'],'FM 6-99 FRAGO format; FM 5-0 fragmentary order'
    elif fam=='warnord': L,doc,sec=lay(WARN_L,split,2),SRC['FM50'],'FM 5-0 warning order format'
    else: L,doc,sec=lay(OPORD_L,split,2),SRC['FM50'],'FM 5-0 operation order (five-paragraph) format'
    i=rng.randrange(len(L)); t=rng.choice(tasks(split)); T2=re.sub(r'^(\w+?)s\b',r'\1',t).replace('assume','assume').replace('conduct ','conduct ')
    T2=t.split(' ',1); T2=(T2[0][:-1] if T2[0].endswith('s') else T2[0])+' '+T2[1]
    txt=L[i].replace('{S}',sit(prec)).replace('{T}',t).replace('{T2}',T2).replace('{W}',timing(prec,split)).replace('{SU}',rng.choice(SUST))
    return F(txt,p),'FRAGO/order',prec,doc,sec,f'{fam}:{L[i][:30]}'
def gen_re(prec,split,p):
    L=lay(RE_L,split,2); secs=[s for s,v in RE_SEC.items() if v[prec]]; s=rng.choice(secs); i=rng.randrange(len(L))
    return F(L[i].replace('{sec}',s).replace('{X}',RE_SEC[s][prec]),p),'SITREP',prec,SRC['FM50'],'FM 5-0 / FM 6-0 running estimate (facts, assumptions, conclusions, recommendations)',f're:{L[i][:30]}'
def gen_brief(prec,split,p):
    L=lay(BRIEF_L,split,2); i=rng.randrange(len(L)); t=rng.choice(tasks(split)); T2=t.split(' ',1); T2=(T2[0][:-1] if T2[0].endswith('s') else T2[0])+' '+T2[1]
    return F(L[i].replace('{T2}',T2).replace('{W}',timing(prec,split)).replace('{k}',str(rng.randint(1,2))),p),'other',prec,SRC['FM50'],'FM 5-0 MDMP briefings (mission analysis, COA, decision brief)',f'brief:{L[i][:30]}'
# ---------- short type calls: (type, content level, content, test_only) ----------
TC=[('SITREP','Routine','all quiet, PACE green',1),('SITREP','Routine','no change, all elements REDCON 2',0),('SITREP','Routine','NSTR, 100% accountability',0),('SITREP','Routine','quiet night, LOGPAC complete',0),
 ('SITREP','Routine','nothing significant, comms green',1),('SITREP','Priority','Class III at 60%, need resupply tomorrow',0),('SITREP','Priority','1 vic NMC, parts on order, still mission capable',0),
 ('SITREP','Priority','MEDEVAC complete, all pax at Role 2, 1 vic down',1),('SITREP','Immediate','lost comms with 2nd PLT 30 min, trying alternate',0),('SITREP','Immediate','CP taking sporadic IDF, no cas, displacing tonight',1),
 ('SITREP','Flash','CP under attack, displacing now',0),('SITREP','Flash','perimeter breached at {g}, fighting in the CP',1),
 ('SPOTREP','Flash','TIC grid {g}',0),('SPOTREP','Flash','contact {d}, squad-size EN with RPGs',0),('SPOTREP','Flash','in contact, EN platoon, {g}',1),('SPOTREP','Immediate','EN vics moving toward PL {pl}',0),
 ('SPOTREP','Immediate','4 dismounts emplacing IED vic {g}',1),('SPOTREP','Priority','EN OP observed {d}, no movement',0),('SPOTREP','Priority','fresh tracks at {g}, likely EN recon',1),
 ('SPOTREP','Routine','old fighting positions found {g}, unoccupied',0),('SPOTREP','Routine','abandoned EN vic {g}, no activity',1),
 ('SALUTE','Flash','8 pax assaulting OP {n}, PKM, {g}',0),('SALUTE','Immediate','6 dismounts digging in {d} of BP {bp}',0),('SALUTE','Priority','2 pax observing MSR {msr} from treeline',0),
 ('SALUTE','Routine','1 civ vic parked {g}, no activity, 48 hrs',0),('SALUTE','Immediate','squad moving toward PL {pl} w/ RPG',1),('SALUTE','Routine','EN trench line, unoccupied, {g}',1),
 ('9-line MEDEVAC','Flash','2x urgent surgical, grid {g}',0),('9-line MEDEVAC','Flash','1x urgent, arterial bleed, LZ hot',1),('9-line MEDEVAC','Immediate','1x priority, condition worsening, LZ cold',0),
 ('9-line MEDEVAC','Priority','2x priority litter, stable, {g}',0),('9-line MEDEVAC','Routine','1x routine ambulatory, sprained ankle',0),('9-line MEDEVAC','Routine','2x routine, sick call transport',1),
 ('9-line MEDEVAC','Priority','1x priority, fracture, stable',1),
 ('RFI','Flash','confirm FRD positions now, fires pending danger close',0),('RFI','Immediate','confirm bridge class before {hh}00 crossing',0),('RFI','Immediate','need status of MSR {msr} before LD today',1),
 ('RFI','Priority','need Class IX parts NLT tomorrow',0),('RFI','Priority','updated graphics needed by 0600 tomorrow',1),('RFI','Routine','request next week\'s range schedule',0),('RFI','Routine','request updated SOP copy, no suspense',1),
 ('FRAGO/order','Flash','all elements occupy alternate BPs now, attack imminent',0),('FRAGO/order','Immediate','{co} Co assumes QRF NLT {hh}00L today',0),('FRAGO/order','Immediate','{co} Co shifts to BP {bp} within 3 hrs',1),
 ('FRAGO/order','Priority','SP moves to 0800 tomorrow',0),('FRAGO/order','Priority','convoy brief moves to 1000 tomorrow',1),('FRAGO/order','Routine','maintenance stand-down moves to next week',0),('FRAGO/order','Routine','motor pool hours change next month',1),
 ('CCIR trigger','Flash','PIR 1 met: EN armor crossing PL {pl}',0),('CCIR trigger','Immediate','FFIR met: {co} Co below 70% strength',0),('CCIR trigger','Priority','PIR 3 met: EN recon element at NAI {n}',0),
 ('CCIR trigger','Routine','FFIR 5 met: generator NMC at the TOC, backup running',0),('CCIR trigger','Flash','PIR 2 met: EN artillery displacing forward',1),('CCIR trigger','Priority','FFIR 4 met: Class V at 70%',1),
 ('routine admin','Priority','PERSTAT due 1800 today',0),('routine admin','Priority','sensitive items report due NLT 2100',1),('routine admin','Routine','range day next month',0),('routine admin','Routine','awards packets due end of month',0),
 ('routine admin','Routine','unit photo moved to Friday',1)]
TW={'SITREP':['SITREP','sitrep','SIT REP'],'SPOTREP':['SPOTREP','SPOT REPORT','spot rpt'],'SALUTE':['SALUTE','SALUTE RPT'],'9-line MEDEVAC':['MEDEVAC','9-LINE','9-line MEDEVAC','medevac req'],
 'RFI':['RFI','rfi'],'FRAGO/order':['FRAGO','FRAGORD'],'CCIR trigger':['CCIR','CCIR ALERT'],'routine admin':['ADMIN','S1 NOTE','admin']}
SW={'Flash':['FLASH','flash','FLASH FLASH'],'Immediate':['IMMEDIATE','immediate','IMMED'],'Priority':['PRIORITY','priority','PRIO'],'Routine':['ROUTINE','routine','RTN']}
TC_L=['{S} {T}: {C}','{T} ({S}): {C}','{c} {S} {T}: {C}','{S} {T} - {C}','{S} // {T} // {C}','{C}, {S} {T}']  # last two test-only
def gen_tc(split,p,want_stated_gap):
    items=[x for x in TC if (x[3]==1)==(split=='test')]; ty,cl,c,_=rng.choice(items); L=lay(TC_L,split,2)
    # stated precedence: None, same, 1 level off, >=2 levels off
    if want_stated_gap is None: st=None
    else:
        cand=[q for q in PRECS if abs(LV[q]-LV[cl])==want_stated_gap] if want_stated_gap<2 else [q for q in PRECS if abs(LV[q]-LV[cl])>=2]
        if ty=='routine admin': cand=[q for q in cand if LV[policy(cl,q)]>=2]
        if not cand: return None
        st=rng.choice(cand)
    lab=policy(cl,st); i=rng.randrange(len(L)); tmpl=L[i] if st else rng.choice(['{T}: {C}','{c} {T}: {C}','{T} - {C}'] if split=='train' else ['{T} // {C}','{C} ({T})'])
    txt=F(tmpl.replace('{S}',rng.choice(SW[st]) if st else '').replace('{T}',rng.choice(TW[ty])).replace('{C}',c),p)
    if rng.random()<0.2: txt=txt.upper()
    gap=None if st is None else abs(LV[cl]-LV[st])
    return dict(text=re.sub(r'\s{2,}',' ',txt).strip(),type=ty,precedence=lab,content_prec=cl,stated_prec=st,stated_gap=gap,source_doc=SRC['MED'] if ty=='9-line MEDEVAC' else (SRC['FM699'] if ty!='CCIR trigger' else SRC['FM60']),
                source_section='FM 6-99 report formats + FM 1-02.1 abbreviations (short type call)',template_id=f'tc:{tmpl}|{c[:24]}')
SUBS=[(r'\bpersonnel\b','pax'),(r'\bvehicles?\b','vics'),(r'\bwith\b','w/'),(r'\benemy\b','EN'),(r'\bfriendly\b','FRD'),(r'\bhours?\b','hrs'),(r'\bminutes?\b','min'),(r'\bposition\b','pos'),
 (r'\bbattalion\b','BN'),(r'\bcompany\b','CO'),(r'\bvicinity\b','vic'),(r'\breport\b','rpt'),(r'\brequest\b','req'),(r'\bno later than\b','NLT'),(r'\btomorrow\b','tmrw')]
def abbreviate(t):
    for x,y in SUBS:
        if rng.random()<0.6: t=re.sub(x,y,t,flags=re.I)
    return t
def build(split,scale):
    p=POOL[split]; out=[]
    Q=dict(frago=dict(Flash=260,Immediate=260,Priority=260,Routine=260),warnord=dict(Flash=120,Immediate=190,Priority=190,Routine=190),
           opord=dict(Flash=100,Immediate=140,Priority=140,Routine=140),re=dict(Flash=80,Immediate=150,Priority=150,Routine=150),brief=dict(Flash=60,Immediate=150,Priority=150,Routine=150))
    for fam,q in Q.items():
        for prec,n in q.items():
            for _ in range(int(n*scale)):
                r=gen_order(fam,prec,split,p) if fam in('frago','warnord','opord') else gen_re(prec,split,p) if fam=='re' else gen_brief(prec,split,p)
                txt=r[0]
                if rng.random()<0.35: txt=abbreviate(txt)
                out.append(dict(text=txt,type=r[1],precedence=r[2],content_prec=r[2],stated_prec=None,stated_gap=None,source_doc=r[3],source_section=r[4],template_id=r[5],kind=fam))
    # type calls: balanced over label precedence, mix of stated gaps
    cell=collections.Counter(); target=int(450*scale); tries=0
    while min(cell[(g,q)] for g in ['none','0','1','2+'] for q in PRECS)<target//4 and tries<400000:
        tries+=1; g=rng.choice([None,0,1,2]); x=gen_tc(split,p,g)
        if not x: continue
        k=('none' if g is None else '2+' if g>=2 else str(g),x['precedence'])
        if cell[k]>=target//4: continue
        if rng.random()<0.3: x['text']=abbreviate(x['text'])
        x['kind']='typecall'; cell[k]+=1; out.append(x)
    for x in out: x['split']=split; x['n_words']=len(x['text'].split()); x['method']='template (rule-based, no LLM)'; x['dist_statement']='Approved for public release; distribution is unlimited'
    return out
trn=build('train',1.0); tst=build('test',0.25)
# ---------- dedupe / leak checks ----------
norm=lambda t:re.sub(r'[^a-z0-9 ]','',t.lower()).split(); key=lambda t:' '.join(norm(t))
rj=lambda f:[json.loads(x) for x in open(f,encoding='utf-8') if x.strip()]
P2=os.path.join(R,'p2','data'); P2C=os.path.join(R,'p2c','data')
p2tr=[x['text'] for x in rj(os.path.join(P2,'train.jsonl'))]; p2te=[x['text'] for x in rj(os.path.join(P2,'test.jsonl'))]
p2caug=[x['text'] for x in rj(os.path.join(P2C,'aug_train.jsonl'))]; p2cte=[x['text'] for x in rj(os.path.join(P2C,'short_test.jsonl'))]+[x['text'] for x in rj(os.path.join(P2C,'noword_test.jsonl'))]
probes=json.load(open(os.path.join(R,'p2c','probes.json')))+json.load(open(os.path.join(R,'p2c','probe_inputs.json')))
class NearIdx:
    def __init__(s,texts):
        s.sets=[set(norm(t)) for t in texts]; s.inv=collections.defaultdict(list)
        for i,x in enumerate(s.sets):
            for w in x: s.inv[w].append(i)
    def near(s,t,th=0.85):
        q=set(norm(t)); 
        if not q: return True
        c=collections.Counter(i for w in q if len(s.inv[w])<5000 for i in s.inv[w])
        return any(len(q&s.sets[i])/len(q|s.sets[i])>=th for i,_ in c.most_common(40))
drop=collections.Counter()
evalkeys=set(map(key,p2te+p2cte+probes)); evalnear=NearIdx(p2te+p2cte+probes)
seen=set(); T=[]
for x in trn:
    k=key(x['text'])
    if k in seen: drop['train_exact_dup']+=1; continue
    if k in evalkeys: drop['train_exact_in_eval']+=1; continue
    if evalnear.near(x['text']): drop['train_near_eval>=0.85']+=1; continue
    seen.add(k); T.append(x)
trn=T
trainkeys=seen|set(map(key,p2tr+p2caug)); trnear=NearIdx(p2tr+p2caug+[x['text'] for x in trn])
tseen=set(); T=[]
for x in tst:
    k=key(x['text'])
    if k in trainkeys: drop['test_exact_in_train']+=1; continue
    if k in tseen: drop['test_dup']+=1; continue
    if k in evalkeys: drop['test_exact_in_existing_eval']+=1; continue
    if trnear.near(x['text']): drop['test_near_train>=0.85']+=1; continue
    tseen.add(k); T.append(x)
tst=T
tt=set(x['template_id'] for x in trn)&set(x['template_id'] for x in tst); assert not tt, f'template overlap {list(tt)[:3]}'
for i,x in enumerate(trn): x['id']=f'p2d-train-{i:05d}'
for i,x in enumerate(tst): x['id']=f'p2d-test-{i:05d}'
def dump(L,f):
    with open(os.path.join(O,f),'w',encoding='utf-8',newline='\n') as fh:
        for x in L: fh.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
dump(trn,'p2d_train.jsonl'); dump(tst,'p2d_test.jsonl')
def stats(L):
    d=pd.DataFrame(L)
    return dict(n=len(L),by_kind_prec=d.groupby(['kind','precedence']).size().unstack(fill_value=0).to_dict('index'),by_type=d.type.value_counts().to_dict(),by_prec=d.precedence.value_counts().to_dict(),
      by_source_doc=d.source_doc.value_counts().to_dict(),typecall_stated_gap=d[d.kind=='typecall'].stated_gap.fillna(-1).astype(int).value_counts().sort_index().rename({-1:'none'}).to_dict(),
      median_words=int(d.n_words.median()),n_templates=int(d.template_id.nunique()))
info=dict(train=stats(trn),test=stats(tst),dropped=dict(drop),template_overlap=0,
 sha256={f:hashlib.sha256(open(os.path.join(O,f),'rb').read()).hexdigest() for f in ['p2d_train.jsonl','p2d_test.jsonl']})
json.dump(info,open(os.path.join(O,'p2d_data_stats.json'),'w'),indent=1,default=str); print(json.dumps(info,indent=1,default=str))
