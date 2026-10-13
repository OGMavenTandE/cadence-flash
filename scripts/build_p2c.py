# Cadence Flash p2c data build: short / shorthand / no-precedence-word / decoy / balance augmentations.
# Rule-based only, from the existing public Distribution A military templates (FM 6-99, ATP 4-02.2, FM 6-0)
# and FM 1-02.1-style abbreviations. Train augmentations use TRAIN rows only; the held-out short slice and
# the test-side eval slices use TEST rows only (FM 5-0 test rows stay out of training). Leak-checked against train.
import os, re, json, random, collections
import pandas as pd
R=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(R,'p2','data'); O=os.path.join(R,'p2c','data'); os.makedirs(O,exist_ok=True)
rng=random.Random(2026)
tr=pd.read_json(os.path.join(D,'train.jsonl'),lines=True); te=pd.read_json(os.path.join(D,'test.jsonl'),lines=True)
PRECS=['Flash','Immediate','Priority','Routine']
PW=re.compile(r'\b(flash|immediate(?:ly)?|priority|routine|urgent|O/O|Z|P|R)\b(?=[\s:.,/-]|$)',re.I)
PWORD=re.compile(r'\b(FLASH|IMMEDIATE|PRIORITY|ROUTINE)\b',re.I)
def strip_prec(t):
    t=re.sub(r'(?i)\b(FLASH\s+)+','',t); t=re.sub(r'(?i)\bIMMEDIATE\b(?=\s*(MEDEVAC|SITREP|SPOTREP|RFI|FRAGO|[:.\-]))','',t)
    t=re.sub(r'(?i)\b(PRIORITY|ROUTINE)\b\s*(?=(SITREP|SPOTREP|RFI|FRAGO|MEDEVAC|SALUTE|CCIR|[:\-]))','',t)
    t=re.sub(r'(?i)\bprecedence\s*:\s*\w+[.;,]?','',t)
    return re.sub(r'\s{2,}',' ',re.sub(r'^\W+','',t)).strip()
nw=lambda t:len(t.split())
ABBR={'TYPE':{'SPOTREP':['SPOTREP','spot rpt','SPOT'],'SALUTE':['SALUTE','SALUTE rpt'],'9-line MEDEVAC':['MEDEVAC','9-line','9 line','medevac req'],
 'CCIR trigger':['CCIR','CCIR met','PIR met','FFIR'],'SITREP':['SITREP','sitrep','SIT'],'FRAGO/order':['FRAGO','FRAGORD','FRAGO:','order'],'RFI':['RFI','rfi'],'routine admin':['',''],'other':['','']}}
SUBS=[(r'\bpersonnel\b','pax'),(r'\bvehicles?\b','vics'),(r'\bwith\b','w/'),(r'\bwithout\b','w/o'),(r'\bcrossroads\b','xroads'),(r'\bapproximately\b|\bapprox\b','~'),
 (r'\benemy\b','EN'),(r'\bfriendly\b','FRD'),(r'\bposition\b','pos'),(r'\breport(?:ed|s)?\b','rpt'),(r'\brequest\b','req'),(r'\bwounded\b','WIA'),(r'\bkilled\b','KIA'),
 (r'\bindirect fire\b','IDF'),(r'\bimprovised explosive device\b','IED'),(r'\bobservation post\b','OP'),(r'\bimmediately\b','ASAP'),(r'\bbattalion\b','BN'),(r'\bcompany\b','CO'),(r'\bhours?\b','hrs'),(r'\bminutes?\b','min')]
def abbreviate(t,p=0.6):
    for a,b in SUBS:
        if rng.random()<p: t=re.sub(a,b,t,flags=re.I)
    return t
KEYS=[r'ACTIVITY:\s*(.+?)(?:\s/\s|\s\|\s|\.\sLine|\.\s[A-Z]+:|$)',r'\bA - (.+?);',r'MISSION:\s*(.+?)(?:\.\sLine|$)',r'REQUEST:\s*(.+?)(?:\.\sLine|$)',
 r'EFFECTIVE:\s*(.+?)(?:\.\sLine|$)',r'EFF:\s*(.+?)(?:\s\|\s|$)',r'(?:has been met|met)\.\s*(.+?)(?:\.\s*Reported|\.\s*Req|$)']
def clause(row):
    tt=row.template_text
    if row.type=='9-line MEDEVAC':
        m=re.search(r'Line 3:\s*([^.]+?)\s*\(([^)]+)\)',tt); g=re.search(r'Line 1:\s*([0-9]{2}[A-Z] [A-Z]{2} \d{4} \d{4})',tt)
        if m: return f"{m.group(1)} {m.group(2)}"+(f", grid {g.group(1)}" if g and rng.random()<0.6 else ''),True
    for k in KEYS:
        m=re.search(k,tt)
        if m and nw(m.group(1))>=2: return m.group(1).strip(' .;'),True
    if row.type in('routine admin','other') or nw(tt)<=25: return tt.strip(),False
    return None,False
def callsign(src):
    return rng.choice(src['calls'])
TRPOOL=dict(calls=['Anvil 3','Talon 2-6','Granite 6','Hawkeye 1-6','Viking 2','Copperhead 6','Lancer 3-1','Ironwood 4'],zones=['31U','32T'],dirs=['north','east','south','west','NE','SW'])
TEPOOL=dict(calls=['Saber 5','Raptor 1-2','Bulldog 6','Outlaw 3','Reaper 2-1','Mustang 4'],zones=['33T','34S'],dirs=['northwest','southeast','N','E','S','W'])
def grid(src): return f"{rng.choice(src['zones'])} {rng.choice(['NB','PC','QD','MV','NV','LU'])} {rng.randint(1000,9999)} {rng.randint(1000,9999)}"
def short_of(row,src,maxw=15):
    c,_=clause(row)
    if not c: return None
    c=strip_prec(c); ab=rng.choice(ABBR['TYPE'][row.type]) if rng.random()<0.65 else ''
    pre=rng.choice(['',f'{callsign(src)}: ',f'{callsign(src)} ',f'{ab}: ' if ab else '',f'{callsign(src)} {ab}: ' if ab else ''])
    t=pre+c
    if rng.random()<0.5: t=abbreviate(t)
    w=t.split()
    if len(w)>maxw: t=' '.join(w[:rng.randint(max(3,maxw-6),maxw)])
    if rng.random()<0.3: t=t.lower()
    t=t.strip(' ,;:')
    return t if 3<=nw(t)<=maxw else None
def rec(text,ty,pr,kind,src_row=None,split='train'):
    return dict(text=text,type=ty,precedence=pr,kind=kind,split=split,source_doc=(src_row.source_doc if src_row is not None else 'FM 6-99 / FM 1-02.1 (shorthand rules)'),
                parent_id=(src_row.id if src_row is not None else None))
# ---- shorthand bank (FM 1-02.1 abbreviations; precedence by Scott's rules) ----
SH=[ # (template, type, precedence, test_only)
 ('TIC','SPOTREP','Flash',0),('troops in contact','SPOTREP','Flash',0),('Troops in contact','SPOTREP','Flash',0),('TIC {g}','SPOTREP','Flash',0),('TIC at grid {g}','SPOTREP','Flash',0),
 ('troops in contact at {g}','SPOTREP','Flash',0),('{c} TIC','SPOTREP','Flash',0),('{c}, troops in contact {d}','SPOTREP','Flash',0),('in contact, {d}','SPOTREP','Flash',0),
 ('contact front','SPOTREP','Flash',0),('contact {d}','SPOTREP','Flash',0),('contact {d}, {m}m, {sz}','SPOTREP','Flash',0),('TIC, {sz} w/ {w}','SPOTREP','Flash',0),
 ('taking fire from the {d}','SPOTREP','Flash',0),('under fire, {sz}, {g}','SPOTREP','Flash',0),('ambush on route {r}','SPOTREP','Flash',0),('TIC, req fire support','SPOTREP','Flash',0),
 ('IDF incoming','SPOTREP','Flash',0),('IDF impacting {loc}','SPOTREP','Flash',0),('incoming mortars at {loc}','SPOTREP','Flash',0),('rockets inbound {loc}','SPOTREP','Flash',0),
 ('{n}x WIA urgent','9-line MEDEVAC','Flash',0),('{n} urgent surgical','9-line MEDEVAC','Flash',0),('{n}x KIA {n}x WIA need MEDEVAC','9-line MEDEVAC','Flash',0),
 ('cas {n} urgent, {g}','9-line MEDEVAC','Flash',0),('MEDEVAC urgent {n} litter','9-line MEDEVAC','Flash',0),('man down, urgent','9-line MEDEVAC','Flash',0),
 ('{n} WIA, arterial bleed','9-line MEDEVAC','Flash',0),('line 3: {n}A urgent','9-line MEDEVAC','Flash',0),('need CAS at {g}','other','Flash',0),('TIC need CAS','other','Flash',0),
 ('req CAS danger close','other','Flash',0),('launch QRF','other','Flash',0),('QRF to {loc} now','other','Flash',0),('TIC, req QRF','other','Flash',0),
 ('S: {n} pax A: firing on us L: {g} U: dismounts T: {t} E: {w}','SALUTE','Flash',0),('SALUTE {n} pax attacking OP, {g}, {w}','SALUTE','Flash',0),
 ('sniper fire, {n} KIA, pinned down','SPOTREP','Flash',0),('TIC, {n}x WIA, need CAS','SPOTREP','Flash',0),('IED strike, {n} urgent cas','9-line MEDEVAC','Flash',0),
 # test-only Flash shorthand phrasings
 ('in contact w/ {sz}, {g}','SPOTREP','Flash',1),('TIC {d} of {loc}','SPOTREP','Flash',1),('{c} taking effective fire','SPOTREP','Flash',1),('IDF on {loc}, {n} WIA urgent','9-line MEDEVAC','Flash',1),
 ('urgent surgical x{n}, {g}','9-line MEDEVAC','Flash',1),('QRF needed, TIC {d}','other','Flash',1),('S {n} pax A ambushing convoy L {g} E {w}','SALUTE','Flash',1),
 # Immediate
 ('IED found route {r}, cordon set','SPOTREP','Immediate',0),('EN massing {d} of {loc}','SPOTREP','Immediate',0),('S: {n} pax A: moving toward OP L: {g} U: dismounts T: {t} E: {w}','SALUTE','Immediate',0),
 ('contact broken, {n}x WIA priority','9-line MEDEVAC','Immediate',0),('{sz} moving toward FRD lines {d}','SPOTREP','Immediate',0),('UAS overhead {loc}','SPOTREP','Immediate',0),
 ('suspected IED {g}','SPOTREP','Immediate',0),('req fire support for planned move in 1 hr','RFI','Immediate',0),('{sz} digging in {m}m {d}','SPOTREP','Immediate',1),('possible VBIED near ECP','SPOTREP','Immediate',1),
 # Priority
 ('{n}x WIA priority','9-line MEDEVAC','Priority',0),('{n} priority patients, {g}','9-line MEDEVAC','Priority',0),('class I resupply needed in 24 hrs','RFI','Priority',0),
 ('PERSTAT due 1800 today','routine admin','Priority',0),('sensitive items inventory due NLT 2100','routine admin','Priority',0),('weapons cards due to S1 by COB','routine admin','Priority',0),
 ('class IX parts needed NLT tomorrow','RFI','Priority',0),('EN obs post spotted {d}, no movement','SPOTREP','Priority',0),('updated graphics needed by 0600','RFI','Priority',1),('DTS vouchers due by Friday','routine admin','Priority',1),
 # Routine
 ('S: {n} pax A: maintenance far from lines L: {g} T: {t}','SALUTE','Routine',0),('{n} routine patients','9-line MEDEVAC','Routine',0),('PACE green, all quiet','SITREP','Routine',0),
 ('all quiet, no contact','SITREP','Routine',0),('old tracks {g}, 48 hrs old','SPOTREP','Routine',0),('laundry point moves next week','FRAGO/order','Routine',0),('req class I resupply, 3 days','RFI','Routine',0),
 ('chaplain visit Tuesday','routine admin','Routine',0),('range day next month','routine admin','Routine',0),('nothing significant to report','SITREP','Routine',1),('PT uniform change effective next week','routine admin','Routine',1),
]
def fill(t,src):
    return t.format(g=grid(src),c=callsign(src),d=rng.choice(src['dirs']),m=rng.choice([100,200,300,400,500,800]),sz=rng.choice(['squad-size EN','EN platoon','2 dismounts','fire team','EN squad','~10 pax']),
        w=rng.choice(['RPGs','PKM','AK-type rifles','RPG and PKM','82mm mortar']),r=rng.choice(['BLUE','IRON','ANVIL','GOLD','SILVER']),loc=rng.choice(['FOB','OP 2','the ECP','BP 3','the TOC','PB 4']),
        n=rng.randint(1,5),t=rng.choice(['0930','1415','2200','0545']))
def sh_variants(n,test,src):
    pool=[s for s in SH if (s[3]==1)==test or (test and s[3]==0 and s[2]!='Flash')] if test else [s for s in SH if s[3]==0]
    out=[]
    while len(out)<n:
        t,ty,pr,_=rng.choice(pool); x=fill(t,src)
        if rng.random()<0.25: x=x.upper()
        elif rng.random()<0.25: x=x.lower()
        if rng.random()<0.2: x=x+rng.choice(['.',' over',', over.','!'])
        out.append(rec(x,ty,pr,'shorthand',split='test' if test else 'train'))
    return out
DECOY_WORD={'Flash':['ROUTINE','PRIORITY','routine'],'Immediate':['ROUTINE'],'Priority':['FLASH'],'Routine':['FLASH','IMMEDIATE','FLASH FLASH']}
def build(df,src,split,counts):
    out=[]; df=df[df.source_doc!='FM 5-0'] if split=='train' else df
    st=df.assign(s=df.type+'|'+df.precedence)
    # short versions, balanced over type x precedence
    rows=st.groupby('s',group_keys=False).apply(lambda g:g.sample(min(len(g),counts['short']//st.s.nunique()+5),random_state=7))
    for r in rows.itertuples():
        x=short_of(r,src); 
        if x: out.append(rec(x,r.type,r.precedence,'short',r,split))
    # no-precedence-word copies (full or short)
    w=st[st.text.str.contains(PWORD.pattern,case=False)]; w=w.sample(min(len(w),counts['noword']),random_state=8)
    for r in w.itertuples():
        x=strip_prec(r.text); x=re.sub(PWORD,'',x); x=re.sub(r'\s{2,}',' ',x).strip(' :-')
        if nw(x)>=2: out.append(rec(x,r.type,r.precedence,'no_prec_word',r,split))
    # misleading-word counter-examples: word conflicts with content, label follows content
    d=st[st.precedence.isin(['Flash','Routine','Priority','Immediate'])].sample(counts['decoy']*2,random_state=9)
    for r in d.itertuples():
        base=short_of(r,src) if rng.random()<0.6 else re.sub(PWORD,'',strip_prec(r.text)).strip()
        if not base: continue
        x=f"{rng.choice(DECOY_WORD[r.precedence])} {base}" if rng.random()<0.7 else f"{base} ({rng.choice(DECOY_WORD[r.precedence]).lower()})"
        out.append(rec(x,r.type,r.precedence,'decoy',r,split))
        if sum(o['kind']=='decoy' for o in out)>=counts['decoy']: break
    # short Routine/Priority balance (deadline admin => Priority)
    b=st[st.precedence.isin(['Routine','Priority'])].sample(counts['balance'],random_state=10,replace=True)
    for r in b.itertuples():
        x=short_of(r,src,maxw=10)
        if x: out.append(rec(x,r.type,r.precedence,'balance',r,split))
    out+=sh_variants(counts['shorthand'],split=='test',src)
    return out
norm=lambda t:re.sub(r'[^a-z0-9 ]','',t.lower()).split()
trn=build(tr,TRPOOL,'train',dict(short=4600,noword=2000,decoy=500,balance=1350,shorthand=4000))
tst=build(te,TEPOOL,'test',dict(short=420,noword=0,decoy=120,balance=90,shorthand=500))
tst=[x for x in tst if x['kind']!='no_prec_word']
probes=set(' '.join(norm(p)) for p in json.load(open(os.path.join(R,'p2c','probes.json')))+json.load(open(os.path.join(R,'p2c','chief_inputs.json'))))
# exact-dup removal; probes never enter train
seen=set(); T2=[]
for x in trn:
    k=' '.join(norm(x['text']))
    if k in seen or k in probes: continue
    seen.add(k); T2.append(x)
trn=T2
trkeys=seen|set(' '.join(norm(t)) for t in tr.text)
trsets=[set(norm(t)) for t in list(tr.text)+[x['text'] for x in trn]]
inv=collections.defaultdict(list)
for i,s in enumerate(trsets):
    for w in s: inv[w].append(i)
def near(s,th=0.85):
    if not s: return True
    c=collections.Counter(i for w in s if len(inv[w])<4000 for i in inv[w])
    return any(len(s&trsets[i])/len(s|trsets[i])>=th for i,_ in c.most_common(30))
leak=collections.Counter(); T3=[]; tseen=set()
for x in tst:
    k=' '.join(norm(x['text']))
    if k in trkeys: leak['exact']+=1; continue
    if k in tseen: leak['dup_in_test']+=1; continue
    if near(set(norm(x['text']))): leak['near>=0.85']+=1; continue
    tseen.add(k); T3.append(x)
tst=T3
for x in trn+tst: x['n_words']=nw(x['text'])
pd.DataFrame(trn).to_json(os.path.join(O,'aug_train.jsonl'),orient='records',lines=True,force_ascii=False)
pd.DataFrame(tst).to_json(os.path.join(O,'short_test.jsonl'),orient='records',lines=True,force_ascii=False)
# test-side eval slices from TEST rows only
nop=[]
for r in te.itertuples():
    if PWORD.search(r.text):
        x=re.sub(r'\s{2,}',' ',re.sub(PWORD,'',strip_prec(r.text))).strip(' :-')
        nop.append(dict(text=x,type=r.type,precedence=r.precedence,kind='no_prec_word',source_doc=r.source_doc))
pd.DataFrame(nop).to_json(os.path.join(O,'noword_test.jsonl'),orient='records',lines=True,force_ascii=False)
st=lambda L:{k:dict(n=len(v),by_prec=dict(collections.Counter(x['precedence'] for x in v)),median_words=sorted(x['n_words'] for x in v)[len(v)//2]) for k,v in pd.DataFrame(L).groupby('kind').apply(lambda g:g.to_dict('records')).items()}
info=dict(train=len(trn),test=len(tst),leak_dropped=dict(leak),train_kinds=st(trn),test_kinds=st(tst),noword_test=len(nop),
 fm50_in_train=int(sum(x['source_doc']=='FM 5-0' for x in trn)),train_flash_le8=sum(1 for x in trn if x['precedence']=='Flash' and x['n_words']<=8),
 tic_in_train=sum(1 for x in trn if re.search(r'\bTIC\b',x['text'],re.I)))
json.dump(info,open(os.path.join(O,'p2c_data_stats.json'),'w'),indent=1); print(json.dumps(info,indent=1))
