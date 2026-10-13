import random
SRC={"FM 6-99":("FM 6-99","2021-05-17","Approved for public release; distribution is unlimited"),
"FM 6-0":("FM 6-0","2022-05-16","Approved for public release; distribution is unlimited"),
"FM 5-0":("FM 5-0","2024-11-04","Approved for public release; distribution is unlimited"),
"ATP 4-02.2":("ATP 4-02.2","2019-07","Approved for public release; distribution is unlimited")}
CS=["IRONWOOD","SABER","GRANITE","VIKING","TALON","COPPERHEAD","LANCER","HAWKEYE","BULLDOG","RAVEN","ANVIL","WOLFPACK"]
CO=["A","B","C","D","HHC"]
def unit(): return f"{random.choice(CO)} Co, TF {random.choice(CS)}"
def call(): return f"{random.choice(CS).title()} {random.choice(['6','3','2','1-6','2-6','3-1'])}"
def grid(): return f"{random.choice(['31U','32T','33T'])} {random.choice(['NB','PC','QD','MV'])} {random.randint(1000,9999)} {random.randint(1000,9999)}"
def dtg():
    return f"{random.randint(1,28):02d}{random.randint(0,23):02d}{random.choice(['00','15','30','45'])}Z{random.choice(['JAN','MAR','MAY','JUL','SEP','NOV'])}{random.choice(['25','26'])}"
def pick(*a): return random.choice(a)
# each builder returns (text, precedence, source, page, fields)
def spotrep():
    p=pick("Flash","Immediate","Priority","Routine")
    act={"Flash":pick("engaging our position with small arms and RPG","attacking north toward OP 2, troops in contact","emplacing indirect fire 800m from our perimeter, rounds incoming"),
         "Immediate":pick("moving south on MSR TAMPA toward our battle position","establishing a hasty checkpoint on Route BLUE","observed digging in on the ridge overlooking the crossing site"),
         "Priority":pick("conducting resupply at a cache site, no movement toward friendly forces","observed in a static assembly area, vehicles under camo netting"),
         "Routine":pick("abandoned fighting positions found, no personnel present","old vehicle tracks observed, est. 48+ hours old")}[p]
    size=pick("squad-size element (8-10 pax)","2x BTR-type wheeled APCs","platoon (approx 30 pax) with 3x trucks","1x technical vehicle, 4 pax","no personnel")
    t=(f"SPOTREP. Line 1: {dtg()}. Line 2: {unit()}, observed via {pick('binoculars','thermal','UAS','unaided')}. Line 3 SIZE: {size}. Line 4 ACTIVITY: {act}. "
       f"Line 5 LOCATION: {grid()}. Line 6 UNIT: unknown, {pick('dismounted','motorized','mechanized')}. Line 7 TIME: {dtg()}. Line 8 EQUIPMENT: {pick('small arms, RPG','82mm mortar','crew-served MG','none observed')}. "
       f"Line 9 ASSESSMENT: {pick('likely probing our defenses','likely screening force','intent unclear','site no longer in use')}.")
    return t,p,"FM 6-99","A-174",["DTG","UNIT","SIZE","ACTIVITY","LOCATION","UNIT","TIME","EQUIPMENT","ASSESSMENT","NARRATIVE","AUTH"]
def salute():
    p=pick("Flash","Immediate","Priority","Routine")
    a={"Flash":"firing on our convoy at this time","Immediate":"moving toward our checkpoint, ETA 15 min","Priority":"stationary at a crossroads, observing traffic","Routine":"conducting what appears to be routine maintenance far from friendly lines"}[p]
    t=(f"SALUTE report from {call()}: S - {pick('6 personnel','2 vehicles','approx 12 dismounts','1 armored vehicle')}; A - {a}; L - {grid()}; "
       f"U - {pick('unknown uniforms, no insignia','dark fatigues, armbands','civilian clothing with weapons')}; T - {dtg()}; E - {pick('AK-type rifles','RPG and PKM','mounted heavy MG','radios and binoculars')}.")
    return t,p,"FM 6-99","Glossary/A-2 key (SALUTE)",["SIZE","ACTIVITY","LOCATION","UNIT","TIME","EQUIPMENT"]
def medevac():
    p=pick("Flash","Immediate","Priority","Routine")
    pat={"Flash":pick("2A (urgent), 1B (urgent-surgical)","1A (urgent)","1B (urgent-surgical)","3A (urgent)"),"Immediate":pick("2C (priority)","1C (priority), 1D (routine)"),"Priority":pick("1C (priority)","2C (priority)"),"Routine":pick("1D (routine)","2D (routine)")}[p]
    sec={"Flash":pick("E - enemy troops in area, armed escort required","P - possible enemy in area","N - no enemy troops in area"),"Immediate":pick("P - possible enemy in area","E - enemy troops in area, armed escort required"),"Priority":"N - no enemy troops in area","Routine":"N - no enemy troops in area"}[p]
    t=(f"9-line MEDEVAC request. Line 1: {grid()}. Line 2: {random.randint(30,87)}.{random.randint(10,95)} MHz, call sign {call()}. Line 3: {pat}. "
       f"Line 4: {pick('A - none','B - hoist','C - extraction equipment','D - ventilator')}. Line 5: {pick('L2','A1, L1','L1','A2')}. Line 6: {sec}. "
       f"Line 7: {pick('B - panels','C - pyrotechnic signal','D - smoke','E - none')}. Line 8: A - US military. Line 9: {pick('none','N - no NBC contamination')}.")
    return t,p,"FM 6-99 / ATP 4-02.2","FM 6-99 A-114; ATP 4-02.2 ch.3 & app C",["LOC","FREQ/CALLSIGN","PATIENTS BY PRECEDENCE","SPECIAL EQUIP","PATIENTS BY TYPE","SECURITY","MARKING","NATIONALITY","NBC"]
def ccir():
    p=pick("Flash","Immediate","Priority","Routine")
    e={"Flash":pick(("PIR 1","Enemy armor confirmed crossing Phase Line GOLD toward the main defensive belt."),("PIR 1","Enemy rotary-wing assets observed inbound toward the brigade support area."),("FFIR 1","Friendly patrol has lost all communications and is overdue after reported contact."),("PIR 2","Enemy long-range fires systems confirmed moving into range of the main CP.")),
       "Immediate":pick(("PIR 2","Enemy engineer assets observed staging near Bridge 4, consistent with a crossing attempt in the next 6 hours."),("FFIR 2","Two of four bridging vehicles are non-mission capable ahead of tonight's crossing."),("PIR 3","Enemy reconnaissance element detected near the planned assault position.")),
       "Priority":pick(("FFIR 3","Fuel on hand for the battalion has dropped below 60%; projected resupply is 24 hours out."),("FFIR 4","Medical Class VIII stocks projected to fall below 3 days of supply by end of week."),("PIR 4","Local civilian traffic on the main route has dropped sharply over two days.")),
       "Routine":pick(("FFIR 5","Scheduled crypto changeover completed; no impact on current operations, recorded for the CCIR log."),("FFIR 6","Replacement personnel arrived as scheduled; strength reporting updated."),("PIR 5","No change in enemy posture in the northern sector; previous assessment stands."))}[p]
    t=f"CCIR update to {pick('Battalion TOC','Brigade main CP','TF CP')}: {e[0]} has been met. {e[1]} Reported by {unit()} at {dtg()}. Request commander be notified."
    return t,p,"FM 6-0","ch. on CCIR (PIR/FFIR)",["CCIR ID","TRIGGER","REPORTING UNIT","DTG"]
def sitrep():
    p=pick("Immediate","Priority","Routine","Flash")
    ov={"Flash":"Unit under heavy attack, combat ineffective (RED), request immediate reinforcement.","Immediate":"Unit at AMBER after contact; 2 vehicles NMC, ammo at 40%.",
        "Priority":"Unit GREEN; minor maintenance issues, screen line established.","Routine":"Unit GREEN; no enemy contact in last 24 hours, training and maintenance ongoing."}[p]
    t=(f"Commander's SITREP. Line 1: {dtg()}. Line 2: {unit()}. Line 5 PRESENT LOCATION: {grid()}. Line 6 ACTIVITY: {pick('defending BP 3','screen along PL RED','in assembly area','conducting area security')}. "
       f"Line 7 EFFECTIVE: {ov} Line 10 OPERATIONS: {pick('patrols conducted as planned','one patrol made contact','no change')}.")
    return t,p,"FM 6-99","A-60",["DTG","UNIT","REPORTED UNIT","HOME LOC","PRESENT LOC","ACTIVITY","EFFECTIVE","OWN SITUATION","OVERVIEW","OPERATIONS"]
def frago():
    p=pick("Flash","Immediate","Priority","Routine")
    m={"Flash":"Effective immediately, all elements occupy alternate battle positions; enemy attack imminent.","Immediate":"NLT 1800L, Co B assumes the screen on PL BLUE from Co C.",
       "Priority":"Change to tomorrow's convoy timeline: SP moves from 0600 to 0800.","Routine":"Change to sustainment annex: laundry and shower point relocates to the BSA next week."}[p]
    t=(f"FRAGORD {random.randint(1,40):03d} to OPORD {random.randint(1,12):02d}-{pick('25','26')}. Line 1: {dtg()}. Line 2: {unit()}. Line 4 TIME ZONE: {pick('Z','L')}. "
       f"Line 6 SITUATION: {pick('no change','enemy activity increasing to the east','weather degrading')}. Line 7 MISSION: {m} Line 11: ACKNOWLEDGE.")
    return t,p,"FM 6-99","A-92",["DTG","UNIT","FRAGORD NO","TIME ZONE","REFERENCES","SITUATION","MISSION","EXECUTION","SUSTAINMENT","C&S","ACK"]
def rfi():
    p=pick("Flash","Immediate","Priority","Routine")
    r={"Flash":("ONE","Is the friendly patrol at grid {g} still in position? Supporting fire mission is about to fire on that grid.","NOW"),
       "Immediate":("ONE","Request current enemy disposition near Objective {o} before we cross LD.","within 1 hour"),
       "Priority":("TWO","Request trafficability assessment for Route {o} for heavy vehicles.","within 24 hours"),
       "Routine":("THREE","Request updated imagery of the training area for next month's planning.","in 7 days")}[p]
    t=f"RFI. Line 1: {dtg()}. Line 2: {unit()}. Line 3 REQUEST: {r[1].format(g=grid(),o=pick('FALCON','ANVIL','BLUE','OAK'))} Line 4 PRIORITY: {r[0]}. Line 5 BY: {r[2]}."
    return t,p,"FM 6-99","A-155",["DTG","UNIT","REQUEST","PRIORITY","BY","LTIOV","NARRATIVE","AUTH"]
def warno():
    p=pick("Immediate","Priority","Routine")
    t=(f"WARNORD {random.randint(1,30):02d}. Reference: OPORD {random.randint(1,12):02d}-26. Time zone: {pick('Z','L')}. 1. SITUATION: {pick('enemy forces reinforcing positions near the river line','no significant change','enemy activity increasing in sector')}. "
       f"2. MISSION: {unit()} {pick('conducts an attack to seize Objective OAK','establishes a screen along PL RED','conducts a route clearance of MSR TAMPA')} {pick('NLT 0600 tomorrow','in 72 hours','next week')}. "
       f"3. EXECUTION: initial movement NLT {dtg()}; rehearsal at {pick('1400','1600','0900')}. 4. SUSTAINMENT: {pick('top off Class III','draw 2 DOS Class I','no change')}. ACKNOWLEDGE.")
    if "tomorrow" in t: p=pick("Immediate","Priority")
    elif "next week" in t: p="Routine"
    return t,p,"FM 5-0","Figure D-4, p. 218",["SITUATION","MISSION","EXECUTION","SUSTAINMENT","C&S","ACK"]
def admin():
    p=pick("Routine","Routine","Priority")
    t=pick(f"Admin: {unit()} PERSTAT submitted, {random.randint(80,140)} assigned / {random.randint(70,130)} present for duty, no changes.",
           f"Reminder: Monthly sensitive-items inventory due to S4 by {dtg()}.",
           f"S1 note: Award packets for {unit()} due Friday; use the updated form.",
           f"Admin: Range control requests updated roster for next week's qualification range.",
           f"S6: Scheduled network maintenance tonight 2200-2300L, expect brief email outage.",
           f"S4: Vehicle dispatch for {unit()} must be renewed before Monday.", f"S1: Leave requests for next month due to the 1SG.",
           f"Admin: Updated phone roster for {unit()} attached; verify your entry.", f"S4: Turn in expended brass from last week's range to the ammo point.")
    if p=="Priority": t=t+" Suspense is tomorrow; late submission holds up the higher HQ report."
    return t,p,"FM 6-99","general admin formats (PERSTAT etc.)",["FREE TEXT"]
def other():
    p=pick("Routine","Priority","Immediate","Flash")
    t={"Flash":pick(f"Serious incident: vehicle rollover at {grid()}, two personnel trapped, notifying CP now.",f"Fire in the ammunition holding area at {grid()}, evacuating personnel now."),
       "Immediate":pick(f"Weather advisory: high winds 45 kt expected within 2 hours, all aviation ops at {unit()} to be reviewed.",f"Civilian crowd of about 200 gathering at checkpoint {grid()}, growing, no weapons seen."),
       "Priority":"Lost sensitive item: one NVG reported missing from storage locker; search under way.",
       "Routine":pick("Chaplain services schedule for Sunday is posted at the CP.","Dining facility hours change starting Monday.","Weather outlook: clear skies expected for the next 72 hours.")}[p]
    return t,p,"FM 6-99","A-1 serious incident report; sensitive-items report",["FREE TEXT"]
BUILDERS={"SPOTREP":spotrep,"SALUTE":salute,"9-line MEDEVAC":medevac,"CCIR trigger":ccir,"SITREP":sitrep,"FRAGO/order":frago,"RFI":rfi,"routine admin":admin,"other":other}
