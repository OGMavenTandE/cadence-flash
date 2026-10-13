import random,json,os,re,time,urllib.request,concurrent.futures as cf
from templates import BUILDERS,SRC
KEY=os.environ["GEMINI_API_KEY"]
def gem(model,prompt,tries=3):
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}"
    body=json.dumps({"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"responseMimeType":"application/json","temperature":0.7 if 'flash' in model else 0,"thinkingConfig":{"thinkingLevel":"low" if 'flash' in model else "high"}}}).encode()
    for i in range(tries):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,body,{"Content-Type":"application/json"}),timeout=180))
            u=r.get("usageMetadata",{}); return r["candidates"][0]["content"]["parts"][0]["text"],u
        except Exception as e: time.sleep(5*(i+1)); err=e
    raise err
