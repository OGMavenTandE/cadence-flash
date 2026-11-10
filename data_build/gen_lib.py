import json, os, re, time, urllib.error, urllib.request
from urllib.parse import urlparse
KEY=os.environ["GEMINI_API_KEY"]
if any(c in KEY for c in "\r\n\x00"): raise RuntimeError("invalid GEMINI_API_KEY")
_HOST="generativelanguage.googleapis.com"
_MODEL=re.compile(r"[A-Za-z0-9._-]{1,64}")
class _PinHost(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u=urlparse(newurl)
        if u.scheme!="https" or (u.hostname or "").lower()!=_HOST:
            raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)
        return urllib.request.HTTPRedirectHandler.redirect_request(self, req, fp, code, msg, headers, newurl)
_opener=urllib.request.build_opener(_PinHost)
def gem(model,prompt,tries=3):
    if not _MODEL.fullmatch(model): raise ValueError("invalid model id")
    url=f"https://{_HOST}/v1beta/models/{model}:generateContent"
    body=json.dumps({"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"responseMimeType":"application/json","temperature":0.7 if 'flash' in model else 0,"thinkingConfig":{"thinkingLevel":"low" if 'flash' in model else "high"}}}).encode()
    err=None
    for i in range(tries):
        try:
            req=urllib.request.Request(url,body,{"Content-Type":"application/json","x-goog-api-key":KEY})
            with _opener.open(req,timeout=180) as resp: r=json.load(resp)
            u=r.get("usageMetadata",{}); return r["candidates"][0]["content"]["parts"][0]["text"],u
        except Exception as e:
            time.sleep(5*(i+1))
            err=RuntimeError(str(e).replace(KEY,"[redacted]")) if KEY and KEY in str(e) else e
    raise err
