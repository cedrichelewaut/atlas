"""Recommendation wizard: books in the shared FlexViz view -> ~50 candidates -> 2-4 multiple-choice questions -> top 3.
The LLM (Ollama, strict JSON schema) writes questions + final picks; everything is validated and falls back to deterministic logic."""
import os, re, json, time, uuid, math, asyncio
import polars as pl
from . import agent

HERE=os.path.dirname(os.path.abspath(__file__))
from . import store
EXTRA=store.EXTRA          # i, tags, pages, desc (360 chars)
MAXC, MINQ, MAXQ, FINAL_MAX = 50, 2, 4, 5
SESS={}

def _extra(ids):
    if not os.path.exists(EXTRA): return {}
    return {r['i']:r for r in pl.scan_parquet(EXTRA).filter(pl.col('i').is_in(list(ids))).collect().iter_rows(named=True)}

STOP=set('a an the and or of for to in on with about like some something book books novel novels read reading want looking find me i im i\'m that is are be good great best any my please would love more new not but by from it its this'.split())
def _terms(q): return [w for w in re.findall(r"[a-z][a-z'-]{2,}",(q or '').lower()) if w not in STOP][:8]
def candidates(B,expr,n=MAXC,query=''):
    v=B.filter(expr); total=v.height
    if not total: return [],0
    C=float(v['rating'].drop_nulls().mean() or 3.9); m=300.0
    v=v.with_columns(((pl.col('rating').fill_null(C)*pl.col('ratings_count').fill_null(0)+C*m)/(pl.col('ratings_count').fill_null(0)+m)).alias('score'))
    v=v.sort(['score','ratings_count'],descending=True).unique(subset=['title'],keep='first',maintain_order=True)
    T=_terms(query)
    if T:   # typed request: among the best-scored 4000 in view, prefer books whose title/shelves/blurb mention the words
        pool=v.head(4000); ex=pl.scan_parquet(EXTRA).filter(pl.col('i').is_in(pool['i'].to_list())).select('i','tags','desc').collect() if os.path.exists(EXTRA) else None
        if ex is not None:
            pool=pool.join(ex,on='i',how='left').with_columns(pl.concat_str([pl.col('title'),pl.col('tags').fill_null(''),pl.col('desc').fill_null('')],separator=' ').str.to_lowercase().alias('_t'))
            pool=pool.with_columns(pl.sum_horizontal([pl.col('_t').str.contains(t,literal=True).cast(pl.Int32) for t in T]).alias('_m'))
            hit=pool.filter(pl.col('_m')>0).sort(['_m','score'],descending=True).head(n)
            v=pl.concat([hit.select(v.columns),v.filter(~pl.col('i').is_in(hit['i'].to_list())).head(n-hit.height)]) if hit.height<n else hit.select(v.columns)
    v=v.head(n)
    ex=_extra(v['i'].to_list()); out=[]
    for k,r in enumerate(v.iter_rows(named=True)):
        e=ex.get(r['i'],{})
        out.append({'n':k+1,'i':r['i'],'title':r['title'],'author':r['author'],'year':r['year'],'rating':None if r['rating'] is None else round(float(r['rating']),2),
                    'ratings_count':int(r['ratings_count'] or 0),'genre':r['genre'],'cluster':r['cluster'],'pages':e.get('pages'),'tags':e.get('tags') or '',
                    'desc':(e.get('desc') or '').strip(),'score':round(float(r['score']),3)})
    return out,total

def _profile():
    p=agent.load_profile(); txt=p.get('profile_md') or ''
    lines=[l for l in txt.splitlines() if l.strip().startswith('-') and '(fill in)' not in l]
    g=p.get('goodreads_export'); fav=''
    if isinstance(g,dict) and g.get('top_rated'): fav='Books they rated highly: '+'; '.join(f"{b['title']} ({b.get('author')})" for b in g['top_rated'][:12])
    s='\n'.join(lines+([fav] if fav else [])).strip()
    return s or 'No profile filled in yet (treat as an open-minded fiction reader).'

def _line(b,full=False):
    s=f"#{b['n']} {b['title'][:70]} | {b['author'] or '?'} | {b['year'] or '?'} | {b['rating'] or '?'}★ {b['ratings_count']:,} ratings | {b['pages'] or '?'} pages | {b['genre'] or ''} | {b['tags'][:70]}"
    return s+' | '+(b['desc'][:330] if full else b['desc'][:75])

# ---------- LLM (Ollama structured outputs) ----------
QSCHEMA={'type':'object','properties':{'question':{'type':'string'},'options':{'type':'array','minItems':2,'maxItems':4,'items':{'type':'object','properties':{'label':{'type':'string'},'books':{'type':'array','items':{'type':'integer'}}},'required':['label','books']}}},'required':['question','options']}
FSCHEMA={'type':'object','properties':{'top':{'type':'array','minItems':3,'maxItems':3,'items':{'type':'object','properties':{'book':{'type':'integer'},'summary':{'type':'string'},'reason':{'type':'string'}},'required':['book','summary','reason']}}},'required':['top']}
QSYS=("You help a reader pick a book from a numbered candidate list by asking ONE multiple-choice question at a time. "
      "Ask about taste (mood, tone, theme, setting, pace, length, era, popularity), never about a specific title. If there is a READER REQUEST, ask what it leaves open. "
      "Keep the question under 18 words, plain text. Give 2-4 short options (max 6 words each). Every option lists the numbers (#) of the REMAINING books that fit it; each remaining book should fit at least one option, "
      "each option must have at least 2 books and fewer than all remaining books. Do not repeat earlier questions. Reply with JSON only: "
      '{"question": "...", "options": [{"label": "...", "books": [1, 4, 9]}]}')
FSYS=("You recommend books. From the FINALISTS pick the best 3 for this reader, best first, using their answers and profile. "
      "For each: book = its # number, summary = one sentence on what the book is (from its description, no spoilers), "
      "reason = one sentence why it fits THIS reader's answers. Only use facts from the list. Reply with JSON only: "
      '{"top": [{"book": 3, "summary": "...", "reason": "..."}]}')

async def _llm(system,prefix,ask,schema,timeout_note=''):
    if not agent.ollama_ok(): raise RuntimeError('ollama unavailable')
    body={'model':agent.REC_MODEL(),'stream':False,'keep_alive':'30m','format':schema,'options':{'temperature':0.2,'num_ctx':8192,'num_predict':500},
          'messages':[{'role':'system','content':system},{'role':'user','content':prefix},{'role':'assistant','content':'Understood. I have the list.'},{'role':'user','content':ask}]}
    r=await agent.post(agent.OLLAMA+'/api/chat',{},body); txt=(r.get('message') or {}).get('content') or ''
    m=re.search(r'\{.*\}',txt,re.S)
    return json.loads(m.group(0) if m else txt)

def _clean(s,n): return re.sub(r'\s+',' ',re.sub(r'[*`#]+','',str(s or ''))).strip()[:n]

def _valid_q(q,remaining,asked):
    if not isinstance(q,dict): return None
    rem=set(remaining); qt=_clean(q.get('question'),200); opts=[]
    if len(qt)<8 or qt.lower() in asked: return None
    for o in q.get('options') or []:
        if not isinstance(o,dict): continue
        lab=_clean(o.get('label'),60); bs=[]
        for x in o.get('books') or []:
            try: x=int(x)
            except Exception: continue
            if x in rem and x not in bs: bs.append(x)
        if lab and len(bs)>=min(2,len(rem)) and len(bs)<len(rem) and lab.lower() not in [p['label'].lower() for p in opts] and bs not in [p['books'] for p in opts]: opts.append({'label':lab,'books':bs})
    if not 2<=len(opts)<=4: return None
    cov=set().union(*[set(o['books']) for o in opts])
    if len(cov)<0.6*len(rem): return None          # options must cover most of the remaining books
    return {'question':qt,'options':opts,'source':'llm'}

# ---------- deterministic questions ----------
def _split(name,books,q,rules):
    opts=[{'label':lab,'books':[b['n'] for b in books if f(b)]} for lab,f in rules]
    opts=[o for o in opts if o['books']]
    if len(opts)<2: return None
    bal=min(len(o['books']) for o in opts)/max(1,len(books))
    return {'key':name,'question':q,'options':opts,'source':'fallback','balance':bal}
def _fallbacks(books,asked_keys):
    out=[]; ys=sorted(b['year'] or 0 for b in books); cs=sorted(b['ratings_count'] for b in books)
    pg=[b for b in books if b['pages']]
    if len(pg)>=0.7*len(books):
        out.append(_split('length',books,'How long a read do you want?',[('Short (under 300 pages)',lambda b:(b['pages'] or 0)<300 and b['pages']),('Medium (300-450 pages)',lambda b:300<=(b['pages'] or 0)<=450),('Long (over 450 pages)',lambda b:(b['pages'] or 0)>450),("Doesn't matter",lambda b:True)]))
    ym=ys[len(ys)//2]
    out.append(_split('year',books,'Newer or older books?',[(f'Before {ym}',lambda b:(b['year'] or 0)<ym),(f'{ym} or later',lambda b:(b['year'] or 0)>=ym),("Doesn't matter",lambda b:True)]))
    cm=cs[len(cs)//2]
    out.append(_split('pop',books,'Crowd favourite or hidden gem?',[('Crowd favourite (widely read)',lambda b:b['ratings_count']>=cm),('Hidden gem (fewer ratings)',lambda b:b['ratings_count']<cm),("Doesn't matter",lambda b:True)]))
    rs=sorted((b['rating'] or 0) for b in books); rm=rs[len(rs)//2]
    out.append(_split('rating',books,'How much do ratings matter to you?',[(f'Only the top rated ({rm:.2f}★+)',lambda b:(b['rating'] or 0)>=rm),("I'm open to mixed reviews",lambda b:True)]))
    gs={}
    for b in books: gs[b['genre'] or 'other']=gs.get(b['genre'] or 'other',0)+1
    top=[g for g,c in sorted(gs.items(),key=lambda t:-t[1]) if c>=2][:3]
    if len(top)>=2: out.append(_split('genre',books,'Which kind of story?',[(g.title(),(lambda g: lambda b:(b['genre'] or 'other')==g)(g)) for g in top]+([('Something else',lambda b,t=set(top):(b['genre'] or 'other') not in t)] if sum(gs[g] for g in top)<len(books) else [])))
    out=[o for o in out if o and o['key'] not in asked_keys and max(len(x['books']) for x in o['options'] if not x['label'].startswith(("Doesn't","I'm open")))<len(books)]
    pri=['genre','length','year','pop','rating']
    return sorted(out,key=lambda o:(o['balance']<0.2,pri.index(o['key'])))

# ---------- session flow ----------
def _pub(S,q=None,final=None):
    rem=S['remaining']
    out={'session':S['id'],'round':len(S['answers'])+(1 if q else 0),'max_rounds':MAXQ,'candidates':len(S['books']),'in_view':S['in_view'],'remaining':len(rem),
         'timings':S['timings'],'model':S['model']}
    if q: out['question']={'text':q['question'],'source':q['source'],'options':[{'id':k,'label':o['label'],'count':(min(len(rem),max(FINAL_MAX,math.ceil(len(rem)*0.6))) if (o.get('any') or o['label'].startswith(("Doesn't","I'm open"))) and len(o['books'])==len(rem) else len(o['books']))} for k,o in enumerate(q['options'])]}
    if final: out['final']=final
    return out

def _ok(S): return agent.ollama_ok() and not S.get('no_llm')
async def _next_question(S):
    books={b['n']:b for b in S['books']}; rem=S['remaining']; t=time.time(); q=None; err=None
    if _ok(S):
        ask=(f"Questions already asked: {json.dumps([a['question'] for a in S['answers']]) or 'none'}. Reader's answers so far: {json.dumps([a['label'] for a in S['answers']])}.\n"
             f"REMAINING books ({len(rem)}): {rem}\nWrite question {len(S['answers'])+1}: split the REMAINING books by a NEW aspect of taste.")
        for attempt in range(2):
            try:
                q=_valid_q(await _llm(QSYS,S['prefix'],ask+('' if not attempt else '\nYour last reply was invalid: use only REMAINING numbers, 2-4 options, each with 2+ books.'),QSCHEMA),rem,[a['question'].lower() for a in S['answers']])
            except Exception as e: err=str(e)[:120]
            if q: break
        S['llm_attempts']=S.get('llm_attempts',0)+attempt+1
        if not q and err and 'unavailable' in err: S['no_llm']=True
    if not q:
        fb=_fallbacks([books[n] for n in rem],{a.get('key') for a in S['answers']})
        q=fb[0] if fb else None
    S['timings'].append({'step':f"question {len(S['answers'])+1}",'secs':round(time.time()-t,1),'source':q['source'] if q else 'none'})
    if q and q['source']=='llm' and len(q['options'])<4:
        q['options'].append({'label':'No preference','books':list(rem),'any':True})
    S['q']=q; return q

def _det_final(S,fin):
    out=[]
    picks=[a['label'] for a in S['answers'] if a['label']!='No preference']
    for b in fin[:3]:
        d=b['desc']; summ=(re.split(r'(?<=[.!?])\s',d)[0] if d else f"A {b['genre'] or ''} novel from {b['year']}.")[:220]
        why=(f"Fits your picks ({'; '.join(picks)}). " if picks else '')+f"Rated {b['rating']}★ by {b['ratings_count']:,} readers."
        out.append({'book':b['n'],'summary':summ,'reason':why})
    return out

async def _final(S):
    books={b['n']:b for b in S['books']}; rem=S['remaining']
    fin=sorted([books[n] for n in rem],key=lambda b:-b['score'])[:FINAL_MAX]; S['finalists']=[b['n'] for b in fin]; t=time.time(); top=None; src='fallback'
    if _ok(S):
        ctx=((f"READER REQUEST: {S['query']}\n\n" if S.get('query') else '')+"READER PROFILE:\n"+_profile()+"\n\nREADER'S ANSWERS:\n"+'\n'.join(f"- {a['question']} -> {a['label']}" for a in S['answers'])+
             "\n\nFINALISTS:\n"+'\n'.join(_line(b,True) for b in fin))
        for attempt in range(2):
            try:
                r=await _llm(FSYS,ctx,'Pick the top 3 now (3 different # numbers from the FINALISTS).',FSCHEMA); seen=[]; top=[]
                for x in r.get('top') or []:
                    try: n=int(x.get('book'))
                    except Exception: continue
                    if n in S['finalists'] and n not in seen and _clean(x.get('summary'),300) and _clean(x.get('reason'),300):
                        seen.append(n); top.append({'book':n,'summary':_clean(x['summary'],300),'reason':_clean(x['reason'],300)})
                if len(top)>=3: top=top[:3]; src='llm'; break
                if top and len(fin)>=3:      # pad a partial answer deterministically
                    pad=[p for p in _det_final(S,[b for b in fin if b['n'] not in seen]+fin)][:3-len(top)]
                    if attempt: top+=pad; src='llm+fallback'; break
                top=None
            except Exception: top=None
    if not top: top=_det_final(S,fin)
    S['timings'].append({'step':'top 3','secs':round(time.time()-t,1),'source':src})
    res=[]
    for k,x in enumerate(top):
        b=books[x['book']]; res.append({'rank':k+1,'i':b['i'],'title':b['title'],'author':b['author'],'year':b['year'],'rating':b['rating'],'ratings_count':b['ratings_count'],
            'pages':b['pages'],'genre':b['genre'],'summary':x['summary'],'reason':x['reason']})
    S['done']=True
    return {'top3':res,'source':src,'finalists':[{'title':books[n]['title'],'author':books[n]['author']} for n in S['finalists']],'answers':[{'q':a['question'],'a':a['label']} for a in S['answers']],'query':S.get('query','')}

async def start(B,expr,view_desc='',query=''):
    books,total=candidates(B,expr,query=query)
    if len(books)<3: return {'error':f'Only {total} book(s) in view — zoom out or loosen the filters to get a recommendation.'}
    sid=uuid.uuid4().hex[:10]
    S={'id':sid,'query':_clean(query,300),'books':books,'in_view':total,'remaining':[b['n'] for b in books],'answers':[],'timings':[],'view':view_desc,'t0':time.time(),
       'model':agent.REC_MODEL() if agent.ollama_ok() else 'deterministic',
       'prefix':(f"READER REQUEST: {_clean(query,300)}\n" if query else '')+'CANDIDATES (# | title | author | year | rating | ratings | pages | genre | shelves | blurb):\n'+'\n'.join(_line(b) for b in books)}
    SESS[sid]=S
    for k in [k for k,v in SESS.items() if time.time()-v['t0']>3600]: SESS.pop(k,None)
    if len(books)<=FINAL_MAX: return _pub(S,final=await _final(S))
    return _pub(S,q=await _next_question(S))

async def answer(sid,option):
    S=SESS.get(sid)
    if not S: return {'error':'unknown or expired session; start again'}
    if S.get('done') or not S.get('q'): return {'error':'session already finished'}
    q=S['q']; opts=q['options']
    if isinstance(option,str) and not option.isdigit():
        m=[k for k,o in enumerate(opts) if o['label'].lower()==option.lower()]; option=m[0] if m else -1
    option=int(option)
    if not 0<=option<len(opts): return {'error':f'option must be 0..{len(opts)-1}','question':q['question'],'options':[o['label'] for o in opts]}
    o=opts[option]; prev=S['remaining']; keep=[n for n in prev if n in set(o['books'])]
    if len(keep)<3:                           # never narrow below 3: pad with best-scored remaining
        sc={b['n']:b['score'] for b in S['books']}; keep+= [n for n in sorted(prev,key=lambda n:-sc[n]) if n not in keep][:3-len(keep)]
    if o.get('any') or o['label'].startswith(("Doesn't","I'm open")) and len(keep)==len(prev):
        sc={b['n']:b['score'] for b in S['books']}; keep=sorted(prev,key=lambda n:-sc[n])[:max(FINAL_MAX,math.ceil(len(prev)*0.6))]  # still narrow: keep best-scored
    S['answers'].append({'question':q['question'],'label':o['label'],'key':q.get('key'),'kept':len(keep)}); S['remaining']=keep; S['q']=None
    r=len(S['answers'])
    if (r>=MINQ and len(keep)<=FINAL_MAX) or r>=MAXQ: return _pub(S,final=await _final(S))
    q=await _next_question(S)
    if not q: return _pub(S,final=await _final(S))
    return _pub(S,q=q)
