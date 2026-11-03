"""Atlas web app: FlexViz is the whole visualisation; its client state is the shared surface for a human and Atlas AI.
Run via the CLI:  atlas serve   (or: ATLAS_DATA=<dataset dir> uvicorn atlas_books.app:app --port 8010)
"""
import os, re, json, time, asyncio, threading
import numpy as np, polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, Response
from flexviz import Dashboard, LayoutSpec, ToolbarConfig, GridItem, mount_into
from flexviz.server import register_source
from . import store, paths
from . import agent, rec, librarian

HERE=str(paths.WEB); ROOT=HERE
SRC='books'; SOURCE='fv_books'
DATA=store.DATA
B=store.books()                                                        # i,title,author,year,x,y,cluster_label,ratings_count,cluster,genre,rating,url
B=B.with_columns((pl.col('ratings_count').fill_null(0)+1).log10().cast(pl.Float32).alias('log10_ratings'))   # log10(ratings+1) for the ratings-count chart
B=B.join(pl.read_parquet(f'{DATA}/extra.parquet',columns=['i','pages']),on='i',how='left',suffix='_x').sort('i')
if 'pages_x' in B.columns: B=B.with_columns(pl.coalesce('pages','pages_x').cast(pl.Int32).alias('pages')).drop('pages_x')
HAS_RATING=B['rating'].is_not_null().any()
FV=B.select('x','y','year','rating','log10_ratings','pages')
register_source(SOURCE,FV.lazy(),cache=True)
LOWER=B['title'].str.to_lowercase()
URLS=B['url'].to_list()
CORPUS=store.describe(B.height)
_yr=B['year'].drop_nulls()
if _yr.len(): librarian.YMIN,librarian.YMAX=int(_yr.quantile(.01)),int(_yr.max())
agent.SYSTEM=agent.SYSTEM.replace('~482k English-language Goodreads fiction books (1980+)',CORPUS)
def gr_url(i):
    if not 0<=int(i)<len(URLS): return None
    return URLS[int(i)]
_GL={}
def mention_links(text,limit=20):
    '''Map *Title* / **Title** spans in a reply to book URLs (exact title match, most-rated edition).'''
    out={}
    for t in re.findall(r'\*\*([^*\n]{2,140})\*\*|\*([^*\n]{2,140})\*',text or ''):
        t=(t[0] or t[1]).strip()
        if not t or t in out or len(out)>=limit: continue
        hit=B.filter(LOWER==t.lower())
        if hit.height==0:
            if 'ns' not in _GL: _GL['ns']=LOWER.str.replace(r'\s*\([^)]*#[^)]*\)$','')   # title without '(Series, #n)'
            hit=B.filter(_GL['ns']==t.lower())
        if hit.height: out[t]=gr_url(hit.sort('ratings_count',descending=True)['i'][0])
    return {k:v for k,v in out.items() if v}
X0,X1=[float(v) for v in np.percentile(B['x'].to_numpy(),[0.3,99.7])]; Y0,Y1=[float(v) for v in np.percentile(B['y'].to_numpy(),[0.3,99.7])]

# ---------- FlexViz dashboard (the only visualisation) ----------
FONT='Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
INK=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'family':FONT,'color':'#4a5162','size':11},
         hoverlabel={'bgcolor':'#ffffff','bordercolor':'rgba(15,23,42,.15)','font':{'family':FONT,'color':'#141824'}},
         xaxis={'gridcolor':'rgba(15,23,42,0.06)','zerolinecolor':'rgba(15,23,42,0.10)','linecolor':'rgba(15,23,42,0.15)','tickfont':{'color':'#6b7280'},'title':{'font':{'color':'#6b7280'}}},
         yaxis={'gridcolor':'rgba(15,23,42,0.06)','zerolinecolor':'rgba(15,23,42,0.10)','linecolor':'rgba(15,23,42,0.15)','tickfont':{'color':'#6b7280'},'title':{'font':{'color':'#6b7280'}}})
def th(**kw):
    out=json.loads(json.dumps(INK))
    for k,v in kw.items(): out[k]=(out.get(k,{})|v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out
d=Dashboard(FV.lazy(),cache=True)
TITLE={'font':{'family':FONT,'color':'#141824','size':13},'x':0.02,'xanchor':'left'}
def fig(t,**kw): return d.add_figure(title=t,**th(**kw))
fig('Description map  ·  books with similar blurbs sit together  ·  hover for details',margin={'l':10,'r':6,'t':34,'b':10},xaxis={'title':{'text':''},'showticklabels':False,'showgrid':False,'zeroline':False,'showline':False,'ticks':''},yaxis={'title':{'text':''},'showticklabels':False,'showgrid':False,'zeroline':False,'showline':False,'ticks':''}).add_histogram2d(x='x',y='y',x_bins=80,y_bins=80,color_norm='log',color_scale='Viridis')
fig('First published',margin={'l':44,'t':34,'b':28,'r':10},xaxis={'title':{'text':''}},yaxis={'title':{'text':''}}).add_histogram(x='year',bins=40,color='#14b8a6')
fig('Average rating',margin={'l':44,'t':34,'b':28,'r':10},xaxis={'title':{'text':''}},yaxis={'title':{'text':''}}).add_histogram(x='rating',bins=30,color='#f59e0b')
fig('Pages',margin={'l':44,'t':34,'b':28,'r':10},xaxis={'title':{'text':''}},yaxis={'title':{'text':''}}).add_histogram(x='pages',bins=40,color='#64748b')
fig('Number of ratings (log scale)',margin={'l':44,'t':34,'b':28,'r':10},xaxis={'title':{'text':''},'tickmode':'array','tickvals':[0,1,2,3,4,5,6,7],'ticktext':['1','10','100','1k','10k','100k','1M','10M']},yaxis={'title':{'text':''}}).add_histogram(x='log10_ratings',bins=36,color='#0ea5e9')
U=[f.uid for f in d.to_spec(source_name=SOURCE).figures]; FIG=dict(zip(['map','year','rating','pages','votes'],U))
def dash_url(px_h=None):
    # only the map is laid out; the year/rating/pages/ratings histograms stay in the spec (hidden) as the selection sources
    # for the page-side range sliders, so every filter lives in the one FlexViz state
    rows=max(4,min(14,int(((px_h or 600)-40)//80)))
    lay=LayoutSpec(draggable=False,grid_items=[GridItem(fig_uid=FIG['map'],x=0,y=0,w=12,h=rows)],
       toolbar=ToolbarConfig(show_share=False,show_export=False,show_import=False,show_grid=False,show_lock_all_axes=False,show_cfmode=False))
    return d.share_url(server_url='/flexviz',source_name=SOURCE,cache=True,layout=lay)
DASH_URL=dash_url()
COLFIG={'x':FIG['map'],'y':FIG['map'],'year':FIG['year'],'rating':FIG['rating'],'log10_ratings':FIG['votes'],'pages':FIG['pages']}
FILTERS={'year':'year','rating':'rating','ratings_count':'log10_ratings','pages':'pages'}     # slider name -> state column
def filters_of(state):
    """The slider view of the shared state: {year:[a,b]|None, rating, ratings_count, pages, region}."""
    out={k:None for k in FILTERS}|{'region':None}
    for sel in state.get('selections') or []:
        for p in sel.get('predicates',[]):
            cl={c['column']:c for c in p.get('clauses',[]) if c.get('range') is not None}
            if 'x' in cl and 'y' in cl: out['region']=[*cl['x']['range'],*cl['y']['range']]; continue
            for k,col in FILTERS.items():
                if col in cl:
                    r=[float(v) for v in cl[col]['range']]
                    out[k]=[round(10**r[0]-1),round(10**r[1]-1)] if k=='ratings_count' else [round(v,2) for v in r]
    return out

# ---------- shared state: mirror of the browser's FlexViz state ----------
INIT={'viewport':{f"{FIG['map']}/x":{'min':X0,'max':X1},f"{FIG['map']}/y":{'min':Y0,'max':Y1}},'selections':[]}
M={'state':json.loads(json.dumps(INIT)),'client_state':{},'fv_revision':0,'rev':0,'by':'init','at':time.time(),'browser_seen':0.0}
LOG=[]; CLIENTS=set(); PATCH_ID=[0]; ACK={}; FOCUS={'book':None}
def _log(by,what): LOG.append({'t':time.strftime('%H:%M:%S'),'by':by,'what':what}); del LOG[:-40]
def describe(state):
    out=[]
    for k,v in (state.get('viewport') or {}).items():
        if v: out.append(f"zoom {k.split('/')[1]}=[{v['min']:.2f},{v['max']:.2f}]" if k.startswith(FIG['map']) else f"zoom {k}")
    for s in state.get('selections') or []:
        for p in s.get('predicates',[]):
            for c in p.get('clauses',[]):
                if c['column']=='log10_ratings' and c.get('range'): out.append(f"select ratings received∈[{max(0,round(10**float(c['range'][0])-1)):,}–{round(10**float(c['range'][1])-1):,}]"); continue
                if c.get('range'):
                    lo,hi=(float(x) for x in c['range']); f=lambda v:f"{round(v,2):g}"
                    out.append(f"select {c['column']}∈[{f(lo)}–{f(hi)}]" if hi<1e5 else f"select {c['column']}≥{f(lo)}"); continue
                out.append(f"select {c['column']}∈{c.get('values')}")
    return '; '.join(out) or 'full map, no selection'

def filt(state,ignore_viewport=False):
    e=pl.lit(True); vp=state.get('viewport') or {}
    if not ignore_viewport:
        for ax in 'xy':
            r=vp.get(f"{FIG['map']}/{ax}")
            if r: e=e&pl.col(ax).is_between(r['min'],r['max'])
    for s in state.get('selections') or []:
        pe=pl.lit(False)
        for p in s.get('predicates',[]):
            ce=pl.lit(True)
            for c in p.get('clauses',[]):
                col=c['column']
                if c.get('values') is not None: ce=ce&pl.col(col).cast(pl.Utf8).is_in([str(v) for v in c['values']])
                elif c.get('range') is not None: ce=ce&pl.col(col).is_between(float(c['range'][0]),float(c['range'][1]))
            pe=pe|ce
        e=e&pe
    return e
def bookrow(r): return {'i':r['i'],'title':r['title'],'author':r['author'],'year':r['year'],'rating':None if r['rating'] is None else round(r['rating'],2),'ratings_count':r['ratings_count'],'pages':r.get('pages'),'x':round(r['x'],6),'y':round(r['y'],6),'url':gr_url(r['i'])}   # full precision: the focus ring must sit exactly on the dot at any zoom
def books_in_view(state,limit=20,offset=0):
    v=B.filter(filt(state)); top=v.sort('ratings_count',descending=True).slice(offset,limit)
    return v.height,[bookrow(r) for r in top.iter_rows(named=True)]
def summary(state,limit=15):
    v=B.filter(filt(state)); n=v.height
    mix=lambda c,k:[{'label':a,'n':b,'pct':round(100*b/max(n,1),1)} for a,b in v[c].value_counts().sort('count',descending=True).head(k).iter_rows()]
    return {'books_in_view':n,'of_total':B.height,'description':describe(state),
            'top_books':[bookrow(r) for r in v.sort('ratings_count',descending=True).head(limit).iter_rows(named=True)],
            'filters':filters_of(state),
            'years':None if n==0 else {'median':v['year'].median(),'p10':v['year'].quantile(.1),'p90':v['year'].quantile(.9)},
            'avg_rating':None if n==0 else (round(v['rating'].mean(),2) if v['rating'].mean() is not None else None),
            'focus':FOCUS['book'],'recommended':[f"{r['rank']}. {r['title']}" for r in HIGHLIGHT['recs']]}
def view_state():
    return {'figures':FIG,'state':M['state'],'fv_revision':M['fv_revision'],'rev':M['rev'],'last_change_by':M['by'],
            'browser_connected':len(CLIENTS)>0,'recent_changes':LOG[-8:],'summary':summary(M['state'],10),'highlights':HIGHLIGHT}

LOOP={}
async def _broadcast(msg):
    for q in list(CLIENTS): q.put_nowait(msg)
async def set_state(patch,by='agent',what=None,wait=4.0):
    """Merge a state patch one level deep (FlexViz semantics), push to the browser, wait for it to apply."""
    for k,v in patch.items(): M['state'][k]=v
    M['rev']+=1; M['by']=by; M['at']=time.time(); _log(by,what or describe(patch))
    PATCH_ID[0]+=1; pid=PATCH_ID[0]; applied=False
    if CLIENTS:
        ev=asyncio.Event(); ACK[pid]=ev; await _broadcast({'type':'apply','id':pid,'patch':{'state':patch},'by':by})
        try: await asyncio.wait_for(ev.wait(),wait); applied=True
        except asyncio.TimeoutError: pass
        ACK.pop(pid,None)
    return {'applied_in_browser':applied,'browser_connected':bool(CLIENTS),'state':M['state'],'summary':summary(M['state'],10)}

# ---------- embeddings (lazy) ----------
EMB={}; ELK=threading.Lock()
def _vecs():
    """fp16 vectors if present, else int8 + per-dimension scale (portable package)."""
    n,get=store.vec_getter(); return range(n),get
def emb():
    with ELK:
        if not EMB:
            V,get=_vecs()
            try:
                import faiss
                ix=faiss.IndexScalarQuantizer(get(0,1).shape[1],faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT)
                for a in range(0,len(V),100000): ix.add(get(a,a+100000))
            except ImportError: ix=None          # numpy fallback below (no faiss wheel for this Python)
            EMB.update(V=V,get=get,n=len(V),ix=ix)
    return EMB
def _search(E,q,k):
    if E['ix'] is not None: D,I=E['ix'].search(q,k); return D[0],I[0]
    best=[]
    for a in range(0,E['n'],100000):
        s=E['get'](a,a+100000)@q[0]; j=np.argpartition(-s,min(k,len(s)-1))[:k]; best+=list(zip(s[j],j+a))
    best.sort(key=lambda t:-t[0]); best=best[:k]; return np.array([b[0] for b in best]),np.array([b[1] for b in best])
def find(title=None,i=None):
    if i is not None: return int(i)
    h=search(title,1)
    if not h: raise HTTPException(404,f'no book matching {title!r}')
    return h[0]['i']
def search(q,limit=8):
    ql=(q or '').lower().strip()
    if not ql: return []
    h=(B.with_columns(_l=LOWER).filter(pl.col('_l').str.contains(ql,literal=True)).with_columns(_p=pl.col('_l').str.starts_with(ql),_n=pl.col('title').str.len_chars())
         .sort(['_p','ratings_count','_n'],descending=[True,True,False]).head(limit))
    return [bookrow(r) for r in h.iter_rows(named=True)]
def similar(i,k=10):
    E=emb(); D,I=_search(E,E['get'](i,i+1),k+1)
    return [bookrow(B.row(int(j),named=True))|{'similarity':round(float(s),3)} for j,s in zip(I,D) if j!=i and j>=0][:k]

# ---------- tools (shared by the in-page agent and external agents) ----------
def _rng(v): return [float(v[0]),float(v[1])]
def _sel(col,clause): return {'source_figure_uid':COLFIG[col],'predicates':[{'clauses':[clause]}]}
async def t_get_view_state(): return view_state()
async def t_list_books_in_view(limit=20,offset=0):
    n,rows=books_in_view(M['state'],min(int(limit),100),int(offset)); return {'books_in_view':n,'books':rows,'sorted_by':'ratings_count desc'}
async def t_search_book(title,limit=5): return {'matches':search(title,min(int(limit),20))}
async def t_set_view(x_min=None,x_max=None,y_min=None,y_max=None,reset=False):
    if reset: vp=dict(INIT['viewport']); FOCUS['book']=None; await set_recs([],by='reset',frame=False)
    else:
        if None in (x_min,x_max,y_min,y_max): raise HTTPException(400,'set_view needs x_min, x_max, y_min, y_max (or reset=true)')
        x_min,x_max=sorted([float(x_min),float(x_max)]); y_min,y_max=sorted([float(y_min),float(y_max)])
        if x_max-x_min<1e-3 or y_max-y_min<1e-3: raise HTTPException(400,'empty rectangle')
        vp={f"{FIG['map']}/x":{'min':x_min,'max':x_max},f"{FIG['map']}/y":{'min':y_min,'max':y_max}}; FOCUS['book']=None
    out=await set_state({'viewport':vp},what='reset map zoom' if reset else None)
    if FOCUS['book'] is None: await _broadcast({'type':'focus','book':None})
    return out
async def t_set_filters(year=None,rating=None,ratings_count=None,pages=None,region=None,clear=False):
    """Set slider ranges in the shared FlexViz state. Unspecified filters are kept; [] removes one; clear=true removes all."""
    if not HAS_RATING: rating=None          # dataset without ratings: a rating filter would hide every book
    cur={} if clear else {k:v for k,v in filters_of(M['state']).items() if v}
    for k,v in (('year',year),('rating',rating),('ratings_count',ratings_count),('pages',pages),('region',region)):
        if v is None: continue
        if not isinstance(v,(list,tuple)): raise HTTPException(400,f'{k} must be a list [min,max]')
        if len(v)==0: cur.pop(k,None); continue
        if len(v)!=(4 if k=='region' else 2): raise HTTPException(400,f'{k} must be [min,max]' if k!='region' else 'region must be [x_min,x_max,y_min,y_max]')
        try: v=[float(x) for x in v]
        except (TypeError,ValueError): raise HTTPException(400,f'{k} values must be numbers')
        if k!='region': v=sorted(v)
        lim={'year':(0,2100),'rating':(1,5),'ratings_count':(0,1e7),'pages':(0,1e5)}.get(k)
        if lim: v=[min(max(x,lim[0]),lim[1]) for x in v]
        cur[k]=v
    sels=[]
    for k,v in cur.items():
        if k=='region': sels.append({'source_figure_uid':FIG['map'],'predicates':[{'clauses':[{'column':'x','range':_rng(v[:2])},{'column':'y','range':_rng(v[2:])}]}]})
        elif k=='ratings_count': sels.append(_sel('log10_ratings',{'column':'log10_ratings','range':[float(np.log10(max(float(x),0)+1)) for x in v[:2]]}))
        else: sels.append(_sel(FILTERS[k],{'column':FILTERS[k],'range':_rng(v)}))
    out=await set_state({'selections':sels},what='clear filters' if not sels else None)
    return out|{'filters':filters_of(M['state'])}
async def t_focus_book(title=None,i=None,radius=None,keep_selections=True):
    """Zoom to a book and its neighbourhood (~400 nearest books on the map). Slider filters are kept unless they hide the book."""
    j=find(title,i) if i is not None else (title_lookup(title,strict=False) or {'i':find(title)})['i']
    r=B.row(j,named=True); FOCUS['book']=bookrow(r)
    rad=float(radius) if radius else _neigh_r(r['x'],r['y'])
    cleared=False; patch={'viewport':_vp(r['x'],r['y'],rad,rad*max(ASPECT,.7))}
    if not keep_selections: patch['selections']=[]; cleared=bool(M['state'].get('selections'))
    else:
        one=B.slice(j,1).filter(filt({'selections':M['state'].get('selections') or []},ignore_viewport=True))
        if one.height==0: patch['selections']=[]; cleared=True
    out=await set_state(patch,what=f"focus on '{r['title']}' (±{rad:.2f})"+(' and clear filters that hid it' if cleared else ''))
    await _broadcast({'type':'focus','book':FOCUS['book']}); return {'focus':FOCUS['book'],'cleared_filters':cleared}|out
async def t_similar_books(title=None,i=None,k=10):
    j=find(title,i); res=similar(j,min(int(k),50)); st=M['state']; inview=set(B.filter(filt(st))['i'].to_list())
    return {'book':bookrow(B.row(j,named=True)),'similar':[r|{'in_current_view':r['i'] in inview} for r in res],
            'note':f"similarity is cosine over {store.META.get('embedding_model','all-MiniLM-L6-v2')} embeddings of {store.META.get('embedding_text','the description')}"+'; map position is a 2D UMAP and does not always keep neighbours together'}
async def t_get_user_profile(): return agent.load_profile(B)
async def t_recommend_start(query=''): r=await rec.start(B,filt(M['state']),describe(M['state']),query=query or ''); await _wizard_recs(r); return r
async def t_recommend_answer(session,option): r=await rec.answer(session,option); await _wizard_recs(r); return r
# ---------- recommendation highlights (shared state): every recommended book gets a yellow ring + "rank title" tag ----------
HIGHLIGHT={'recs':[],'by':None,'at':None}
REC_RE=re.compile(r"\b(recommend\w*|suggest\w*|books? like|similar|what (?:should|can|could) i read|top[- ]?rated|best|good (?:books?|reads?)|must[- ]reads?|picks?|worth reading)\b",re.I)
def _rec_row(b,k):
    i=int(b['i']); r=B.row(i,named=True)
    return {'rank':k+1,'i':i,'title':r['title'],'author':r['author'],'x':round(float(r['x']),6),'y':round(float(r['y']),6),'url':gr_url(i)}
async def set_recs(books,by='assistant',frame=True):
    """Replace the highlighted recommendations (kept until the next recommendation or a reset) and frame the view on them."""
    recs=[_rec_row(b,k) for k,b in enumerate(books[:10]) if b and b.get('i') is not None]
    if not recs and not HIGHLIGHT['recs']: return
    HIGHLIGHT.update(recs=recs,by=by,at=time.time())
    if recs: _log(by,'highlight recommendations: '+'; '.join(f"{r['rank']}. {r['title']}" for r in recs))
    await _broadcast({'type':'recs','books':recs})
    if frame and recs:
        pts=[(r['x'],r['y']) for r in recs]+([(FOCUS['book']['x'],FOCUS['book']['y'])] if FOCUS['book'] else [])
        x0,x1,y0,y1=_cur_vp(); mx,my=(x1-x0)*.08,(y1-y0)*.12
        if not all(x0+mx<=x<=x1-mx and y0+my<=y<=y1-my for x,y in pts):
            xs=[x for x,_ in pts]; ys=[y for _,y in pts]; cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2
            rx=max((max(xs)-min(xs))/2*1.3,.6); ry=max((max(ys)-min(ys))/2*1.35,.45)
            rx,ry=max(rx,ry/max(ASPECT,1e-6)*.85),max(ry,rx*ASPECT*.85)
            await set_state({'viewport':_vp(cx,cy,rx,ry)},by=by,what='frame the recommended books')
async def _wizard_recs(r):
    f=(r or {}).get('final') if isinstance(r,dict) else None
    if f and f.get('top3'): await set_recs(f['top3'],by='recommendation')
def mention_rows(text,limit=8):
    out=[]
    for t in re.findall(r'\*\*([^*\n]{2,140})\*\*|\*([^*\n]{2,140})\*',text or ''):
        t=(t[0] or t[1]).strip(); h=title_lookup(t) if len(t)>2 else None
        if h and h['title'].strip().lower()==t.lower() and h['i'] not in [o['i'] for o in out] and not (FOCUS['book'] and h['i']==FOCUS['book']['i']): out.append(h)
        if len(out)>=limit: break
    return out
async def recs_from_chat(user_text,reply,via,log):
    """Which books did this chat turn recommend? similar_books results, or top rated / named books when the user asked for recommendations."""
    if 'wizard' in log: return
    picks=None
    if 'similar_books' in log: picks=(log['similar_books'].get('similar') or [])[:5]
    elif REC_RE.search(user_text or ''):
        if 'top_rated_in_view' in log: picks=(log['top_rated_in_view'].get('books') or [])[:5]
        else: picks=mention_rows(reply)[:6]
    if picks: await set_recs(picks,by='assistant')

# ---------- map navigation helpers: top rated, topics, authors, relative zoom ----------
NX=NY=200; _XA=B['x'].to_numpy().astype(np.float64); _YA=B['y'].to_numpy().astype(np.float64)
_CX=np.clip(((_XA-X0)/(X1-X0)*NX).astype(int),0,NX-1); _CY=np.clip(((_YA-Y0)/(Y1-Y0)*NY).astype(int),0,NY-1)
def _integral(mask,w=None):
    H=np.zeros((NX,NY)); np.add.at(H,(_CX[mask],_CY[mask]),1.0 if w is None else w[mask])
    I=np.zeros((NX+1,NY+1)); I[1:,1:]=H.cumsum(0).cumsum(1); return H,I
def _box(I,cx,cy,r):
    a,b=max(0,cx-r),min(NX,cx+r+1); c,e=max(0,cy-r),min(NY,cy+r+1); return I[b,e]-I[a,e]-I[b,c]+I[a,c]
def _mask(state):
    return B.with_columns(filt(state,ignore_viewport=True).alias('_m'))['_m'].fill_null(False).to_numpy()
def _vp(cx,cy,rx,ry):
    return {f"{FIG['map']}/x":{'min':float(cx-rx),'max':float(cx+rx)},f"{FIG['map']}/y":{'min':float(cy-ry),'max':float(cy+ry)}}
def _cur_vp():
    vp=M['state'].get('viewport') or {}; x=vp.get(f"{FIG['map']}/x") or {'min':X0,'max':X1}; y=vp.get(f"{FIG['map']}/y") or {'min':Y0,'max':Y1}
    return x['min'],x['max'],y['min'],y['max']
CELL=(X1-X0)/NX; ASPECT=(Y1-Y0)/(X1-X0)
def best_window(m,base,w=None,mode='topic'):
    """Find the map window where the books in mask m concentrate. Returns (cx,cy,rx,ry,info)."""
    Ht,It=_integral(m&base,w); Ha,Ia=_integral(base)
    tot=It[-1,-1]; p0=tot/max(Ia[-1,-1],1)
    S=Ht.copy()                                                     # smoothed seed map
    for _ in range(3): S=(S+np.roll(S,1,0)+np.roll(S,-1,0)+np.roll(S,1,1)+np.roll(S,-1,1))/5
    seeds=np.dstack(np.unravel_index(np.argsort(-S,axis=None)[:12],S.shape))[0]
    if mode!='topic': seeds=np.vstack([np.dstack(np.unravel_index(np.argsort(-Ht,axis=None)[:6],Ht.shape))[0],seeds])
    best=None
    for cx,cy in seeds:
        for r in (4,6,9,13,18,25,34):
            nt=_box(It,cx,cy,r); na=max(_box(Ia,cx,cy,r),1)
            if mode=='topic':
                if nt<25: continue
                lift=(nt/na)/max(p0,1e-9); sc=nt*lift**2
            else:                                                   # author: smallest window holding half their (weighted) books
                lift=(nt/na)/max(p0,1e-9); sh=nt/max(tot,1e-9); sc=(1.0/r if sh>=.5 else sh/100.0)
                if best is not None and sc<=best[0]: continue
                if sh>=.5: best=(sc,cx,cy,r,nt,na,lift); break
            if best is None or sc>best[0]: best=(sc,cx,cy,r,nt,na,lift)
    if best is None: return None
    sc,cx,cy,r,nt,na,lift=best; rx=(r+.5)*CELL
    return (X0+(cx+.5)*CELL, Y0+(cy+.5)*(Y1-Y0)/NY, rx, max(rx*ASPECT*1.0,rx*.7), {'share_in_window':round(nt/max(tot,1e-9),3),'lift':round(float(lift),2)})
TOPICS={
 'science fiction':(['science-fiction','sci-fi','scifi','sf','space-opera'],['science fiction','sci-fi','sci fi','scifi','sf','space opera','space']),
 'fantasy':(['fantasy','high-fantasy','epic-fantasy'],['fantasy','high fantasy','epic fantasy','sword and sorcery']),
 'urban fantasy':(['urban-fantasy'],['urban fantasy']),
 'romance':(['romance','contemporary-romance'],['romance','romances','romantic','love story','love stories']),
 'historical romance':(['historical-romance','regency','regency-romance'],['historical romance','regency']),
 'paranormal romance':(['paranormal-romance'],['paranormal romance']),
 'mystery':(['mystery','mysteries','detective'],['mystery','mysteries','whodunit','whodunnit','detective','detective stories']),
 'cozy mystery':(['cozy-mystery','cozy-mysteries','mystery-cozy'],['cozy mystery','cozy mysteries','cosy mystery','cosy mysteries']),
 'thriller':(['thriller','thrillers','suspense','mystery-thriller'],['thriller','thrillers','suspense','psychological thriller','psychological thrillers']),
 'crime':(['crime','crime-fiction'],['crime','crime fiction','noir']),
 'horror':(['horror'],['horror','scary','creepy','spooky']),
 'historical fiction':(['historical-fiction','historical'],['historical fiction','historical','history']),
 'young adult':(['young-adult','ya','teen'],['young adult','ya','teen','teens']),
 'middle grade':(['middle-grade'],['middle grade']),
 "children's":(['childrens','children','picture-books','children-s-books','kids'],["children's",'childrens','children','kids','picture books']),
 'humor':(['humor','humour','funny','comedy'],['humor','humour','funny','comedy','comedic','hilarious']),
 'dystopian':(['dystopian','dystopia','post-apocalyptic','apocalyptic'],['dystopian','dystopia','dystopias','post-apocalyptic','post apocalyptic','apocalyptic']),
 'paranormal':(['paranormal','supernatural'],['paranormal','supernatural']),
 'vampires':(['vampires','vampire'],['vampire','vampires']),
 'werewolves':(['werewolves','shifters','shifter','shapeshifters'],['werewolf','werewolves','shifters','shapeshifters']),
 'zombies':(['zombies','zombie'],['zombie','zombies']),
 'dragons':(['dragons'],['dragon','dragons']),
 'magic':(['magic'],['magic','wizards','witches','witch']),
 'graphic novels':(['graphic-novels','graphic-novel','comics','comic'],['graphic novel','graphic novels','comics','comic books','comic']),
 'manga':(['manga','mangas'],['manga']),
 'superheroes':(['superheroes','superhero','marvel','dc-comics'],['superhero','superheroes','marvel','dc']),
 'literary fiction':(['literary-fiction'],['literary fiction','literary']),
 'classics':(['classics'],['classics','classic']),
 'christian fiction':(['christian-fiction','christian','christian-romance'],['christian','christian fiction','faith']),
 'lgbtq':(['lgbt','lgbtq','queer','glbt','m-m','lesbian','gay'],['lgbt','lgbtq','lgbtq+','queer','gay','lesbian']),
 'erotica':(['erotica','erotic-romance','erotic'],['erotica','erotic','steamy','spicy']),
 'western':(['western','cowboys','cowboy'],['western','westerns','cowboy','cowboys']),
 'adventure':(['adventure','action-adventure'],['adventure','adventures','action']),
 'steampunk':(['steampunk'],['steampunk']),
 'time travel':(['time-travel'],['time travel','time-travel']),
 'mythology':(['mythology','fairy-tales'],['mythology','myths','fairy tales','fairytales','retellings']),
 'war':(['war','military','wwii'],['war','military','ww2','wwii','world war']),
 'short stories':(['short-stories','short-story','anthology','short-story-collections'],['short stories','short story','anthology','anthologies']),
 'poetry':(['poetry'],['poetry','poems']),
 'chick lit':(['chick-lit','chicklit','womens-fiction','women-s-fiction'],['chick lit','chick-lit',"women's fiction",'womens fiction']),
 'aliens':(['aliens','alien'],['alien','aliens']),
 'ghosts':(['ghosts'],['ghost','ghosts','ghost stories','haunted']),
 'sports':(['sports','sports-romance'],['sport','sports','football','baseball']),
 'christmas':(['christmas','holiday'],['christmas','holiday','holidays']),
 'gothic':(['gothic'],['gothic']),
 'coming of age':(['coming-of-age'],['coming of age','coming-of-age']),
 'alternate history':(['alternate-history'],['alternate history']),
 'magical realism':(['magical-realism'],['magical realism']),
 'animals':(['animals','dogs','cats'],['animals','animal','dogs','dog','cats','cat']),
 'mafia':(['mafia'],['mafia','mob']),
 'pirates':(['pirates'],['pirate','pirates']),
 'spy':(['spy','espionage'],['spy','spies','espionage']),
 'true crime':(['true-crime'],['true crime']),
 'new adult':(['new-adult','college'],['new adult','college']),
}
ALIAS={a:k for k,(t,al) in TOPICS.items() for a in al+[k]}
_TAGS={}
def tags_col():
    if 'c' not in _TAGS: _TAGS['c']=pl.read_parquet(f'{DATA}/extra.parquet',columns=['i','tags']).sort('i')['tags'].fill_null('')
    return _TAGS['c']
def topic_lookup(text):
    """Longest topic alias contained in text -> (topic key, alias) or None."""
    low=' '+re.sub(r'[^a-z0-9+\' -]',' ',(text or '').lower())+' '
    for a in sorted(ALIAS,key=len,reverse=True):
        if re.search(r'(?<![a-z])'+re.escape(a)+r'(?![a-z])',low): return ALIAS[a],a
    return None
def topic_mask(key):
    tg=TOPICS[key][0]
    return tags_col().str.contains(r'(?:^|, )(?:'+'|'.join(re.escape(t) for t in tg)+r')(?:,|$)').to_numpy()
AUTH={}
def authors():
    if 'df' not in AUTH:
        a=B.group_by('author').agg(pl.col('ratings_count').sum().alias('n'),pl.len().alias('k')).drop_nulls('author')
        AUTH['df']=a.with_columns(pl.col('author').str.to_lowercase().alias('l')).sort('n',descending=True)
    return AUTH['df']
def author_lookup(q):
    q=re.sub(r'\s+',' ',(q or '').lower().strip(" .?!'\"")); q=re.sub(r"'s$",'',q)
    if len(q)<4: return None
    A=authors(); h=A.filter(pl.col('l')==q)
    if h.height==0: h=A.filter(pl.col('l').str.contains(r'(?:^| )'+re.escape(q)+r'(?:$| )')).filter(pl.col('n')>=500)
    return None if h.height==0 else h.row(0,named=True)['author']
def _norm(t): return re.sub(r'^(the|a|an) ','',re.sub(r'\s*\(.*?\)\s*','',(t or '').lower()).strip(' .?!"\''))
def title_lookup(q,strict=True):
    """Title -> book. Exact (series/subtitle-insensitive) matches and series matches ("Harry Potter" -> the series) beat substring hits;
    among them the most-rated wins (exact matches get a 5x boost)."""
    q=(q or '').strip(' .?!"\'“”'); nq=_norm(q)
    if len(nq)<3: return None
    best=None
    for r in {x['i']:x for x in search(q,25)+search(nq,25)}.values():
        tl=r['title'].lower(); nt=_norm(r['title'])
        exact=nt==nq; series=bool(re.search(r'\('+re.escape(nq)+r',?\s*#',tl)); pref=nt.startswith(nq) and len(nq)>=max(5,.5*len(nt))
        if not(exact or series or pref): continue
        sc=(r['ratings_count'] or 0)*(5 if exact else 1)
        if best is None or sc>best[0]: best=(sc,r)
    if best: return best[1]
    if strict: return None
    h=search(nq,1); return h[0] if h else None
def top_rated(state,limit=30,m=1000.0):
    if not HAS_RATING:      # no ratings in this dataset (e.g. the Gutenberg sample): most popular first
        v=B.filter(filt(state)).sort('ratings_count',descending=True,nulls_last=True).head(limit)
        return None,[bookrow(r)|{'score':None} for r in v.iter_rows(named=True)]
    v=B.filter(filt(state)).filter(pl.col('rating').is_not_null())
    if v.height==0: return 0,[]
    C=float(v['rating'].mean()); n=pl.col('ratings_count').fill_null(0).cast(pl.Float64)
    v=v.with_columns(((n*pl.col('rating')+m*C)/(n+m)).alias('score')).sort('score',descending=True).head(limit)
    return C,[bookrow(r)|{'score':round(r['score'],3)} for r in v.iter_rows(named=True)]
async def t_top_rated_in_view(limit=10):
    C,rows=top_rated(M['state'],min(int(limit),50)); n=B.filter(filt(M['state'])).height
    if C is None: return {'books_in_view':n,'books':rows,'ranking':'this dataset has no ratings: most popular (ratings_count) first'}
    return {'books_in_view':n,'books':rows,'ranking':f'Bayesian average: (ratings_count*rating + 1000*{C:.2f})/(ratings_count+1000), i.e. rating weighted by number of ratings'}
async def t_zoom(factor=2.0,direction=None):
    """factor>1 zooms in, <1 zooms out (around the current centre or the focused book); direction pans by half a view."""
    x0,x1,y0,y1=_cur_vp(); cx,cy=(x0+x1)/2,(y0+y1)/2; rx,ry=(x1-x0)/2,(y1-y0)/2
    f=float(factor or 1)
    if direction:
        dx={'left':-1,'west':-1,'right':1,'east':1}.get(direction,0); dy={'up':1,'north':1,'down':-1,'south':-1}.get(direction,0)
        cx+=dx*rx; cy+=dy*ry
    if f>1 and FOCUS['book'] and x0<=FOCUS['book']['x']<=x1 and y0<=FOCUS['book']['y']<=y1 and not direction: cx,cy=FOCUS['book']['x'],FOCUS['book']['y']
    rx/=f; ry/=f
    if rx*2>=(X1-X0)*.98: return await t_set_view(reset=True)
    out=await set_state({'viewport':_vp(cx,cy,rx,ry)},what=f"zoom {'in' if f>1 else 'out' if f<1 else ''} ×{f:g}"+(f' pan {direction}' if direction else ''))
    return out
async def t_zoom_to_topic(topic):
    hit=topic_lookup(topic); key=hit[0] if hit else None
    if not key: raise HTTPException(404,f'unknown topic {topic!r}; known: {sorted(TOPICS)}')
    base=_mask(M['state']); w=best_window(topic_mask(key),base,mode='topic')
    if not w: raise HTTPException(404,f'no {key} books under the current filters')
    cx,cy,rx,ry,info=w; FOCUS['book']=None
    out=await set_state({'viewport':_vp(cx,cy,rx,ry)},what=f'zoom to the {key} area'); await _broadcast({'type':'focus','book':None})
    return {'topic':key,**info,'note':'genre is not a filter; this zooms to where books tagged '+key+' concentrate on the description map'}|out
async def t_zoom_to_author(author):
    name=author_lookup(author)
    if not name: raise HTTPException(404,f'no author matching {author!r}')
    m=(B['author']==name).fill_null(False).to_numpy(); w=np.log10(B['ratings_count'].fill_null(0).to_numpy()+10.0)
    win=best_window(m,np.ones(B.height,bool),w,mode='author'); cx,cy,rx,ry,info=win
    top=B.filter(pl.col('author')==name).sort('ratings_count',descending=True).head(5)
    tb=top.row(0,named=True); rx=max(rx,.6); ry=max(ry,.5)
    if not(cx-rx<=tb['x']<=cx+rx and cy-ry<=tb['y']<=cy+ry):     # always show their best-known book
        x0,x1=min(cx-rx,tb['x']-.3),max(cx+rx,tb['x']+.3); y0,y1=min(cy-ry,tb['y']-.3),max(cy+ry,tb['y']+.3)
        cx,cy,rx,ry=(x0+x1)/2,(y0+y1)/2,(x1-x0)/2,(y1-y0)/2
    FOCUS['book']=bookrow(top.row(0,named=True))
    out=await set_state({'viewport':_vp(cx,cy,rx,ry)},what=f'zoom to books by {name}')
    await _broadcast({'type':'focus','book':FOCUS['book']})
    return {'author':name,'books':[bookrow(r) for r in top.iter_rows(named=True)],**info}|out
def _neigh_r(x,y):
    """Smallest square around (x,y) holding ~400 books (all books, not filtered)."""
    if 'I' not in _TAGS: _TAGS['I']=_integral(np.ones(B.height,bool))[1]
    I=_TAGS['I']; cx=int(np.clip((x-X0)/(X1-X0)*NX,0,NX-1)); cy=int(np.clip((y-Y0)/(Y1-Y0)*NY,0,NY-1))
    for r in range(2,40):
        if _box(I,cx,cy,r)>=400: return max((r+.5)*CELL,.35)
    return 1.5

TOOLS={
 'get_view_state':(t_get_view_state,'What the human currently sees: map zoom, the four range-slider filters (year, rating, ratings_count, pages; null = not set), who changed it last, recent changes, and a summary (books in view, top books by ratings_count, years, avg rating).',{}),
 'list_books_in_view':(t_list_books_in_view,'Books inside the current map zoom AND all selections, most-rated first.',{'limit':{'type':'integer','default':20},'offset':{'type':'integer','default':0}}),
 'search_book':(t_search_book,'Find books by title substring (prefix matches first, then most rated). Returns i, title, author, year, rating, map x/y.',{'title':{'type':'string'},'limit':{'type':'integer','default':5}}),
 'set_view':(t_set_view,'Zoom the description map to a rectangle in UMAP coordinates (the hist2d re-bins), or reset=true for the full map.',{'x_min':{'type':'number'},'x_max':{'type':'number'},'y_min':{'type':'number'},'y_max':{'type':'number'},'reset':{'type':'boolean','default':False}}),
 'set_filters':(t_set_filters,'Move the range sliders under the map (shared FlexViz state; the UI updates live). Each is [min,max]: year, rating (1-5 average), ratings_count (number of ratings / popularity, e.g. [1000,1000000]), pages (page count; books without one are excluded while set). Other filters are kept unless clear=true; pass [] to remove one. region=[x_min,x_max,y_min,y_max] selects a map rectangle.',
     {'year':{'type':'array','items':{'type':'number'}},'rating':{'type':'array','items':{'type':'number'}},'ratings_count':{'type':'array','items':{'type':'number'}},'pages':{'type':'array','items':{'type':'number'}},'region':{'type':'array','items':{'type':'number'}},'clear':{'type':'boolean','default':False}}),
 'focus_book':(t_focus_book,'Find a book by title (or index i) and zoom the map to it and its neighbourhood (books with similar blurbs). Keeps the slider filters unless they would hide the book.',{'title':{'type':'string'},'i':{'type':'integer'},'radius':{'type':'number','description':'optional half-width in map units; default = neighbourhood of ~400 books'},'keep_selections':{'type':'boolean','default':True}}),
 'zoom':(t_zoom,'Relative zoom/pan of the map: factor>1 zooms in (2 = twice as close), factor<1 zooms out (0.5); direction = left|right|up|down pans by half a view.',{'factor':{'type':'number','default':2.0},'direction':{'type':'string','enum':['left','right','up','down']}}),
 'zoom_to_topic':(t_zoom_to_topic,'Genre/topic is NOT a filter. This zooms the map to where books tagged with a topic concentrate (e.g. science fiction, fantasy, romance, mystery, thriller, horror, historical fiction, young adult, humor, dystopian, vampires, dragons...). Respects the current slider filters.',{'topic':{'type':'string'}}),
 'zoom_to_author':(t_zoom_to_author,"Zoom the map to where an author's books sit and list their most-rated books.",{'author':{'type':'string'}}),
 'top_rated_in_view':(t_top_rated_in_view,'Best books in the current zoom + filters, ranked by rating weighted by number of ratings (Bayesian average). Use for "best", "top rated", "what should I read here".',{'limit':{'type':'integer','default':10}}),
 'similar_books':(t_similar_books,'Nearest books by description embedding (not by map position). Flags which ones are in the current view.',{'title':{'type':'string'},'i':{'type':'integer'},'k':{'type':'integer','default':10}}),
 'recommend_start':(t_recommend_start,'Start the recommendation wizard on the books currently in view (zoom + slider filters; best ~50 by Bayesian rating, boosted by words from query). Returns the first multiple-choice question (options with id) or, if few books, the final top 3.',{'query':{'type':'string','description':'what the reader is looking for, in their words'}}),
 'recommend_answer':(t_recommend_answer,'Answer the current wizard question with option id (or exact label). Returns the next question or the final top 3 with summary and reason.',{'session':{'type':'string'},'option':{'type':'integer'}}),
 'get_user_profile':(t_get_user_profile,"The user's reading profile (user_profile.md and, if present, goodreads_library_export.csv in ATLAS_HOME matched to this catalogue).",{}),
}
def tool_list(): return [{'name':n,'description':d,'inputSchema':{'type':'object','properties':p}} for n,(f,d,p) in TOOLS.items()]
async def call_tool(name,args):
    if name not in TOOLS: raise HTTPException(404,f'unknown tool {name}')
    return await TOOLS[name][0](**(args or {}))

# ---------- HTTP ----------
app=FastAPI(title='Atlas'); mount_into(app,'/flexviz')
@app.get('/',response_class=HTMLResponse)
def index():
    """index.html with dataset-aware example prompts (the originals assume a large modern fiction catalogue)."""
    h=open(f'{HERE}/index.html').read()
    top=B.sort('ratings_count',descending=True,nulls_last=True).head(1)
    if top.height:
        ex=re.sub(r'\s*[(:;].*$','',top['title'][0]).strip()[:40] or top['title'][0][:40]
        h=h.replace('The Martian',ex.replace('"','').replace("'",''))
    if not HAS_RATING: h=h.replace('only 4★+ sci-fi after 2010','zoom to science fiction').replace('Only highly rated sci-fi after 2010','Zoom to science fiction')
    if not B['pages'].is_not_null().any(): h=h.replace('under 300 pages','before 1900').replace('Under 300 pages','Before 1900')
    return h
@app.get('/api/config')
def config(h:int=760): return {'dash_url':dash_url(h),'figures':FIG,'llm':agent.status(),'total':B.height,'subtitle':CORPUS,'has_rating':bool(HAS_RATING),'attribution':store.META.get('attribution','')}
from fastapi.responses import FileResponse
import flexviz as _fv
@app.get('/assets/wordmark-{v}.png')
def wordmark(v:str):
    if v not in('dark','light'): raise HTTPException(404)
    return FileResponse(os.path.join(os.path.dirname(_fv.__file__),'adapters',f'wordmark-{v}.png'))
@app.get('/assets/fv-light.css')
def fv_light():
    """FlexViz's own light design tokens (first :root block of its theme.css), re-declared unconditionally so an OS dark mode can't override them."""
    import re
    from fastapi.responses import PlainTextResponse
    css=open(os.path.join(os.path.dirname(_fv.__file__),'adapters','js','theme.css')).read()
    block=re.search(r'^:root\s*\{.*?\n\}',css,re.S|re.M).group(0)
    block=re.sub(r'--fv-brand-image:[^;]*;','--fv-brand-image: url(/assets/wordmark-light.png);',block)
    return PlainTextResponse(block,media_type='text/css')
@app.get('/assets/{name}')
def asset(name:str):
    p=os.path.join(HERE,'assets',os.path.basename(name))
    if not os.path.exists(p): raise HTTPException(404)
    return FileResponse(p)
@app.get('/api/state')
def get_state(): return view_state()
@app.post('/api/state')
async def post_state(req:Request):
    """Body: {"state": {...FlexViz state patch...}} (raw) or {"tool": name, "args": {...}}."""
    b=await req.json()
    if 'tool' in b: return await call_tool(b['tool'],b.get('args'))
    return await set_state(b['state'],by=b.get('by','external'))
@app.post('/api/browser_state')
async def browser_state(req:Request):
    """The page posts flexvizState({compact:true}) whenever its revision changes."""
    b=await req.json()
    if _stale(b.get('state') or {}): return {'stale':True,'boot':BOOT}
    M['browser_seen']=time.time()
    if b.get('ack') and b['ack'] in ACK: ACK[b['ack']].set()
    st=b['state']; changed=json.dumps(st,sort_keys=True)!=json.dumps(M['state'],sort_keys=True)
    M['client_state']=b.get('client_state',{}); M['fv_revision']=b.get('revision',0)
    if changed:
        M['state']=st; M['rev']+=1
        if not b.get('ack'): M['by']='human'; M['at']=time.time(); _log('human',describe(st))
    n,rows=books_in_view(M['state'],50); C,top=top_rated(M['state'],40)
    return {'books_in_view':n,'books':rows,'top_rated':top,'focus':FOCUS['book'],'by':M['by'],'description':describe(M['state']),'filters':filters_of(M['state'])}
@app.get('/api/top')
def api_top(limit:int=30): C,rows=top_rated(M['state'],min(limit,100)); return {'books':rows,'mean':C}
PROFILE=str(paths.home()/'user_profile.md')
# ---- Librarian model picker: local Ollama models or an OpenAI-compatible endpoint ----
def _model_state(extra=None):
    tags=agent.ollama_tags(True)
    return {'current':agent.public_conf(),'provider':agent.provider(),'ollama':{'reachable':tags is not None,'host':agent.OLLAMA,'models':tags or []},
            'default_local':os.environ.get('AGENT_MODEL','ministral-3:8b'),'rec_model':agent.REC_MODEL()}|(extra or {})
@app.get('/api/librarian/model')
def lib_model_get(): return _model_state()
@app.post('/api/librarian/model')
async def lib_model_set(req:Request):
    b=await req.json(); kind=b.get('kind')
    if kind=='default':     # back to the server default (local Ollama, AGENT_MODEL); no test call
        agent.save_conf({'kind':'ollama','model':None,'base_url':None,'api_key':None}); return _model_state({'saved':True})
    if kind=='ollama':
        d={'kind':'ollama','model':str(b.get('model') or ''),'base_url':None,'api_key':None}
    elif kind=='openai':
        base=str(b.get('base_url') or '').strip().rstrip('/'); model=str(b.get('model') or '').strip()
        if not re.match(r'^https?://[^\s/]+',base): raise HTTPException(400,'base URL must start with http:// or https://')
        if not model: raise HTTPException(400,'model name is required')
        key=b.get('api_key')
        if not key and agent.CONF['kind']=='openai' and agent.CONF['base_url']==base: key=agent.CONF['api_key']   # keep the saved key
        d={'kind':'openai','model':model,'base_url':base,'api_key':(str(key).strip() or None) if key else None}
    else: raise HTTPException(400,"kind must be 'ollama' or 'openai'")
    t=await agent.test_model(d)
    if t['ok']: agent.save_conf(d)
    return _model_state({'test':t,'saved':t['ok']})
@app.post('/api/librarian/model/test')
async def lib_model_test(): return _model_state({'test':await agent.test_model(dict(agent.CONF,model=agent.OLLAMA_MODEL() if agent.CONF['kind']=='ollama' else agent.CONF['model']))})
@app.get('/api/profile')
def get_profile():
    return {'text':open(PROFILE).read() if os.path.exists(PROFILE) else '','goodreads':{'connected':False,'status':'coming soon'}}
@app.post('/api/profile')
async def post_profile(req:Request):
    b=await req.json(); t=str(b.get('text',''))[:8000]
    tmp=PROFILE+'.tmp'; open(tmp,'w').write(t); os.replace(tmp,PROFILE); _log('human','updated reading profile')
    return {'ok':True,'chars':len(t),'saved_at':time.strftime('%H:%M:%S')}
@app.get('/api/events')
async def events(req:Request):
    q=asyncio.Queue(); CLIENTS.add(q)
    async def gen():
        try:
            yield f"data: {json.dumps({'type':'hello','boot':BOOT,'state':M['state'],'focus':FOCUS['book'],'recs':HIGHLIGHT['recs']})}\n\n"
            while True:
                if await req.is_disconnected(): break
                try: msg=await asyncio.wait_for(q.get(),15); yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError: yield ': ping\n\n'
        finally: CLIENTS.discard(q)
    return StreamingResponse(gen(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
@app.get('/api/tools')
def tools(): return {'tools':tool_list()}
@app.post('/api/tools/{name}')
async def tool(name:str,req:Request):
    try: args=await req.json()
    except Exception: args={}
    return await call_tool(name,args)
# ---------- map overlay data (page-side layer on top of the FlexViz hist2d) ----------
def _view(x0,x1,y0,y1):
    st={'viewport':{f"{FIG['map']}/x":{'min':x0,'max':x1},f"{FIG['map']}/y":{'min':y0,'max':y1}},'selections':M['state'].get('selections') or []}
    return B.filter(filt(st))
def _lab(c): return c.split(' ',1)[-1].replace('-',' ')
@app.get('/api/map/overlay')
def map_overlay(x0:float,x1:float,y0:float,y1:float,all_max:int=2500,top:int=600,points:int=1):
    v=_view(x0,x1,y0,y1); n=v.height
    pts=v.sort('ratings_count',descending=True).head((all_max if n<=all_max else top) if points else 0)
    P={k:pts[c].to_list() for k,c in [('i','i'),('x','x'),('y','y'),('t','title'),('a','author'),('yr','year'),('r','rating'),('n','ratings_count'),('g','genre'),('c','cluster')]}
    P['x']=[round(float(x),4) for x in P['x']]; P['y']=[round(float(y),4) for y in P['y']]; P['r']=[None if r is None else round(float(r),2) for r in P['r']]
    L=(v.group_by('cluster').agg(pl.len().alias('n'),pl.col('x').median().alias('x'),pl.col('y').median().alias('y')).sort('n',descending=True))
    L=L.filter(pl.col('n')>=max(5,int(n*0.004)))
    labels=[]   # cluster labels removed with the topic clusters
    return {'in_view':n,'all':n<=all_max,'full':(x1-x0)>=0.8*(X1-X0),'points':P,'labels':labels}
# all books as compact columns for the client-side WebGL layer (sent once, cached by the browser)
def _points_blob():
    o=B.sort(['ratings_count','i'],nulls_last=False)          # low-count first -> popular drawn on top
    _top=lambda c:sorted(B[c].drop_nulls().value_counts(sort=True).head(254)[c].to_list())   # u8 codes: 254 most common, 255 = other
    G=_top('genre'); C=_top('cluster')
    gi={g:k for k,g in enumerate(G)}; ci={c:k for k,c in enumerate(C)}
    cols=[o['i'].cast(pl.UInt32).to_numpy(),o['x'].cast(pl.Float32).to_numpy(),o['y'].cast(pl.Float32).to_numpy(),
          o['year'].fill_null(-1).cast(pl.Int16).to_numpy(),(o['rating'].fill_null(0)*1000).round().cast(pl.UInt16).to_numpy(),
          (o['log10_ratings'].fill_null(0)*1000).round().cast(pl.UInt16).to_numpy(),
          np.array([gi.get(g,255) for g in o['genre'].to_list()],dtype=np.uint8),np.array([ci.get(c,255) for c in o['cluster'].to_list()],dtype=np.uint8),np.zeros((4-(20*o.height)%4)%4,dtype=np.uint8),o['ratings_count'].fill_null(0).cast(pl.UInt32).to_numpy(),o['pages'].fill_null(0).clip(0,65535).cast(pl.UInt16).to_numpy()]
    blob=b''.join(np.ascontiguousarray(c).tobytes() for c in cols)
    import hashlib
    meta={'n':o.height,'layout':['i:u32','x:f32','y:f32','year:i16','rating:u16/1000','log10_ratings:u16/1000','genre:u8','cluster:u8','<pad to 4>','ratings_count:u32','pages:u16 (0=unknown)'],'genres':G,'clusters':C,
          'full':{'x':[X0,X1],'y':[Y0,Y1]},'version':hashlib.md5(blob).hexdigest()[:12],'bytes':len(blob)}
    return meta,blob
_PB={}
def points_blob():
    if not _PB: _PB['meta'],_PB['blob']=_points_blob()
    return _PB['meta'],_PB['blob']
PERF=[]
import uuid as _uuid
BOOT=_uuid.uuid4().hex[:12]
def _stale(st):
    '''A tab loaded from an older server run references figure uids that no longer exist; ignore it so it cannot clobber the shared state.'''
    ok=set(FIG.values())
    if any(s.get('source_figure_uid') not in ok for s in st.get('selections') or []): return True
    return any(k.split('/')[0] not in ok for k in (st.get('viewport') or {}))
@app.post('/api/perf')
async def perf_post(req:Request):
    d=await req.json(); d['t']=time.strftime('%H:%M:%S'); PERF.append(d); del PERF[:-20]; return {'ok':True}
@app.get('/api/perf')
def perf_get(): return PERF

@app.get('/api/map/meta')
def map_meta(): return points_blob()[0]
@app.get('/api/map/points.bin')
def map_points(v:str=''):
    meta,blob=points_blob()
    return Response(blob,media_type='application/octet-stream',headers={'Cache-Control':'public, max-age=31536000, immutable' if v==meta['version'] else 'no-cache','ETag':meta['version']})
@app.get('/api/map/bin')
def map_bin(x0:float,x1:float,y0:float,y1:float):
    v=_view(x0,x1,y0,y1); top=v.sort('ratings_count',descending=True).head(3)
    cl=v.group_by('cluster').len().sort('len',descending=True).head(1)['cluster'].to_list()
    return {'count':v.height,'cluster':_lab(cl[0]) if cl else None,'top':[{'title':r['title'],'author':r['author'],'rating':None if r['rating'] is None else round(r['rating'],2)} for r in top.iter_rows(named=True)]}
@app.get('/go/book/{i}')
def go_book(i:int):
    u=gr_url(i)
    if not u: raise HTTPException(404,'no link for this book')
    return RedirectResponse(u,status_code=302)
@app.get('/api/book/{i}')
def api_book(i:int):
    if not 0<=i<B.height: raise HTTPException(404,'no such book')
    r=bookrow(B.row(i,named=True)); e=rec._extra([i]).get(i,{})
    d=(e.get('desc') or '').strip(); r['blurb']=(d[:200].rsplit(' ',1)[0]+'…') if len(d)>200 else d; r['pages']=e.get('pages')
    return r
@app.get('/api/search')
def api_search(q:str,limit:int=8): return search(q,min(limit,20))
@app.post('/api/rec/start')
async def rec_start(req:Request):
    try: b=await req.json()
    except Exception: b={}
    return await t_recommend_start((b or {}).get('query',''))
@app.post('/api/rec/answer')
async def rec_answer(req:Request):
    b=await req.json(); return await t_recommend_answer(b.get('session'),b.get('option'))
def _H():
    return {'title':title_lookup,'author':author_lookup,'topic':topic_lookup,'filters':lambda:filters_of(M['state']),'top':lambda n=5:top_rated(M['state'],n)[1],
            'focus':lambda:FOCUS['book'],'full':(X0,X1,Y0,Y1),'cur_vp':_cur_vp}
@app.get('/api/librarian/parse')
def librarian_parse(q:str):
    p=librarian.parse(q,_H())
    return {'recognised':librarian.recognised(p),'filters':p.f,'remove':sorted(p.remove),'clear':p.clear,'view':p.view,'zoom':p.zoom,'pan':p.pan,
            'focus':p.focus and p.focus['title'],'like':p.like and p.like['title'],'author':p.author,'topic':p.topic,'ask':p.ask,'wizard':p.wizard,'leftover':p.t.strip()}
@app.post('/api/chat')
async def chat(req:Request):
    b=await req.json()
    log={}
    async def ct(name,args):
        res=await call_tool(name,args)
        if name in ('similar_books','top_rated_in_view') and isinstance(res,dict): log[name]=res
        if name.startswith('recommend_'): log['wizard']=True
        return res
    r=await agent.chat(b.get('messages',[]),tool_list(),ct,view_state,H=_H(),model=b.get('model'),rules=b.get('rules',True) is not False)
    try:
        msgs=b.get('messages') or []; await recs_from_chat(msgs[-1].get('content','') if msgs else '',r.get('reply',''),r.get('via'),log)
    except Exception as e: print('recs_from_chat',e)
    r['recommended']=HIGHLIGHT['recs']
    try: r['links']=mention_links(r.get('reply',''))
    except Exception: r['links']={}
    return r
threading.Thread(target=points_blob,daemon=True).start()   # warm the WebGL point blob
