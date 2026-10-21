"""CoRe-TFM resumable real-inference and evidence engine, protocol v2 (2026-10-19).
Standalone implementation incorporating the paired-OpenML specification and four
view construction from bnssaanirudh/core-tfm. New experiments must be analyzed
separately from the historical main-branch benchmark.
"""
from __future__ import annotations
import os, sys, json, hashlib, tempfile, time, traceback, gc, importlib.metadata as im
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROTOCOL = "coretfm-v2-independent-fixed-test-2026-10-19"
DATASETS = {
    "anneal":(46906,"classes","steel"), "credit":(46918,"good_or_bad_customer","checking_status"),
    "phishing":(46963,"WebsiteType","SFH"), "mic":(46980,"LET_IS","ASP_S_n"),
    "customer":(46938,"bad_client_target","education"), "car":(40975,"class","safety"),
    "marketing":(46940,"Response","Marital_Status"), "nursery":(26,"class","health"),
    "diamonds":(42225,"cut","color"), "wine":([40691,40498],"class","color"),
}
CHECKPOINTS={"tabiclv2":"tabicl-classifier-v2-20260212.ckpt","tabpfn3":"ModelVersion.V3"}


def utcnow(): return datetime.now(timezone.utc).isoformat()
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_obj(v): return sha_bytes(json.dumps(v,sort_keys=True,default=str).encode())
def _atomic_bytes(path:Path, data:bytes):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.tmp_',delete=False) as f:
        f.write(data); tmp=Path(f.name)
    os.replace(tmp,path)
def put_json(path,payload): _atomic_bytes(Path(path),json.dumps(payload,indent=2,sort_keys=True,default=str).encode())
def get_json(path): return json.loads(Path(path).read_text())
def out_root():
    custom=os.environ.get('COREFM_OUT')
    if custom: return Path(custom).expanduser().resolve()
    mounted=Path('/content/drive/MyDrive')
    if mounted.exists() and mounted.is_dir(): return mounted/'CoRe_TFM_Resumable_v2'
    return Path.cwd()/'CoRe_TFM_Resumable_v2_results'
def init_dirs(root=None):
    root=Path(root) if root is not None else out_root()
    for k in ['datasets','splits','real_caches','smoke_caches','reports','runtime','failures']:(root/k).mkdir(parents=True,exist_ok=True)
    return root

def cuda_status():
    try:
        import torch
        ok=bool(torch.cuda.is_available())
        return {'cuda':ok,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if ok else None}
    except Exception as e: return {'cuda':False,'reason':repr(e)}

def version(package):
    try: return im.version(package)
    except im.PackageNotFoundError: return 'not_installed'


def load_pair_dataset(name,root=None,download=True):
    """Pinned two-outcome OpenML pair; dataset bytes cached on Drive as pickle.
    Missing values and derived features follow the repository's openml.py policy.
    No label from validation/test ever enters model input other than hypothetical
    conditioning values enumerated over every possible category.
    """
    if name not in DATASETS:raise KeyError(name)
    root=init_dirs(root);loc=root/'datasets'/f'{name}.pkl'
    if loc.exists():
        data=pd.read_pickle(loc)
        return data['X'],data['a'],data['b'],data['source']
    if not download: raise FileNotFoundError(loc)
    from sklearn.datasets import fetch_openml
    ids,ta,tb=DATASETS[name]
    if name=='wine':
        cols=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']
        parts=[]
        for oid,c in ((40691,'red'),(40498,'white')):
            ds=fetch_openml(data_id=oid,as_frame=True,parser='auto')
            x=ds.data.copy();x.columns=cols;x['__a__']=pd.Series(ds.target).to_numpy();x['__b__']=c;parts.append(x)
        frame=pd.concat(parts,ignore_index=True)
        source={'openml_ids':[40691,40498],'target_a':ta,'target_b':tb}
    else:
        ds=fetch_openml(data_id=ids,as_frame=True,parser='auto')
        x=ds.data.copy()
        if tb not in x:raise ValueError(f'{name}: expected target B {tb} absent; columns={list(x.columns)[:15]}')
        x['__a__']=np.asarray(ds.target);x['__b__']=x.pop(tb)
        if name=='mic':
            features=x.drop(columns=['__a__','__b__'])
            x=x.loc[:, list(features.columns[features.isna().mean()<=.02])+['__a__','__b__']]
        if name=='marketing':
            for col in x.columns:
                if 'date' in str(col).lower() or 'enroll' in str(col).lower():
                    parsed=pd.to_datetime(x[col],errors='coerce',dayfirst=True)
                    if parsed.notna().mean()>.95:
                        x[col]=(parsed-parsed.min()).dt.days.astype(float);break
        frame=x
        source={'openml_id':ids,'target_a':ta,'target_b':tb}
    frame=frame.dropna(axis=0).reset_index(drop=True)
    a=frame.pop('__a__').astype('category'); b=frame.pop('__b__').astype('category')
    X=frame.copy()
    # NOTE: no leakage - targets are removed from predictors.
    if X.columns.duplicated().any():raise ValueError('duplicate feature names')
    data={'X':X,'a':a,'b':b,'source':source}
    pd.to_pickle(data,loc)
    return X,a,b,source


def coverage_order(indices, a, b, rng):
    """Deterministically frontload both marginal class coverage for nested caps."""
    seq=np.array(indices,dtype=int).copy();rng.shuffle(seq)
    all_a=set(a.iloc[seq].astype(str));all_b=set(b.iloc[seq].astype(str))
    seen_a=set();seen_b=set();picks=[];unused=list(seq)
    while (seen_a!=all_a or seen_b!=all_b) and unused:
        gains=[(int(str(a.iloc[i]) not in seen_a)+int(str(b.iloc[i]) not in seen_b)) for i in unused]
        pos=int(np.argmax(gains)); i=unused.pop(pos)
        picks.append(i);seen_a.add(str(a.iloc[i]));seen_b.add(str(b.iloc[i]))
    return np.array(picks+unused,dtype=int),len(picks)


def make_split(name,seed=23,fold=0,root=None,test_n=128,val_n=205):
    from sklearn.model_selection import train_test_split
    root=init_dirs(root);X,a,b,source=load_pair_dataset(name,root)
    path=root/'splits'/f'{name}__seed{seed}__fold{fold}.json'
    if path.exists():
        p=get_json(path)
        if (p.get('protocol')!=PROTOCOL or p['n_total']!=len(X) or p.get('requested_test_n')!=test_n or p.get('requested_val_n')!=val_n):
            raise ValueError('split incompatible with protocol, dataset size or test/validation reservation; refusing overwrite')
        return p
    if len(X)<test_n+val_n+65:raise ValueError(f'{name}: insufficient rows for fixed test/validation split')
    idx=np.arange(len(X));r=seed*1009+fold*100003
    # Deterministic random outer split; no stratification promise for very rare joint classes.
    pool,test=train_test_split(idx,test_size=test_n,random_state=r,shuffle=True)
    train_pool,val=train_test_split(pool,test_size=val_n,random_state=r+17,shuffle=True)
    ordered,coverage=coverage_order(train_pool,a,b,np.random.default_rng(r+29))
    vordered=np.array(val,dtype=int);np.random.default_rng(r+31).shuffle(vordered)
    split={'protocol':PROTOCOL,'dataset':name,'seed':seed,'fold':fold,'n_total':len(X),'source':source,
           'train_pool_ids':ordered.tolist(),'val_ids':vordered.tolist(),'test_ids':np.array(test,dtype=int).tolist(),
           'coverage_frontload_rows':coverage,'requested_test_n':test_n,'requested_val_n':val_n,'classes_a':list(map(str,a.cat.categories)),
           'classes_b':list(map(str,b.cat.categories))}
    assert_no_overlap(split)
    split['split_sha256']=sha_obj({k:v for k,v in split.items() if k!='split_sha256'})
    put_json(path,split);return split

def assert_no_overlap(s):
    i=[set(map(int,s[k])) for k in ['train_pool_ids','val_ids','test_ids']]
    if any(len(set(s[k]))!=len(s[k]) for k in ['train_pool_ids','val_ids','test_ids']):raise AssertionError('duplicate IDs within split')
    if i[0]&i[1] or i[0]&i[2] or i[1]&i[2]:raise AssertionError('train/val/test overlap')

def _condition_col(X,name,values,value):
    v=X.copy(); v[name]=pd.Categorical([str(value)]*len(v),categories=list(map(str,values)))
    return v

def _aligned_proba(est,X,classes):
    pred=np.asarray(est.predict_proba(X),dtype=float)
    order={str(c):i for i,c in enumerate(np.asarray(est.classes_).tolist())}
    if set(order)!=set(map(str,classes)):raise RuntimeError('model classes differ from the frozen global label domain')
    pred=pred[:,[order[str(c)] for c in classes]]
    if pred.shape!=(len(X),len(classes)) or not np.isfinite(pred).all():raise ValueError('invalid model predict_proba')
    if np.min(pred)<-1e-9 or np.max(pred)>1+1e-9:raise ValueError('out-of-range probability')
    pred=np.maximum(pred,0)
    if (pred.sum(axis=1)<=0).any():raise ValueError('zero probability mass')
    return pred/pred.sum(axis=1,keepdims=True)

def make_factory(model,device='cuda',seed=23):
    if model=='tabiclv2':
        from tabicl import TabICLClassifier
        def f():return TabICLClassifier(n_estimators=2,kv_cache=False,batch_size=2,checkpoint_version=CHECKPOINTS['tabiclv2'],device=device,random_state=seed)
        return f
    if model=='tabpfn3':
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        if not hasattr(ModelVersion,'V3'):raise RuntimeError('TabPFN V3 not supported by installed package')
        def f():
            obj=TabPFNClassifier.create_default_for_version(ModelVersion.V3)
            p=obj.get_params();changes={k:v for k,v in {'n_estimators':2,'device':device,'random_state':seed}.items() if k in p}
            if changes:obj.set_params(**changes)
            return obj
        return f
    if model=='smoke_logreg':
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder,StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression
        def f():
            # Fully self-contained smoke model; never publish as a TFM run.
            return make_pipeline(ColumnTransformer([('category',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),lambda df:list(df.select_dtypes(exclude='number').columns)),('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),lambda df:list(df.select_dtypes(include='number').columns))],remainder='drop'),LogisticRegression(max_iter=300))
        return f
    raise ValueError(f'Unknown model: {model}')

def four_views(factory,Xtrain,atrain,btrain,Xval,Xtest,classes_a,classes_b):
    """Fit four estimators with distinct views; return all arrays and fit/predict timings.
    Only training labels enter conditional training features. For prediction each
    possible conditioning value is enumerated; test ground truth is never used.
    """
    nva=len(Xval);nte=len(Xtest);ka=len(classes_a);kb=len(classes_b)
    trainA=pd.Series(pd.Categorical(atrain.astype(str),categories=classes_a))
    trainB=pd.Series(pd.Categorical(btrain.astype(str),categories=classes_b))
    Xa=Xtrain.reset_index(drop=True).copy();V=Xval.reset_index(drop=True);T=Xtest.reset_index(drop=True)
    if any(c in Xa for c in ['__cond_A__','__cond_B__']):raise ValueError('reserved feature collision')
    views={};times={};
    def view(key,y,extra_col=None,extra_vals=None,condition_on=None):
        if extra_col is None:
            train=Xa; xvs=[V];xts=[T]
        else:
            train=_condition_col(Xa,extra_col,extra_vals,np.nan) # overwritten below
            train[extra_col]=pd.Categorical(np.asarray(condition_on).astype(str),categories=list(map(str,extra_vals)))
            xvs=[_condition_col(V,extra_col,extra_vals,c) for c in extra_vals]
            xts=[_condition_col(T,extra_col,extra_vals,c) for c in extra_vals]
        start=time.perf_counter();est=factory();est.fit(train,y);fit_secs=time.perf_counter()-start
        pred_start=time.perf_counter()
        expected=classes_a if key in ('a','a_given_b') else classes_b
        pv=np.stack([_aligned_proba(est,x,expected) for x in xvs],axis=1)
        pt=np.stack([_aligned_proba(est,x,expected) for x in xts],axis=1)
        inference_secs=time.perf_counter()-pred_start
        if pv.shape[1]==1:pv=pv[:,0,:];pt=pt[:,0,:]
        views[key+'_val']=pv;views[key+'_test']=pt
        times[key]={'fit_with_load_sec':fit_secs,'predict_sec':inference_secs,'n_predict_calls':len(xvs)+len(xts)}
        del est;gc.collect()
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except ImportError:pass
    view('a',trainA)
    view('b',trainB)
    view('a_given_b',trainA,extra_col='__cond_B__',extra_vals=classes_b,condition_on=trainB)
    view('b_given_a',trainB,extra_col='__cond_A__',extra_vals=classes_a,condition_on=trainA)
    assert views['a_given_b_val'].shape==(nva,kb,ka)
    assert views['b_given_a_val'].shape==(nva,ka,kb)
    assert views['a_given_b_test'].shape==(nte,kb,ka)
    return views,times

def _validate_probability_view(arr,axis=-1):
    arr=np.asarray(arr,dtype=float)
    if not np.all(np.isfinite(arr)) or (arr<-1e-8).any():raise ValueError('invalid probabilities')
    if not np.allclose(arr.sum(axis=axis),1,atol=1e-4):raise ValueError('probability rows not normalized')

def run_case(dataset,model,seed=23,fold=0,train_cap=64,root=None,test_n=128,val_n=205,allow_cpu=False):
    root=init_dirs(root)
    if model!='smoke_logreg' and not cuda_status()['cuda'] and not allow_cpu:
        return {'status':'SKIPPED_NO_GPU','dataset':dataset,'model':model,'explanation':'Run in a CUDA Colab session, or explicitly allow CPU (slow).'}
    split=make_split(dataset,seed,fold,root,test_n=test_n,val_n=val_n)
    assert_no_overlap(split)
    ids=np.asarray(split['train_pool_ids'],dtype=int)
    if train_cap>len(ids):return {'status':'SKIPPED_INSUFFICIENT_TRAIN_POOL','train_cap':train_cap,'train_pool_size':len(ids)}
    if train_cap<split['coverage_frontload_rows']:
        return {'status':'SKIPPED_CLASS_COVERAGE','train_cap':train_cap,'minimum_required':split['coverage_frontload_rows']}
    filekey=f'{dataset}__{model}__seed{seed}__fold{fold}__train{train_cap}'
    cdir=root/('smoke_caches' if model=='smoke_logreg' else 'real_caches')
    path=cdir/(filekey+'.npz')
    if path.exists():
        try:
            _,meta=read_cache(path)
            if meta['split_sha256']==split['split_sha256'] and meta['protocol']==PROTOCOL and meta['train_cap']==train_cap and meta['model']==model:
                pkg={'tabiclv2':'tabicl','tabpfn3':'tabpfn','smoke_logreg':'scikit-learn'}[model]
                current=version(pkg)
                if current not in ('not_installed',meta['package_version']):
                    return {'status':'CACHED_VERSION_CONFLICT','path':str(path),'cached_package_version':meta['package_version'],'current_package_version':current}
                return {'status':'CACHED_VERIFIED','path':str(path)}
        except Exception as ex:raise RuntimeError(f'Existing cache is corrupt or incompatible; DO NOT overwrite: {path}: {ex}')
    X,a,b,source=load_pair_dataset(dataset,root)
    tr=ids[:train_cap];va=np.array(split['val_ids'],dtype=int);te=np.array(split['test_ids'],dtype=int)
    ca=split['classes_a'];cb=split['classes_b']
    if set(map(str,a.iloc[tr]))!=set(ca) or set(map(str,b.iloc[tr]))!=set(cb):
        return {'status':'SKIPPED_CLASS_COVERAGE','dataset':dataset,'train_cap':train_cap}
    first=time.perf_counter()
    arrays,times=four_views(make_factory(model,device='cuda' if cuda_status()['cuda'] else 'cpu',seed=seed),X.iloc[tr],a.iloc[tr].reset_index(drop=True),b.iloc[tr].reset_index(drop=True),X.iloc[va],X.iloc[te],ca,cb)
    for k,v in arrays.items():_validate_probability_view(v)
    lookup_a={c:i for i,c in enumerate(ca)};lookup_b={c:i for i,c in enumerate(cb)}
    for role,ix in [('val',va),('test',te)]:
        arrays[f'ya_{role}']=np.array([lookup_a[str(v)] for v in a.iloc[ix]],dtype=np.int32)
        arrays[f'yb_{role}']=np.array([lookup_b[str(v)] for v in b.iloc[ix]],dtype=np.int32)
    for role,ix in [('train',tr),('val',va),('test',te)]:arrays[role+'_ids']=np.array(ix,dtype=np.int32)
    meta={'protocol':PROTOCOL,'dataset':dataset,'model':model,'artifact_kind':'smoke_only' if model=='smoke_logreg' else 'real_tfm',
          'seed':seed,'fold':fold,'train_cap':train_cap,'n_train':len(tr),'n_val':len(va),'n_test':len(te),
          'split_sha256':split['split_sha256'],'train_ids_sha':sha_obj(tr.tolist()),'val_ids_sha':sha_obj(va.tolist()),
          'test_ids_sha':sha_obj(te.tolist()),'classes_a':ca,'classes_b':cb,
          'package_version':version({'tabiclv2':'tabicl','tabpfn3':'tabpfn','smoke_logreg':'scikit-learn'}[model]),
          'checkpoint_identifier':CHECKPOINTS.get(model,'not_a_TFM'),
          'checkpoint_sha256':'NOT_VERIFIED','checkpoint_revision_confirmed':False,
          'device':cuda_status(),'timing':times,'elapsed_s':time.perf_counter()-first,'created_utc':utcnow(),
          'source':source,'n_probabilities':sum(np.size(x) for x in arrays.values()),
          'evidence_caveat':'Checkpoint filename recorded; actual weight-byte SHA256 not obtained.'}
    with tempfile.NamedTemporaryFile(dir=cdir,prefix='.npz_tmp_',suffix='.npz',delete=False) as f:tmp=Path(f.name)
    try:
        np.savez_compressed(tmp,**arrays)
        with np.load(tmp,allow_pickle=False) as test:
            if set(test.files)!=set(arrays):raise AssertionError('NPZ key mismatch')
        meta['npz_sha256']=sha_bytes(tmp.read_bytes())
        os.replace(tmp,path)
        put_json(path.with_suffix('.json'),meta)
    finally:
        if tmp.exists():tmp.unlink()
    return {'status':'REAL_MODEL_CACHE_WRITTEN' if model!='smoke_logreg' else 'SMOKE_ONLY_CACHE_WRITTEN','path':str(path),'n_train':len(tr),'n_val':len(va),'n_test':len(te),'elapsed_sec':meta['elapsed_s']}

def read_cache(path):
    path=Path(path);side=path.with_suffix('.json')
    if not side.exists():raise FileNotFoundError(f'Cache sidecar missing: {side}')
    meta=get_json(side)
    if meta.get('npz_sha256')!=sha_bytes(path.read_bytes()):raise ValueError(f'Cache checksum mismatch: {path}')
    if meta.get('protocol')!=PROTOCOL:raise ValueError('incompatible protocol')
    with np.load(path,allow_pickle=False) as npz: data={k:npz[k] for k in npz.files}
    expected=['a_val','b_val','a_given_b_val','b_given_a_val','a_test','b_test','a_given_b_test','b_given_a_test','ya_val','yb_val','ya_test','yb_test','train_ids','val_ids','test_ids']
    if set(expected)-set(data):raise ValueError('missing probability arrays/labels/ids')
    if len(data['a_val'])!=meta['n_val'] or len(data['a_test'])!=meta['n_test']:raise ValueError('cache length mismatch')
    for k in ['a_val','b_val','a_given_b_val','b_given_a_val','a_test','b_test','a_given_b_test','b_given_a_test']:_validate_probability_view(data[k])
    if len(set(data['train_ids']) & set(data['val_ids'])) or len(set(data['train_ids']) & set(data['test_ids'])) or len(set(data['test_ids']) & set(data['val_ids'])):raise ValueError('cache train/test leakage')
    if sha_obj(data['test_ids'].tolist())!=meta['test_ids_sha']:raise ValueError('test ids hash invalid')
    if sha_obj(data['val_ids'].tolist())!=meta['val_ids_sha']:raise ValueError('validation ids hash invalid')
    return data,meta

def normalize(x):
    x=np.maximum(np.asarray(x,dtype=float),1e-12)
    return x/x.sum(axis=(1,2),keepdims=True)

def implied(data,suffix):
    pa=data['a_'+suffix];pb=data['b_'+suffix];pab=data['a_given_b_'+suffix];pba=data['b_given_a_'+suffix]
    j1=np.transpose(pab,(0,2,1))*pb[:,None,:]
    j2=pa[:,:,None]*pba
    return normalize(j1),normalize(j2)

def hard_project(j,pa,pb,max_iter=150,tol=1e-9):
    q=normalize(j);pa=np.maximum(pa,1e-12);pa=pa/pa.sum(axis=1,keepdims=True);pb=np.maximum(pb,1e-12);pb=pb/pb.sum(axis=1,keepdims=True)
    for _ in range(max_iter):
        q=q*(pa[:,:,None]/np.maximum(q.sum(axis=2,keepdims=True),1e-15))
        q=q*(pb[:,None,:]/np.maximum(q.sum(axis=1,keepdims=True),1e-15))
        if max(np.max(np.abs(q.sum(axis=2)-pa)),np.max(np.abs(q.sum(axis=1)-pb)))<tol:break
    return normalize(q)

def candidates(data,suffix):
    j1,j2=implied(data,suffix)
    arith=normalize(.5*j1+.5*j2)
    geo=normalize(np.sqrt(np.maximum(j1,1e-12)*np.maximum(j2,1e-12)))
    hard=hard_project(arith,data['a_'+suffix],data['b_'+suffix])
    return {'arithmetic':arith,'raw_BA':j1,'raw_AB':j2,'geometric':geo,'hard':hard,
            'soft_25':normalize(.75*arith+.25*hard),'soft_50':normalize(.5*arith+.5*hard),
            'soft_75':normalize(.25*arith+.75*hard)}

def per_sample_metrics(j,ya,yb):
    flat=j.reshape(len(j),-1);ka,kb=j.shape[1:]
    y=np.asarray(ya)*kb+np.asarray(yb)
    nll=-np.log(np.maximum(flat[np.arange(len(y)),y],1e-12))
    brier=(flat*flat).sum(axis=1)+1-2*flat[np.arange(len(y)),y]
    return nll,brier

def select_method(cand_val,ya,yb,n_val,policy='full',beta=.5):
    m=min(int(n_val),len(ya))
    if m<4:raise ValueError('too few validation labels')
    keys={'pool':['arithmetic','geometric'],'full':list(cand_val),'regularized':list(cand_val)}[policy]
    losses={k:float(per_sample_metrics(cand_val[k][:m],ya[:m],yb[:m])[0].mean()) for k in keys}
    best=min(losses,key=losses.get)
    if policy=='regularized':
        # Predeclared finite-validation guard; no test tuning and no confidence claim.
        pen=beta*np.sqrt(2*np.log(max(len(keys),2))/m)
        if losses['arithmetic']-losses[best]<=pen:best='arithmetic'
    return best,losses

def score_cache(path,budgets=(16,32,52,103,205)):
    d,meta=read_cache(path)
    if meta.get('artifact_kind')!='real_tfm':raise ValueError('Smoke cache is not scientific real-TFM evidence')
    V=candidates(d,'val');T=candidates(d,'test')
    yaV=d['ya_val'];ybV=d['yb_val'];yaT=d['ya_test'];ybT=d['yb_test']
    j1,j2=implied(d,'test')
    tv=float((.5*np.abs(j1-j2).sum(axis=(1,2))).mean())
    rows=[]
    for v in budgets:
        if v>len(yaV):continue
        for policy in ['pool','full','regularized']:
            chosen,vloss=select_method(V,yaV,ybV,v,policy=policy)
            nll,brier=per_sample_metrics(T[chosen],yaT,ybT)
            pool_nll,pool_brier=per_sample_metrics(T['arithmetic'],yaT,ybT)
            rows.append({'dataset':meta['dataset'],'model':meta['model'],'seed':meta['seed'],'fold':meta['fold'],
                         'train_cap':meta['train_cap'],'val_budget':int(v),'policy':policy,'selected':chosen,
                         'joint_nll':float(nll.mean()),'joint_brier':float(brier.mean()),
                         'pool_nll':float(pool_nll.mean()),'pool_brier':float(pool_brier.mean()),
                         'delta_nll':float((nll-pool_nll).mean()),'delta_brier':float((brier-pool_brier).mean()),
                         'factor_tv':tv,'n_val':len(yaV),'n_test':len(yaT),'test_ids_sha':meta['test_ids_sha'],
                         'split_sha256':meta['split_sha256'],'cache_file':str(path),
                         'checkpoint_revision_confirmed':meta['checkpoint_revision_confirmed'],
                         'checkpoint_sha256':meta['checkpoint_sha256']})
    # Additional direct marginal and conditional diagnostics (proper scores).
    marginal=[]
    for label,direct,cond in [('A',d['a_test'],d['a_given_b_test'][np.arange(len(yaT)),ybT,:]),
                              ('B',d['b_test'],d['b_given_a_test'][np.arange(len(yaT)),yaT,:])]:
        y=yaT if label=='A' else ybT
        for kind,p in [('direct',direct),('conditional',cond)]:
            nll=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12))
            brier=(p*p).sum(axis=1)+1-2*p[np.arange(len(y)),y]
            marginal.append({'dataset':meta['dataset'],'model':meta['model'],'seed':meta['seed'],'fold':meta['fold'],
                             'train_cap':meta['train_cap'],'target':label,'view':kind,'nll':float(nll.mean()),'brier':float(brier.mean())})
    return rows,marginal

def analyze_real(root=None,budgets=(16,32,52,103,205)):
    root=init_dirs(root); paths=sorted((root/'real_caches').glob('*.npz'))
    if not paths:
        result={'status':'BLOCKED_NO_REAL_CACHE','real_cache_count':0,'explanation':'Run NB36 or NB37 in GPU Colab first.'}
        put_json(root/'reports'/'analysis_status.json',result);return result
    rows=[];marginal=[];bad=[]
    for p in paths:
        try:
            r,m=score_cache(p,budgets);rows.extend(r);marginal.extend(m)
        except Exception as e:bad.append({'path':str(p),'error':str(e)})
    if not rows:
        status={'status':'BLOCKED_NO_VALID_CACHE','errors':bad}
        put_json(root/'reports'/'analysis_status.json',status);return status
    df=pd.DataFrame(rows);m=pd.DataFrame(marginal)
    # For a given outer split/seed/fold the test IDs MUST stay fixed across caps and models.
    same=df.groupby(['dataset','seed','fold'])['test_ids_sha'].nunique()
    if (same>1).any():raise AssertionError(f'fixed test violation: {same[same>1].to_dict()}')
    df.to_csv(root/'reports'/'fixed_test_factorial.csv',index=False)
    m.to_csv(root/'reports'/'marginal_conditional_scores.csv',index=False)
    # Dataset-blocked inference: collapse folds and seeds/models within dataset.
    blocks=df.groupby(['dataset','train_cap','val_budget','policy'],as_index=False).agg(delta_nll=('delta_nll','mean'),delta_brier=('delta_brier','mean'))
    blocks.to_csv(root/'reports'/'dataset_blocks.csv',index=False)
    inf=[]
    rng=np.random.default_rng(20261019)
    for key,g in blocks.groupby(['train_cap','val_budget','policy']):
        deltas=g.delta_nll.to_numpy();n=len(deltas)
        ci=(np.nan,np.nan);p=np.nan
        if n>=2:
            bootstrap=rng.choice(deltas,size=(3000,n),replace=True).mean(axis=1)
            ci=np.quantile(bootstrap,[.025,.975]);
        if n>=5 and np.any(np.abs(deltas)>1e-12):
            try:p=float(wilcoxon(deltas,alternative='two-sided').pvalue)
            except ValueError:pass
        inf.append(dict(zip(['train_cap','val_budget','policy'],key))|{'datasets':n,'mean_delta_nll':float(deltas.mean()),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'wilcoxon_p_exploratory':p,
                          'inferential_note':'Exploratory unless independent datasets are sufficient; hyperparameters fixed before test.'})
    pd.DataFrame(inf).to_csv(root/'reports'/'dataset_blocked_uncertainty.csv',index=False)
    result={'status':'ANALYZED_REAL_CACHES','real_cache_count':len(paths),'valid_real_cache_count':len(paths)-len(bad),'invalid_cache_count':len(bad),
            'method_rows':len(df),'datasets':int(df.dataset.nunique()),'models':sorted(df.model.unique().tolist()),'errors':bad,
            'reports_dir':str(root/'reports'),'caveat':'Exploratory pilot and model weights not authenticated unless checkpoint hashes are separately verified.'}
    put_json(root/'reports'/'analysis_status.json',result);return result

def summarize_runtime(root=None):
    root=init_dirs(root);rows=[]
    for path in sorted((root/'real_caches').glob('*.npz')):
        try:
            _,meta=read_cache(path)
            t=meta['timing']; direct=sum(t[k]['fit_with_load_sec']+t[k]['predict_sec'] for k in ['a','b'])
            four=direct+sum(t[k]['fit_with_load_sec']+t[k]['predict_sec'] for k in ['a_given_b','b_given_a'])
            rows.append({'dataset':meta['dataset'],'model':meta['model'],'seed':meta['seed'],'fold':meta['fold'],
                         'train_cap':meta['train_cap'],'direct_two_view_wall_s':direct,
                         'four_view_wall_s':four,'four_to_two_ratio':four/direct if direct else np.nan,
                         'n_test':meta['n_test'],'timing_type':'sequential end-to-end estimator fit+predict; includes possible checkpoint loading/download',
                         'gpu':meta['device'].get('gpu')})
        except Exception as e: rows.append({'cache':str(path),'error':str(e)})
    if rows:pd.DataFrame(rows).to_csv(root/'runtime'/'cache_runtime.csv',index=False)
    return {'status':'RUNTIME_SUMMARY' if rows else 'NOT_RUN_NO_REAL_CACHES','rows':len(rows),'path':str(root/'runtime'/'cache_runtime.csv')}

def controlled_stress(root=None,repeats=8,start_repeat=0,validation_sizes=(8,16,32,64,128,256)):
    """Checkpointed CPU known-truth study, a small number of repeats per call.
    A new call appends only unique (regime,repeat,n_validation) cases. Deterministic
    per-repeat seeds make results independent of execution interruption/batching.
    """
    root=init_dirs(root)
    n=500; true=np.repeat(np.array([[[.65,.10],[.10,.15]]]),n,axis=0)
    rows=[]
    regimes=['marginal_reliable','marginal_noisy','shifted_validation']
    for regime_idx,regime in enumerate(regimes):
        for r in range(start_repeat,start_repeat+repeats):
            rng=np.random.default_rng(20261019+10001*r+regime_idx)
            noise=rng.normal(0,.13,(n,2,2));v1=normalize(np.maximum(true+noise,1e-5));v2=normalize(np.maximum(true-noise,1e-5))
            pooled=normalize(.5*v1+.5*v2)
            if regime=='marginal_reliable':
                pa=true.sum(axis=2);pb=true.sum(axis=1)
            else:
                fake=normalize(np.maximum(true+rng.normal(0,.16,(n,2,2)),1e-5))
                pa=fake.sum(axis=2);pb=fake.sum(axis=1)
            hard=hard_project(pooled,pa,pb)
            for vb in validation_sizes:
                rng_idx=np.random.default_rng(300001+1001*r+vb+regime_idx*100000)
                yval=rng_idx.choice(4,size=vb,p=true[0].reshape(-1))
                ytest=rng_idx.choice(4,size=200,p=true[0].reshape(-1))
                va_pool=pooled[:vb];va_hard=hard[:vb]
                if regime=='shifted_validation':va_hard=normalize(.8*va_hard+.2*np.roll(va_hard,1,axis=1))
                score_pool=-np.log(va_pool.reshape(vb,4)[np.arange(vb),yval]).mean()
                score_hard=-np.log(va_hard.reshape(vb,4)[np.arange(vb),yval]).mean()
                chosen='hard' if score_hard<score_pool else 'pool'
                ptest=(hard if chosen=='hard' else pooled)[-200:]
                loss=-np.log(ptest.reshape(200,4)[np.arange(200),ytest]).mean()
                loss_pool=-np.log(pooled[-200:].reshape(200,4)[np.arange(200),ytest]).mean()
                rows.append({'regime':regime,'repeat':r,'n_validation':vb,'chosen':chosen,
                             'selected_minus_pool_nll':float(loss-loss_pool),'selected_nll':float(loss),'pool_nll':float(loss_pool)})
    new_df=pd.DataFrame(rows);dest=root/'reports'/'SYNTHETIC_only_stress.csv'
    if dest.exists():
        prior=pd.read_csv(dest)
        new_df=pd.concat([prior,new_df],ignore_index=True).drop_duplicates(['regime','repeat','n_validation'],keep='last')
    new_df.sort_values(['regime','repeat','n_validation']).to_csv(dest,index=False)
    return {'status':'SYNTHETIC_COMPLETE','new_rows':len(rows),'cumulative_rows':len(new_df),'path':str(dest),'not_real_model_evidence':True}

def final_archive(root=None,strict=True):
    root=init_dirs(root);cache_paths=sorted((root/'real_caches').glob('*.npz'))
    valid=[];invalid=[];datasets=set();models=set()
    for p in cache_paths:
        try:
            _,meta=read_cache(p)
            if meta['artifact_kind']!='real_tfm':raise ValueError('not a real TFM cache')
            valid.append(str(p));datasets.add(meta['dataset']);models.add(meta['model'])
        except Exception as e:invalid.append({'file':str(p),'reason':str(e)})
    reports=['analysis_status.json','fixed_test_factorial.csv','marginal_conditional_scores.csv','dataset_blocks.csv','dataset_blocked_uncertainty.csv']
    report_ok=all((root/'reports'/s).exists() for s in reports)
    # Archive readiness deliberately stricter than successful notebook execution.
    complete=bool(len(valid)>=8 and len(datasets)>=3 and {'tabiclv2','tabpfn3'}<=models and report_ok and not invalid)
    status={'protocol':PROTOCOL,'status':'COMPLETE_PILOT_GATE' if complete else 'PARTIAL_EVIDENCE_ONLY',
            'valid_cache_count':len(valid),'datasets':sorted(datasets),'models':sorted(models),
            'invalid':invalid,'report_files_complete':report_ok,
            'complete_pilot_gate':complete,'full_paper_submission_ready':False,
            'checkpoint_bytes_authenticated':False,
            'notes':'Passing this modest pilot gate is not a full manuscript replication. Verify checkpoint weights and 10-dataset five-fold statistics separately.'}
    import zipfile
    zip_path=root/'reports'/('CoReTFM_V2_PILOT_COMPLETE.zip' if complete else 'CoReTFM_V2_PARTIAL_EVIDENCE.zip')
    status['zip_path']=str(zip_path)
    put_json(root/'reports'/'EVIDENCE_GATE.json',status)
    # Archive outputs but exclude dataset pickle caches, secrets, model weights and temp files.
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        items=[]
        for folder in ['splits','real_caches','smoke_caches','reports','runtime','failures']:
            for p in sorted((root/folder).rglob('*')):
                if p.is_file() and p.resolve()!=zip_path.resolve() and p.suffix.lower() not in ('.ckpt','.safetensors','.token','.zip'):
                    items.append(p)
        for p in items:z.write(p,p.relative_to(root).as_posix())
        manifest=[{'path':str(p.relative_to(root)),'sha256':sha_bytes(p.read_bytes()),'bytes':p.stat().st_size} for p in items]
        z.writestr('ARCHIVE_SHA256_MANIFEST.json',json.dumps(manifest,indent=2))
    return status
