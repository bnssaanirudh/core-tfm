"""CoRe-TFM v4 research extension. Uses genuine q.read_cache four-view archives.
All model policy choices are validation-only; test labels are never used for selection.
The baseline engine protocol remains v2 for backwards cache compatibility.
"""
from __future__ import annotations
import hashlib, json, math, os, time, traceback, zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

METHODS = ("arithmetic", "geometric", "raw_BA", "raw_AB", "hard", "soft_25", "soft_50", "soft_75")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMPERATURES = (0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
POLICIES = ("fixed_arithmetic", "fixed_geometric", "raw_BA", "raw_AB", "hard", "soft_50",
            "pool_only", "raw_or_pool", "repair_or_pool", "full_family", "guarded_full",
            "temperature_pool", "weighted_pool", "weighted_temperature_pool")


def _write_json(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.writing')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str),encoding='utf-8');tmp.replace(p)

def recover(zipfile_path, root, q):
    """Restore authentic split/cache pairs only, never overwrite differing bytes.
    Validate member paths, old manifest, and recovered cache; reports are recomputed.
    """
    root=Path(root); report=[]; allowed={'splits','real_caches'}
    zpath=Path(zipfile_path)
    if not zpath.is_file():return {'status':'ZIP_MISSING','zip':str(zpath)}
    with zipfile.ZipFile(zpath) as z:
        bad=z.testzip()
        if bad:raise ValueError('Corrupt zip member '+bad)
        names=set(z.namelist())
        if 'ARCHIVE_SHA256_MANIFEST.json' in names:
            manifest=json.loads(z.read('ARCHIVE_SHA256_MANIFEST.json'))
            for entry in manifest:
                if entry['path'] not in names:raise ValueError('Missing signed manifest member')
                data=z.read(entry['path'])
                if hashlib.sha256(data).hexdigest()!=entry['sha256'] or len(data)!=entry['bytes']:
                    raise ValueError('Original archive manifest hash/size failed for '+entry['path'])
        for entry in z.infolist():
            if entry.is_dir():continue
            rel=Path(entry.filename)
            if (rel.is_absolute() or '..' in rel.parts or not rel.parts or
                rel.parts[0] not in allowed or rel.suffix not in ('.npz','.json')):continue
            target=(root/rel).resolve()
            if not target.is_relative_to(root.resolve()):raise ValueError('Unsafe path')
            new=z.read(entry)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                status='ALREADY_IDENTICAL' if hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(new).digest() else 'CONFLICT_KEPT_EXISTING'
            else:
                q._atomic_bytes(target,new);status='RESTORED'
            report.append({'file':rel.as_posix(),'status':status})
    invalid=[]; valid=[]
    for p in sorted((root/'real_caches').glob('*.npz')):
        try:
            _,m=q.read_cache(p)
            if m.get('artifact_kind')!='real_tfm':raise ValueError('Not real TFM')
            valid.append(str(p))
        except Exception as e: invalid.append({'file':str(p),'error':str(e)})
    for p in sorted((root/'splits').glob('*.json')):
        try:
            s=q.get_json(p);q.assert_no_overlap(s)
            if q.sha_obj({k:v for k,v in s.items() if k!='split_sha256'})!=s.get('split_sha256'):
                raise ValueError('Split signature mismatch')
        except Exception as e:invalid.append({'file':str(p),'error':str(e)})
    result={'status':'RESTORED_AND_CHECKED','actions':report,'valid_real_caches':len(valid),'invalid':invalid}
    _write_json(root/'reports'/'V4_RECOVERY.json',result)
    return result


def plan_jobs(datasets,models,seeds,folds,caps):
    """Coverage-first schedule: new datasets before extra caps, then independent splits.
    Priority cap=128 first (64 on low label budgets), then 256, 64, 512, 1024.
    """
    ds=list(dict.fromkeys(datasets));mod=list(dict.fromkeys(models))
    return [(d,m,int(s),int(f),int(c))
            for s in seeds for f in folds
            for c in sorted(caps,key=lambda n:(0 if int(n)==128 else 1 if int(n)==256 else 2, int(n)))
            for d in ds for m in mod]

def _cache_path(root,job):
    ds,m,s,f,c=job
    return Path(root)/'real_caches'/f'{ds}__{m}__seed{s}__fold{f}__train{c}.npz'

def job_coverage(q,root,jobs):
    rows=[]
    for job in jobs:
        path=_cache_path(root,job);status='MISSING'; reason=''
        if path.exists():
            try:
                _,meta=q.read_cache(path)
                if meta['artifact_kind']!='real_tfm' or (meta['dataset'],meta['model'],int(meta['seed']),int(meta['fold']),int(meta['train_cap']))!=job:
                    raise ValueError('Cache metadata identity mismatch')
                status='VERIFIED_REAL'
            except Exception as exc:status='INVALID';reason=str(exc)
        rows.append(dict(zip(['dataset','model','seed','fold','train_cap'],job)) | {'status':status,'problem':reason})
    df=pd.DataFrame(rows)
    df.to_csv(Path(root)/'reports'/'V4_JOB_COVERAGE.csv',index=False)
    return df


def run_queue(q,root,jobs,ensure_package,token_present=False,max_new=1):
    """Only one or a bounded number of NEW GPU runs. Auth errors are blockers, not successes."""
    root=Path(root);coverage=job_coverage(q,root,jobs)
    if max_new<1:raise ValueError('max_new must be positive')
    if not q.cuda_status().get('cuda'):
        status={'status':'SKIPPED_NO_CUDA','remaining':int((coverage.status!='VERIFIED_REAL').sum())}
        _write_json(root/'reports'/'V4_QUEUE_LAST.json',status);return status
    events=[];count=0
    for job in jobs:
        if count>=max_new:break
        path=_cache_path(root,job)
        if path.exists():
            try:q.read_cache(path);continue
            except Exception as e:
                events.append({'job':job,'status':'BLOCKED_CORRUPT_EXISTING_CACHE','message':str(e)});continue
        ds,model,seed,fold,cap=job
        if model=='tabpfn3' and not token_present:
            events.append({'job':job,'status':'BLOCKED_TABPFN_AUTH','message':'Accept Prior Labs license and set TABPFN_TOKEN in Colab Secrets'});continue
        try:
            pkg=ensure_package(model)
            if pkg.get('status')!='PACKAGE_READY':events.append({'job':job,**pkg});continue
            result=q.run_case(ds,model,seed,fold,cap,root=root,test_n=128,val_n=205)
            events.append({'job':job,**result})
            if result.get('status')=='REAL_MODEL_CACHE_WRITTEN':count+=1
        except Exception as exc:
            # Do not include tokens or environment contents in the persisted traceback.
            events.append({'job':job,'status':'FAILED_INFERENCE','exception_type':type(exc).__name__,
                           'message':str(exc)[:450]})
            # A failed job should not prevent valid runs for other datasets/models.
    result={'status':'QUEUE_FINISHED','new_real_caches':count,'attempted_or_blocked':len(events),'events':events,
            'note':'Re-execute this cell to resume; no test label is used for selection.'}
    _write_json(root/'reports'/'V4_QUEUE_LAST.json',result)
    job_coverage(q,root,jobs)
    return result


def _normalize_joint(x):
    x=np.maximum(np.asarray(x,dtype=np.float64),1e-12)
    return x/x.sum(axis=(1,2),keepdims=True)

def _temperature(x,t):
    x=np.maximum(np.asarray(x,dtype=np.float64),1e-12)
    z=np.log(x)/float(t);z-=z.max(axis=(1,2),keepdims=True)
    return _normalize_joint(np.exp(z))

def _means(q,p,ya,yb):
    n,b=q.per_sample_metrics(p,ya,yb)
    return float(n.mean()),float(b.mean())

def _ece_top(p,ya,yb,bins=10):
    flat=p.reshape(len(p),-1)
    conf=flat.max(axis=1);pred=flat.argmax(axis=1)
    truth=np.asarray(ya)*p.shape[2]+np.asarray(yb)
    partitions=np.minimum(np.floor(conf*bins).astype(int),bins-1)
    return float(sum(((partitions==j).sum()/len(p))*abs((pred[partitions==j]==truth[partitions==j]).mean()-conf[partitions==j].mean())
                     for j in range(bins) if (partitions==j).any()))

def evaluate_cache(q,path,budgets,penalty=0.15):
    """Validation-only candidate/policy selection; test labels accessed only AFTER policies decided."""
    d,m=q.read_cache(path)
    if m.get('artifact_kind')!='real_tfm':raise ValueError('Not real-model evidence')
    val=q.candidates(d,'val');test=q.candidates(d,'test')
    ya_v,yb_v=d['ya_val'],d['yb_val']; ya_t,yb_t=d['ya_test'],d['yb_test']
    assert set(val)==set(test)==set(METHODS)
    j1,j2=q.implied(d,'test')
    tv=float(np.mean(.5*np.abs(j1-j2).sum(axis=(1,2))))
    marg=float(.5*(np.mean(.5*np.abs(test['arithmetic'].sum(axis=2)-d['a_test']).sum(axis=1))+
                   np.mean(.5*np.abs(test['arithmetic'].sum(axis=1)-d['b_test']).sum(axis=1))))
    base_nll,base_brier=_means(q,test['arithmetic'],ya_t,yb_t)
    rows=[]
    for requested_budget in budgets:
        budget=int(requested_budget)
        if budget>len(ya_v) or budget<8:continue
        val_scores={k:_means(q,v[:budget],ya_v[:budget],yb_v[:budget])[0] for k,v in val.items()}
        subsets={
            'pool_only':['arithmetic','geometric'],
            'raw_or_pool':['arithmetic','geometric','raw_BA','raw_AB'],
            'repair_or_pool':['arithmetic','geometric','hard','soft_25','soft_50','soft_75'],
            'full_family':list(METHODS),
        }
        choices={k:(k,{}) for k in ('fixed_arithmetic','fixed_geometric','raw_BA','raw_AB','hard','soft_50')}
        choices={p:(('arithmetic' if p=='fixed_arithmetic' else 'geometric' if p=='fixed_geometric' else p),{}) for p in choices}
        for policy,names in subsets.items():choices[policy]=(min(names,key=lambda k:val_scores[k]),{})
        best=choices['full_family'][0]
        guard=penalty*math.sqrt(math.log(len(METHODS))/budget)
        choices['guarded_full']=(best if val_scores['arithmetic']-val_scores[best]>guard else 'arithmetic',{'penalty':guard})
        candidates={}
        # Temperature scaling: train only on same held-out validation pool, never fit to test.
        for t in TEMPERATURES:
            candidates[('temp',t)]=_temperature(val['arithmetic'][:budget],t)
        temp=min(TEMPERATURES,key=lambda t:_means(q,candidates[('temp',t)],ya_v[:budget],yb_v[:budget])[0])
        choices['temperature_pool']=('T',{'temperature':temp})
        best_weight=min(WEIGHTS,key=lambda w:_means(q,_normalize_joint(w*val['raw_BA'][:budget]+(1-w)*val['raw_AB'][:budget]),ya_v[:budget],yb_v[:budget])[0])
        choices['weighted_pool']=('W',{'weight_BA':best_weight})
        bestwt=min(((w,t) for w in WEIGHTS for t in TEMPERATURES),
                   key=lambda wt:_means(q,_temperature(_normalize_joint(wt[0]*val['raw_BA'][:budget]+(1-wt[0])*val['raw_AB'][:budget]),wt[1]),ya_v[:budget],yb_v[:budget])[0])
        choices['weighted_temperature_pool']=('WT',{'weight_BA':bestwt[0],'temperature':bestwt[1]})
        for policy,(pick,hyp) in choices.items():
            if pick=='T':pred=_temperature(test['arithmetic'],hyp['temperature'])
            elif pick=='W':pred=_normalize_joint(hyp['weight_BA']*test['raw_BA']+(1-hyp['weight_BA'])*test['raw_AB'])
            elif pick=='WT':pred=_temperature(_normalize_joint(hyp['weight_BA']*test['raw_BA']+(1-hyp['weight_BA'])*test['raw_AB']),hyp['temperature'])
            else:pred=test[pick]
            nll,brier=_means(q,pred,ya_t,yb_t)
            row={k:m[k] for k in ('dataset','model','seed','fold','train_cap','split_sha256','test_ids_sha','package_version','checkpoint_identifier','checkpoint_sha256')}
            row.update({'val_budget':budget,'policy':policy,'selected':pick,'selected_params':json.dumps(hyp),
                        'test_nll':nll,'test_brier':brier,'delta_nll':nll-base_nll,
                        'delta_brier':brier-base_brier,'base_nll':base_nll,'base_brier':base_brier,
                        'top1_ece_descriptive':_ece_top(pred,ya_t,yb_t),
                        'top1_accuracy':float((pred.reshape(len(pred),-1).argmax(axis=1)==ya_t*pred.shape[2]+yb_t).mean()),
                        'factor_tv':tv,'arithmetic_marginal_tv':marg,'n_test':len(ya_t),
                        'n_val_available':len(ya_v),'cache_sha256':m['npz_sha256']})
            rows.append(row)
    return rows


def _paired_dataset_blocks(df):
    # Strict paired-coverage gate: comparisons within a model/training/budget must use identical split cells.
    data=[];gcols=['model','train_cap','val_budget']
    for key,g in df.groupby(gcols):
        arithmetic=g[g.policy=='fixed_arithmetic'][['dataset','seed','fold','test_ids_sha','test_nll']].rename(columns={'test_nll':'baseline'})
        for policy,p in g.groupby('policy'):
            paired=p.merge(arithmetic,on=['dataset','seed','fold','test_ids_sha'],how='inner',validate='one_to_one')
            if len(paired)!=len(p) or len(paired)!=len(arithmetic):
                raise ValueError('Unmatched split coverage detected; will not silently aggregate')
            for ds,dsub in paired.groupby('dataset'):
                data.append(dict(zip(gcols,key))|{'policy':policy,'dataset':ds,'delta_nll':float((dsub.test_nll-dsub.baseline).mean()),'n_splits':len(dsub)})
    return pd.DataFrame(data)

def analyze(q,root,budgets=(16,32,52,103,205),primary_budget=52,primary_cap=256):
    root=Path(root); root.joinpath('reports').mkdir(parents=True,exist_ok=True)
    all_rows=[];errors=[];validmeta=[]
    for p in sorted((root/'real_caches').glob('*.npz')):
        try:
            _,m=q.read_cache(p)
            if m['artifact_kind']!='real_tfm':raise ValueError('Smoke cache excluded')
            validmeta.append(m)
            all_rows.extend(evaluate_cache(q,p,budgets))
        except Exception as e:errors.append({'file':p.name,'error':str(e)})
    if not all_rows:
        report={'status':'NOT_RUN_NO_VALID_REAL_EVIDENCE','errors':errors}
        _write_json(root/'reports'/'V4_ANALYSIS_STATUS.json',report);return report
    df=pd.DataFrame(all_rows)
    # Never silently pool results from different releases or checkpoint identities.
    mixed_package=df.groupby('model').package_version.nunique()
    mixed_checkpoint=df.groupby('model').checkpoint_identifier.nunique()
    if (mixed_package>1).any() or (mixed_checkpoint>1).any():
        raise RuntimeError('Mixed model package/checkpoint versions. Separate the result protocols before inference.')
    # Holdouts must be frozen across model and training context within a named split.
    if (df.groupby(['dataset','seed','fold'])['test_ids_sha'].nunique()>1).any():
        raise AssertionError('Mismatched test-set identities across training caps/models')
    df.to_csv(root/'reports'/'V4_PER_CACHE_POLICY_RESULTS.csv',index=False)
    blocks=_paired_dataset_blocks(df)
    blocks.to_csv(root/'reports'/'V4_DATASET_BLOCKS.csv',index=False)
    summaries=[]
    for (model,cap,budget,policy),g in blocks.groupby(['model','train_cap','val_budget','policy']):
        vals=g.delta_nll.to_numpy();n=len(vals)
        lo=hi=pvalue=np.nan
        if n>=3:
            rng=np.random.default_rng(20261019)
            boot=np.mean(rng.choice(vals,size=(4000,n),replace=True),axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
        if n>=5 and np.any(vals!=0):
            try:pvalue=float(wilcoxon(vals).pvalue)
            except ValueError:pass
        summaries.append({'model':model,'train_cap':cap,'val_budget':budget,'policy':policy,
                          'datasets':n,'delta_nll_mean':float(vals.mean()),'ci95_low':lo,'ci95_high':hi,
                          'wilcoxon_p_unadjusted_exploratory':pvalue,'wins':int((vals<0).sum()),
                          'dataset_unit_not_folds':True})
    res=pd.DataFrame(summaries)
    # Family-wise Holm adjustment on each model/cap/budget among all NON-baseline policies.
    res['holm_p_exploratory']=np.nan
    for key,ix in res.groupby(['model','train_cap','val_budget']).groups.items():
        index=[j for j in ix if res.loc[j,'policy']!='fixed_arithmetic' and np.isfinite(res.loc[j,'wilcoxon_p_unadjusted_exploratory'])]
        pvals=sorted(index,key=lambda j:res.loc[j,'wilcoxon_p_unadjusted_exploratory'])
        running=0.0;m=len(pvals)
        for jrank,j in enumerate(pvals):
            running=max(running,min(1.0,(m-jrank)*float(res.loc[j,'wilcoxon_p_unadjusted_exploratory'])))
            res.loc[j,'holm_p_exploratory']=running
    res.to_csv(root/'reports'/'V4_DATASET_BLOCKED_INFERENCE.csv',index=False)
    # Reporting completeness: independent dataset/model coverage for primary comparison.
    primary=res[(res.train_cap==primary_cap)&(res.val_budget==primary_budget)&(res.policy.isin(['full_family','temperature_pool','weighted_pool','weighted_temperature_pool','pool_only']))]
    primary.to_csv(root/'reports'/'V4_PRIMARY_COMPARISON.csv',index=False)
    provenance=[]
    for m in validmeta:
        provenance.append({k:m.get(k) for k in ['dataset','model','seed','fold','train_cap','package_version','checkpoint_identifier',
                                               'checkpoint_sha256','checkpoint_revision_confirmed','npz_sha256','split_sha256','device']})
    _write_json(root/'reports'/'V4_PROVENANCE.json',provenance)
    status={'status':'EXPLORATORY_REAL_MODEL_ANALYSIS','real_cache_count':len(validmeta),'datasets':sorted(df.dataset.unique().tolist()),
            'models':sorted(df.model.unique().tolist()),'invalid_caches':errors,
            'primary_cap':primary_cap,'primary_validation_budget':primary_budget,
            'primary_comparison_dataset_counts':{str(r['model'])+'_'+str(r['policy']):int(r['datasets']) for _,r in primary.iterrows()},
            'checkpoint_bytes_verified':bool(validmeta and all(m.get('checkpoint_revision_confirmed') and m.get('checkpoint_sha256')!='NOT_VERIFIED' for m in validmeta)),
            'paper_ready':False,'note':'Dataset-block inference exploratory; no post-hoc best-policy p-value is confirmatory.'}
    _write_json(root/'reports'/'V4_ANALYSIS_STATUS.json',status)
    return status


def plot_reports(root):
    root=Path(root)
    file=root/'reports'/'V4_PER_CACHE_POLICY_RESULTS.csv'
    if not file.exists():return 'NO_REAL_RESULTS'
    import matplotlib.pyplot as plt
    df=pd.read_csv(file)
    limited=df[(df.val_budget==52)&(df.policy.isin(['fixed_arithmetic','full_family','temperature_pool','weighted_pool','weighted_temperature_pool']))]
    if limited.empty:limited=df[(df.val_budget==df.val_budget.min())&(df.policy.isin(['fixed_arithmetic','full_family','temperature_pool']))]
    fig,ax=plt.subplots(figsize=(9,4.8))
    for (dataset,model),g in limited.groupby(['dataset','model']):
        base=g[g.policy=='fixed_arithmetic'].groupby('train_cap').test_nll.mean().sort_index()
        if not base.empty:ax.plot(base.index,base.values,marker='o',label=f'{dataset} / {model}')
    ax.set(xlabel='Training-context cap',ylabel='Arithmetic pool joint NLL',title='Real four-view model evidence: fixed-test learning curves')
    ax.grid(alpha=.25);ax.legend(fontsize=7,loc='best');fig.tight_layout()
    fp=root/'reports'/'V4_REAL_CONTEXT_NLL.png';fig.savefig(fp,dpi=180);plt.close(fig)
    return str(fp)


def final_archive(root,q,planned_jobs=None):
    root=Path(root);root.joinpath('reports').mkdir(parents=True,exist_ok=True)
    caches=[];invalid=[]
    for p in sorted((root/'real_caches').glob('*.npz')):
        try:
            _,m=q.read_cache(p)
            if m.get('artifact_kind')!='real_tfm':raise ValueError('not a TFM')
            caches.append(m)
        except Exception as e:invalid.append({'path':p.name,'error':str(e)})
    datasets={x['dataset'] for x in caches}; models={x['model'] for x in caches}
    gate={'status':'PARTIAL_EVIDENCE_ONLY','verified_real_caches':len(caches),'datasets':sorted(datasets),'models':sorted(models),
          'invalid_cache_files':invalid,'checkpoint_hashes_verified':bool(caches and all(m.get('checkpoint_revision_confirmed') for m in caches)),
          'paper_submission_ready':False,'planned_jobs':len(planned_jobs) if planned_jobs else None,
          'claim_limit':'Even sufficient cache counts cannot establish a publishable gain without dataset-level statistical analysis, matched coverage and provenance.'}
    _write_json(root/'reports'/'V4_EVIDENCE_GATE.json',gate)
    files=[]
    # Only the current V4 reports: stale V2/V3 summaries cannot masquerade as new evidence.
    for folder in ('splits','real_caches'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.json','.npz') and not p.name.endswith('.writing'):
                files.append(p)
    for p in sorted((root/'reports').glob('V4_*')):
        if p.is_file() and p.suffix.lower() in ('.json','.csv','.png','.txt','.md'):
            files.append(p)
    time_report=root/'runtime'/'cache_runtime.csv'
    if time_report.is_file():files.append(time_report)
    archive=root/'reports'/'CoReTFM_V4_RESEARCH_EVIDENCE.zip'
    # No model weights, dataset table pickle, token strings, or archived duplicate ZIPs included.
    manifest=[{'path':str(p.relative_to(root).as_posix()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
               'bytes':p.stat().st_size} for p in files]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:z.write(p,p.relative_to(root).as_posix())
        z.writestr('ARCHIVE_SHA256_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP integrity failed')
        for row in manifest:
            b=z.read(row['path'])
            if hashlib.sha256(b).hexdigest()!=row['sha256'] or len(b)!=row['bytes']:
                raise RuntimeError('Archive content hash mismatch')
    gate['zip_path']=str(archive);gate['files']=len(files);gate['zip_size_mb']=round(archive.stat().st_size/2**20,2)
    return gate


def validation_selection_stability(q,root,budgets=(16,52,103,205),repeats=30):
    """Repeated subsamples of existing real-model validation sets.
    Not new independent real-model training runs; test outcomes are diagnostic only.
    """
    root=Path(root);rows=[];invalid=[]
    for path in sorted((root/'real_caches').glob('*.npz')):
        try:
            d,m=q.read_cache(path)
            if m['artifact_kind']!='real_tfm':continue
            V=q.candidates(d,'val');T=q.candidates(d,'test')
            for budget in budgets:
                if budget>len(d['ya_val']) or budget<8:continue
                for repeat in range(repeats):
                    rng=np.random.default_rng(20261019+repeat*10007+budget*167+int(m['seed'])*17+int(m['fold']))
                    selection=rng.choice(len(d['ya_val']),size=budget,replace=False)
                    scores={k:_means(q,x[selection],d['ya_val'][selection],d['yb_val'][selection])[0] for k,x in V.items()}
                    subsets={
                        'pool_only':['arithmetic','geometric'],
                        'raw_or_pool':['arithmetic','geometric','raw_BA','raw_AB'],
                        'full_family':list(METHODS),
                    }
                    for policy,allowed in subsets.items():
                        best=min(allowed,key=scores.get)
                        score=_means(q,T[best],d['ya_test'],d['yb_test'])[0]
                        baseline=_means(q,T['arithmetic'],d['ya_test'],d['yb_test'])[0]
                        rows.append({'dataset':m['dataset'],'model':m['model'],'seed':m['seed'],'fold':m['fold'],
                                     'train_cap':m['train_cap'],'validation_budget':budget,'subsample_repeat':repeat,
                                     'policy':policy,'selected':best,'test_delta_nll':score-baseline,
                                     'note':'real-model validation-resampling diagnostic; NOT a new independent TFM run'})
        except Exception as e:invalid.append({'file':path.name,'error':str(e)})
    if rows:
        pd.DataFrame(rows).to_csv(root/'reports'/'V4_REAL_VALIDATION_SELECTION_STABILITY.csv',index=False)
    result={'status':'REAL_CACHE_VALIDATION_RESAMPLING' if rows else 'NO_REAL_CACHES','rows':len(rows),
            'real_model_inference_jobs_added':0,'invalid_caches':invalid}
    _write_json(root/'reports'/'V4_VALIDATION_STABILITY_STATUS.json',result)
    return result


def view_reliability_scores(q,root):
    """Real-model per-class and conditional-vs-direct proper scores, no model fitting."""
    root=Path(root);rows=[]
    for p in sorted((root/'real_caches').glob('*.npz')):
        try:
            d,m=q.read_cache(p)
            if m.get('artifact_kind')!='real_tfm':continue
            for target in ('a','b'):
                if target=='a':
                    labels=d['ya_test'];direct=d['a_test']; conditional=d['a_given_b_test'][np.arange(len(labels)),d['yb_test'],:]
                else:
                    labels=d['yb_test'];direct=d['b_test'];conditional=d['b_given_a_test'][np.arange(len(labels)),d['ya_test'],:]
                for kind,pred in [('direct',direct),('observed_conditioned',conditional)]:
                    for label in range(pred.shape[1]):
                        mask=labels==label
                        if not mask.any():continue
                        nll=-np.log(np.maximum(pred[mask,label],1e-12))
                        brier=(pred[mask]**2).sum(axis=1)+1-2*pred[mask,label]
                        rows.append({'dataset':m['dataset'],'model':m['model'],'seed':m['seed'],'fold':m['fold'],
                                     'train_cap':m['train_cap'],'target':target.upper(),'view':kind,'class_index':label,
                                     'support_test':int(mask.sum()),'class_nll':float(nll.mean()),
                                     'class_brier':float(brier.mean()),'rare_support_under_5':bool(mask.sum()<5),
                                     'note':'Conditional proper score conditions on observed other target; diagnostic only.'})
        except Exception:continue
    if rows:pd.DataFrame(rows).to_csv(root/'reports'/'V4_CLASSWISE_VIEW_RELIABILITY.csv',index=False)
    return {'status':'CLASSWISE_REAL' if rows else 'NO_REAL_CACHE','rows':len(rows)}
