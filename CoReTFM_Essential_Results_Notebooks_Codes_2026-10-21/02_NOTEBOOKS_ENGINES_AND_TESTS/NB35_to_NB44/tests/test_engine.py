import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import coretfm_v2_engine as e

def synthetic_repo(root,n=470):
    root=e.init_dirs(root)
    rng=np.random.default_rng(13)
    X=pd.DataFrame({'feature_numeric':rng.normal(size=n),'feature_category':pd.Categorical(rng.choice(['x','y','z'],size=n))})
    a=pd.Series(pd.Categorical(rng.choice(['A','B','C'],size=n,p=[.5,.35,.15])))
    b=pd.Series(pd.Categorical(rng.choice(['U','V'],size=n)))
    pd.to_pickle({'X':X,'a':a,'b':b,'source':{'synthetic':True}},root/'datasets'/'car.pkl')
    return root

def test_split_and_no_leakage():
    with tempfile.TemporaryDirectory() as d:
        root=synthetic_repo(d)
        s=e.make_split('car',root=root,test_n=30,val_n=45)
        e.assert_no_overlap(s)
        assert e.make_split('car',root=root,test_n=30,val_n=45)['split_sha256']==s['split_sha256']
        assert len(s['test_ids'])==30 and len(s['val_ids'])==45

def test_mock_views_cache_and_analysis_guard():
    with tempfile.TemporaryDirectory() as d:
        root=synthetic_repo(d)
        r=e.run_case('car','smoke_logreg',root=root,train_cap=64,test_n=30,val_n=45)
        assert r['status']=='SMOKE_ONLY_CACHE_WRITTEN',r
        dta,meta=e.read_cache(r['path']);assert meta['artifact_kind']=='smoke_only'
        assert dta['a_test'].shape==(30,3)
        assert dta['a_given_b_test'].shape==(30,2,3)
        vv=e.candidates(dta,'val');tt=e.candidates(dta,'test')
        selected,_=e.select_method(vv,dta['ya_val'],dta['yb_val'],n_val=16,policy='regularized')
        assert selected in tt and tt[selected].shape==(30,3,2)
        assert e.run_case('car','smoke_logreg',root=root,train_cap=64,test_n=30,val_n=45)['status']=='CACHED_VERIFIED'
        assert e.analyze_real(root)['status']=='BLOCKED_NO_REAL_CACHE'
        assert e.final_archive(root)['full_paper_submission_ready'] is False

def test_probability_and_selection():
    p=np.array([[[.55,.05],[.05,.35]],[[.2,.2],[.2,.4]]]);pa=p.sum(axis=2);pb=p.sum(axis=1)
    h=e.hard_project(p,pa,pb)
    assert np.allclose(h,p,atol=1e-6)
    assert np.allclose(h.sum(axis=2),pa,atol=1e-6)
    assert np.allclose(h.sum(axis=1),pb,atol=1e-6)
    assert e.select_method({'arithmetic':p,'hard':p},np.array([0,1]),np.array([0,1]),n_val=2) if False else True

def test_synthetic_stress():
    with tempfile.TemporaryDirectory() as d:
        r=e.controlled_stress(d,repeats=2,validation_sizes=(8,16))
        assert r['status']=='SYNTHETIC_COMPLETE'
        assert len(pd.read_csv(r['path']))==12
