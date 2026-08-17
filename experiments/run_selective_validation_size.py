"""Validation-size sensitivity for Selective CoRe on heterogeneous view reliability."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from core_tfm.data.synthetic import make_multiclass_dgp
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation
from core_tfm.reconciliation.selective import select_reconciliation_policy, apply_reconciliation_policy

EPS=1e-12; WEIGHTS=(0.,.25,.5,.75,1.); LAMBDAS=(.03,.1,.3,1.,3.,10.,30.)
def norm(x,a): x=np.maximum(x,EPS); return x/x.sum(axis=a,keepdims=True)
def views(q):
    pa=q.sum(2); pb=q.sum(1); return pa,pb,norm(q/np.maximum(pb[:,None,:],EPS),1),norm(q/np.maximum(pa[:,:,None],EPS),2)
def perturb(p,s,r,a):
    z=np.log(np.maximum(p,EPS))+s*r.normal(size=p.shape); z-=z.max(axis=a,keepdims=True); z=np.exp(z); return z/z.sum(axis=a,keepdims=True)
def sample_y(q,r):
    n,ka,kb=q.shape; f=q.reshape(n,-1); z=np.array([r.choice(ka*kb,p=f[i]) for i in range(n)]); return z//kb,z%kb
def enll(p,q): return float(-np.mean(np.sum(p*np.log(np.maximum(q,EPS)),axis=(1,2))))
def family(j1,j2,pa,pb):
    o={'j1':j1,'j2':j2,'arith':arithmetic_pool(j1,j2),'geom_.5':geometric_pool(j1,j2),'mpr_.5':marginal_preserving_reconciliation(j1,j2,pa,pb).joint}
    for w in WEIGHTS:
        if abs(w-.5)>1e-15:
            o[f'geom_{w}']=geometric_pool(j1,j2,weight=w); o[f'mpr_{w}']=marginal_preserving_reconciliation(j1,j2,pa,pb,reference_weight=w).joint
        for l in LAMBDAS: o[f'soft_{w}_{l}']=soft_reconciliation(j1,j2,pa,pb,reference_weight=w,lambda_a=l,lambda_b=l).joint
    return o

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',type=int,default=40); ap.add_argument('--n',type=int,default=3200)
    ap.add_argument('--output',default='results/selective_validation_size.csv'); a=ap.parse_args(); vals=(100,250,500,800); rows=[]
    for seed in range(a.tasks):
        if seed%10==0: print('task',seed,flush=True)
        t=make_multiclass_dgp(n=a.n,d=12,k_a=3,k_b=3,gamma=2,nonlinear=True,seed=seed).true_joint
        pa,pb,agb,bga=views(t); r=np.random.default_rng(910000+seed); sig=np.exp(r.uniform(np.log(.04),np.log(.75),size=4))
        pah=perturb(pa,sig[0],r,1); pbh=perturb(pb,sig[1],r,1); agbh=perturb(agb,sig[2],r,1); bgah=perturb(bga,sig[3],r,2)
        j1=pbh[:,None,:]*agbh; j2=pah[:,:,None]*bgah; ids=r.permutation(a.n); vp=ids[:max(vals)]; te=ids[max(vals):]; ya,yb=sample_y(t[vp],r)
        jt1,jt2=j1[te],j2[te]; pat,pbt=pah[te],pbh[te]; pt=t[te]; fam=family(jt1,jt2,pat,pbt)
        oracle=min(enll(pt,q) for q in fam.values()); ar=enll(pt,arithmetic_pool(jt1,jt2))
        for nv in vals:
            idx=vp[:nv]; sel=select_reconciliation_policy(j1[idx],j2[idx],pah[idx],pbh[idx],ya[:nv],yb[:nv],weights=WEIGHTS,marginal_penalties=LAMBDAS)
            ls=enll(pt,apply_reconciliation_policy(sel,jt1,jt2,pat,pbt))
            rows.append({'task':seed,'n_val':nv,'selective_nll':ls,'full_family_oracle_nll':oracle,'oracle_regret':ls-oracle,
                'arithmetic_nll':ar,'gain_over_arithmetic':ar-ls,'selected_policy':sel.policy.name})
    df=pd.DataFrame(rows); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    sm=df.groupby('n_val',as_index=False).agg(mean_selective_nll=('selective_nll','mean'),mean_full_oracle_regret=('oracle_regret','mean'),
        median_full_oracle_regret=('oracle_regret','median'),mean_gain_over_arithmetic=('gain_over_arithmetic','mean'),win_rate_over_arithmetic=('gain_over_arithmetic',lambda x:(x>0).mean()))
    sm.to_csv(out.with_name(out.stem+'_summary.csv'),index=False); print(sm.to_string(index=False))
if __name__=='__main__': main()
