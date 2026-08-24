"""Task-mixture benchmark for Selective CoRe.

Each task starts from exact P*(A,B|X) and independently perturbs the four
probability views with task-specific noise levels drawn from a broad range.
Validation labels choose a policy; held-out performance is scored exactly under
P*. Oracle regret is computed against the SAME candidate family available to
selection, but using test truth and is therefore non-deployable headroom only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from core_tfm.data.synthetic import make_multiclass_dgp
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation
from core_tfm.reconciliation.selective import select_reconciliation_policy, apply_reconciliation_policy

EPS=1e-12
WEIGHTS=(0.0,0.25,0.5,0.75,1.0)
LAMBDAS=(0.03,0.1,0.3,1.0,3.0,10.0,30.0)

def normalize(x, axis):
    x=np.maximum(x,EPS); return x/x.sum(axis=axis,keepdims=True)
def views(q):
    pa=q.sum(2); pb=q.sum(1)
    return pa,pb,normalize(q/np.maximum(pb[:,None,:],EPS),1),normalize(q/np.maximum(pa[:,:,None],EPS),2)
def perturb(p,sigma,rng,axis):
    z=np.log(np.maximum(p,EPS))+sigma*rng.normal(size=p.shape); z-=z.max(axis=axis,keepdims=True)
    z=np.exp(z); return z/z.sum(axis=axis,keepdims=True)
def sample_y(q,rng):
    n,ka,kb=q.shape; flat=q.reshape(n,-1); z=np.array([rng.choice(ka*kb,p=flat[i]) for i in range(n)])
    return z//kb,z%kb
def enll(p,q): return float(-np.mean(np.sum(p*np.log(np.maximum(q,EPS)),axis=(1,2))))

def full_family(j1,j2,pa,pb):
    out={'j1':j1,'j2':j2,'arithmetic':arithmetic_pool(j1,j2),'geometric_w0.5':geometric_pool(j1,j2),
         'mpr_w0.5':marginal_preserving_reconciliation(j1,j2,pa,pb).joint}
    for w in WEIGHTS:
        if abs(w-.5)>1e-15:
            out[f'geometric_w{w:g}']=geometric_pool(j1,j2,weight=w)
            out[f'mpr_w{w:g}']=marginal_preserving_reconciliation(j1,j2,pa,pb,reference_weight=w).joint
        for lam in LAMBDAS:
            out[f'soft_w{w:g}_l{lam:g}']=soft_reconciliation(j1,j2,pa,pb,reference_weight=w,lambda_a=lam,lambda_b=lam).joint
    return out

def run_task(seed,n,d,k,gamma):
    dgp=make_multiclass_dgp(n=n,d=d,k_a=k,k_b=k,gamma=gamma,nonlinear=True,seed=seed)
    truth=dgp.true_joint; pa,pb,agb,bga=views(truth); rng=np.random.default_rng(810000+seed)
    sig=np.exp(rng.uniform(np.log(.04),np.log(.75),size=4))
    pah=perturb(pa,sig[0],rng,1); pbh=perturb(pb,sig[1],rng,1); agbh=perturb(agb,sig[2],rng,1); bgah=perturb(bga,sig[3],rng,2)
    j1=pbh[:,None,:]*agbh; j2=pah[:,:,None]*bgah
    ids=rng.permutation(n); nv=max(250,int(.30*n)); va,te=ids[:nv],ids[nv:]; ya,yb=sample_y(truth[va],rng)
    sel=select_reconciliation_policy(j1[va],j2[va],pah[va],pbh[va],ya,yb,weights=WEIGHTS,marginal_penalties=LAMBDAS)
    jt1,jt2=j1[te],j2[te]; pat,pbt=pah[te],pbh[te]; pt=truth[te]
    fixed={'j1':jt1,'j2':jt2,'arithmetic':arithmetic_pool(jt1,jt2),'geometric':geometric_pool(jt1,jt2),
           'hard_core':marginal_preserving_reconciliation(jt1,jt2,pat,pbt).joint,
           'soft_core_1':soft_reconciliation(jt1,jt2,pat,pbt,lambda_a=1,lambda_b=1).joint,
           'selective_core':apply_reconciliation_policy(sel,jt1,jt2,pat,pbt)}
    losses={m:enll(pt,q) for m,q in fixed.items()}
    fam=full_family(jt1,jt2,pat,pbt); fam_losses={m:enll(pt,q) for m,q in fam.items()}
    oracle_method=min(fam_losses,key=fam_losses.get); oracle_nll=fam_losses[oracle_method]
    row={'task':seed,'sigma_pa':sig[0],'sigma_pb':sig[1],'sigma_a_given_b':sig[2],'sigma_b_given_a':sig[3],
         'selected_policy':sel.policy.name,'selected_weight':sel.policy.weight,'selected_lambda':sel.policy.marginal_penalty,
         'oracle_full_family_method':oracle_method,'oracle_full_family_nll':oracle_nll}
    row.update({f'nll_{k}':v for k,v in losses.items()}); return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',type=int,default=100); ap.add_argument('--n',type=int,default=2200)
    ap.add_argument('--d',type=int,default=12); ap.add_argument('--k',type=int,default=3); ap.add_argument('--gamma',type=float,default=2.0)
    ap.add_argument('--output',default='results/selective_mixture.csv'); a=ap.parse_args(); rows=[]
    for seed in range(a.tasks):
        if seed%10==0: print('task',seed,flush=True)
        rows.append(run_task(seed,a.n,a.d,a.k,a.gamma))
    df=pd.DataFrame(rows); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    cols=[c for c in df if c.startswith('nll_')]; summary=[]
    for c in cols:
        x=df[c]; regret=x-df.oracle_full_family_nll
        summary.append({'method':c[4:],'mean_nll':x.mean(),'se_nll':x.std(ddof=1)/np.sqrt(len(x)),
                        'mean_full_oracle_regret':regret.mean(),'median_full_oracle_regret':regret.median(),
                        'within_1e-4_of_oracle':int((regret<=1e-4).sum())})
    sm=pd.DataFrame(summary).sort_values('mean_nll'); sm.to_csv(out.with_name(out.stem+'_summary.csv'),index=False)
    fixed=[c for c in cols if c!='nll_selective_core']; best_fixed=min(fixed,key=lambda c:df[c].mean())
    diff=df.nll_selective_core-df[best_fixed]; stat,p=wilcoxon(diff)
    tst=pd.DataFrame([{'best_fixed':best_fixed[4:],'selective_mean_minus_fixed':diff.mean(),'selective_wins':int((diff<0).sum()),
        'n_tasks':len(diff),'wilcoxon_stat':stat,'p':p,
        'selective_mean_full_oracle_regret':(df.nll_selective_core-df.oracle_full_family_nll).mean()}])
    tst.to_csv(out.with_name(out.stem+'_test.csv'),index=False)
    print(sm.to_string(index=False)); print(tst.to_string(index=False)); print('\nselection counts'); print(df.selected_policy.value_counts())
if __name__=='__main__': main()
