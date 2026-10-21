import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/CoReTFM_Research_Suite_v4/engine')
import coretfm_engine as q
rng=np.random.default_rng(143)
n=150
X=pd.DataFrame({'numeric':rng.normal(size=n),'category':pd.Categorical(rng.choice(['hi','low','medium'],n))})
a=pd.Series(pd.Categorical(np.where(X['numeric']>0,'positive','negative')))
b=pd.Series(pd.Categorical(np.where(np.arange(n)%3==0,'yes','no')))
ids=np.arange(n);rng.shuffle(ids)
tr,va,te=ids[:90],ids[90:120],ids[120:]
ca=list(map(str,a.cat.categories));cb=list(map(str,b.cat.categories))
factory=q.make_factory('smoke_logreg',device='cpu',seed=11)
views,times=q.four_views(factory,X.iloc[tr],a.iloc[tr].reset_index(drop=True),b.iloc[tr].reset_index(drop=True),X.iloc[va],X.iloc[te],ca,cb)
for key in ('a','b','a_given_b','b_given_a'):
 for suffix in ('val','test'):q._validate_probability_view(views[key+'_'+suffix])
assert views['a_given_b_val'].shape==(30,len(cb),len(ca))
assert views['b_given_a_val'].shape==(30,len(ca),len(cb))
print('FOUR_VIEW_CPU_SMOKE_PASS', {k:v.shape for k,v in views.items()})
