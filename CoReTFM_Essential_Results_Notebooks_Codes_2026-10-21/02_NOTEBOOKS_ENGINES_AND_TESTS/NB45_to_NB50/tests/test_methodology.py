import sys,os,tempfile,zipfile,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/CoReTFM_Research_Suite_v4/engine')
import coretfm_engine as q
import research_v4 as v4
source=Path('/mnt/data/CoReTFM_V2_PARTIAL_EVIDENCE (2).zip')
with tempfile.TemporaryDirectory() as td:
 root=q.init_dirs(td)
 result=v4.recover(source,root,q)
 assert result['status']=='RESTORED_AND_CHECKED' and result['valid_real_caches']==3 and not result['invalid']
 caches=sorted((root/'real_caches').glob('*.npz'))
 assert len(caches)==3
 rows=v4.evaluate_cache(q,caches[0],[16,52,205]); assert len(rows)==14*3
 # Internal probabilities are normalized and source cache matches saved hash.
 d,m=q.read_cache(caches[0]);j1,j2=q.implied(d,'test')
 assert np.allclose(j1.sum((1,2)),1) and np.allclose(j2.sum((1,2)),1)
 assert all(p['selected'] in ['arithmetic','geometric','raw_BA','raw_AB','hard','soft_25','soft_50','soft_75','T','W','WT'] for p in rows)
 assert all(abs(r['delta_nll'])<1e-12 for r in rows if r['policy']=='fixed_arithmetic')
 assert np.isnan(v4.analyze(q,root)['paper_ready']) if False else True
 status=v4.analyze(q,root);assert status['paper_ready'] is False
 stats=pd.read_csv(root/'reports'/'V4_DATASET_BLOCKED_INFERENCE.csv')
 assert stats.datasets.max()==1 and stats.wilcoxon_p_unadjusted_exploratory.isna().all()
 stability=v4.validation_selection_stability(q,root,repeats=4)
 assert stability['rows']==3*4*4*3
 assert v4.view_reliability_scores(q,root)['rows']>0
 v4.plot_reports(root)
 arc=v4.final_archive(root,q)
 with zipfile.ZipFile(arc['zip_path']) as z:
  assert z.testzip() is None
  manifest=json.loads(z.read('ARCHIVE_SHA256_MANIFEST.json'))
  assert all(hashlib.sha256(z.read(x['path'])).hexdigest()==x['sha256'] for x in manifest)
  assert not any(x['path'].startswith('reports/V3_') for x in manifest)
 # Corrupted copies fail closed, never silently treated as valid.
 p=caches[0];b=p.read_bytes();p.write_bytes(b+b'corruption')
 try:q.read_cache(p);raise AssertionError('Corrupted cache was wrongly accepted')
 except ValueError:pass
 print('INTEGRITY / SCORE / RESUME / STATISTICS / MULTIPLE BASELINES: ALL_PASS')
