from pathlib import Path
import os,zipfile,nbformat
from nbclient import NotebookClient
root=Path('/mnt/data/coretfm_v4_validation')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('/mnt/data/CoReTFM_V2_PARTIAL_EVIDENCE (2).zip') as z:
 for n in z.namelist():
  if n.startswith(('real_caches/','splits/')):
   t=root/n;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(z.read(n))
os.environ['COREFM_OUT']=str(root)
p='/mnt/data/CoReTFM_Research_Suite_v4/notebooks/CoReTFM_V4_ALL_IN_ONE.ipynb'
nb=nbformat.read(p,as_version=4)
for c in nb.cells:
 if c.cell_type=='code':
  c.source=c.source.replace('RUN_GPU_QUEUE = True','RUN_GPU_QUEUE = True').replace('RUN_PROFILE = \'BALANCED\'','RUN_PROFILE = \'QUICK\'')
  c.source=c.source.replace('DOWNLOAD_ZIP = False','DOWNLOAD_ZIP = False')
client=NotebookClient(nb,timeout=90,kernel_name='python3',allow_errors=False,resources={'metadata':{'path':'/mnt/data'}})
out=client.execute()
for i,c in enumerate(out.cells):
 if c.cell_type=='code':
  texts=' '.join(str(o.get('text','')) for o in c.get('outputs',[]))
  print('CELL',i,'OUTPUT',texts[:600].replace('\n',' / '))
report_path=root/'reports'/'V4_ANALYSIS_STATUS.json'
assert report_path.is_file()
import json
r=json.loads(report_path.read_text());print('ANALYSIS',r)
assert r['real_cache_count']==3
import pandas as pd
rows=pd.read_csv(root/'reports'/'V4_PER_CACHE_POLICY_RESULTS.csv')
print('Rows',len(rows),'POLICIES',rows.policy.nunique(),'caps',rows.train_cap.unique())
assert rows.policy.nunique()==14
assert len(rows)==3*5*14
assert (rows[rows.policy=='fixed_arithmetic'].delta_nll.abs()<1e-12).all()
assert (root/'reports'/'CoReTFM_V4_RESEARCH_EVIDENCE.zip').exists()
print('ALL_IN_ONE_CPU_ARCHIVE_TEST_PASS')
