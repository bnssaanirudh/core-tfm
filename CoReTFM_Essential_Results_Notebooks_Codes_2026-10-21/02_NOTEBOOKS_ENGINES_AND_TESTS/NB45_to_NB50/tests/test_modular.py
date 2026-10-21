import nbformat,os,json
from pathlib import Path
from nbclient import NotebookClient
root=Path('/mnt/data/CoReTFM_Research_Suite_v4')
os.environ['COREFM_OUT']='/mnt/data/coretfm_v4_validation'
for name in ['NB45_Recovery_Protocol_And_Coverage.ipynb','NB47_Strong_Baselines_And_Calibration.ipynb','NB48_Dataset_Blocked_Statistics_And_Figures.ipynb','NB49_Evidence_Audit_And_Download.ipynb','NB50_Optional_TabFM_Third_Model_Preflight.ipynb']:
 n=nbformat.read(root/'notebooks'/name,as_version=4)
 for cell in n.cells:
  if cell.cell_type=='code' and 'RUN_PROFILE' in cell.source:
   cell.source=cell.source.replace("RUN_PROFILE = 'BALANCED'","RUN_PROFILE = 'QUICK'")
  if cell.cell_type=='code' and name.startswith('NB50'):
   cell.source=cell.source.replace("RESULTS_DIR = Path('/content/drive/MyDrive/CoRe_TFM_Resumable_v2/reports') if Path('/content/drive/MyDrive').exists() else Path.cwd()/'tabfm_preflight'", "RESULTS_DIR = Path('/mnt/data/coretfm_v4_validation/reports')")
 result=NotebookClient(n,timeout=90,kernel_name='python3',allow_errors=False).execute()
 print(name,'PASS',sum(c.cell_type=='code' for c in result.cells))
