from pathlib import Path
import nbformat, zlib, base64, hashlib
P=Path(__file__).resolve().parent
ENGINE=(P/'coretfm_v2_engine.py').read_bytes()
PAYLOAD=base64.b64encode(zlib.compress(ENGINE,9)).decode()
SHA=hashlib.sha256(ENGINE).hexdigest()

def md(s):return nbformat.v4.new_markdown_cell(s)
def code(s):return nbformat.v4.new_code_cell(s)

BOOT=f'''# Every notebook independently reconstructs the exact same audited engine.
# In Colab, Google Drive is requested for persistent caches across disconnects.
import sys, os, hashlib, base64, zlib
from pathlib import Path
try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
except ImportError:
    pass
except Exception as ex:
    print('DRIVE_NOT_MOUNTED:', str(ex), '- outputs may be ephemeral')
root = Path(os.environ['COREFM_OUT']).expanduser() if os.environ.get('COREFM_OUT') else (Path('/content/drive/MyDrive/CoRe_TFM_Resumable_v2') if Path('/content/drive/MyDrive').exists() else Path.cwd()/'CoRe_TFM_Resumable_v2_results')
root.mkdir(parents=True, exist_ok=True)
# Base64/zlib payload is identical to the human-readable coretfm_v2_engine.py in the ZIP.
blob = zlib.decompress(base64.b64decode('{PAYLOAD}'))
assert hashlib.sha256(blob).hexdigest() == '{SHA}'
engine_file = root / 'coretfm_v2_engine.py'
if not engine_file.exists() or hashlib.sha256(engine_file.read_bytes()).hexdigest() != '{SHA}':
    engine_file.write_bytes(blob)
if str(root) not in sys.path:sys.path.insert(0,str(root))
import importlib
if 'coretfm_v2_engine' in sys.modules:
    del sys.modules['coretfm_v2_engine']
import coretfm_v2_engine as q
ROOT=q.init_dirs(root)
# Read gated-model credentials from private Colab Secrets; never print secret values.
try:
    from google.colab import userdata
    credential=userdata.get('TABPFN_TOKEN')
    if credential and not os.environ.get('TABPFN_TOKEN'):
        os.environ['TABPFN_TOKEN']=credential
except Exception:
    pass
print('OUTPUT_ROOT:', ROOT, '| CUDA:', q.cuda_status(), '| PROTOCOL:', q.PROTOCOL)
if not str(ROOT).startswith('/content/drive') and 'google.colab' in sys.modules:
    print('WARNING: NOT PERSISTENT - mount Drive or set COREFM_OUT for resumable work')
'''

def notebook(title,details,cells):
    nb=nbformat.v4.new_notebook(cells=[md('# '+title+'\n\n'+details),code(BOOT)]+[md(c[1:]) if c.startswith('§') else code(c) for c in cells],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'coretfm_protocol':'v2_independent_fixed_test'},nbformat=4,nbformat_minor=5)
    nbformat.validate(nb)
    f=P/'notebooks'/(title.split(' — ')[0]+'.ipynb');nbformat.write(nb,f);print(f.name,len(nb.cells));return f

notebook('NB35 — Environment, Tests and Frozen OpenML Splits', '''**Purpose:** CPU-compatible prerequisites and immutable train/validation/test IDs. Installs no GPU model. Run this first; if Colab has no GPU, the split preparation still works.\n\n**Scientific distinction:** These independent v2 fixed-test splits are NOT asserted identical to historical Q1 splits. Default target is Car; additional datasets can be added deliberately.''',[
'''import subprocess
required=['numpy','pandas','scipy','scikit-learn','nbformat']
from importlib import metadata
for p in required:
    try:print(p,metadata.version(p))
    except Exception:print('MISSING:',p)
print('CUDA status:',q.cuda_status())
assert q.PROTOCOL=='coretfm-v2-independent-fixed-test-2026-10-19'
''',
'''# Network-dependent; on local offline Jupyter leave this False.
RUN_OPENML_SPLITS = ('google.colab' in sys.modules)
DATASETS = ['car']
SEEDS = [23]
FOLDS = [0]
if RUN_OPENML_SPLITS:
    for name in DATASETS:
        for seed in SEEDS:
            for fold in FOLDS:
                s=q.make_split(name,seed=seed,fold=fold,root=ROOT)
                print('SPLIT_READY',name,seed,fold,'SHA256:',s['split_sha256'],'training-pool:',len(s['train_pool_ids']))
else:
    print('NOT_RUN_OPENML - set RUN_OPENML_SPLITS=True when network access is available')
''',
'''# CPU mathematical contracts - these do not need checkpoints.
import numpy as np
j=np.array([[[.40,.10],[.20,.30]],[[.15,.25],[.35,.25]]])
p_a=j.sum(axis=2);p_b=j.sum(axis=1)
repaired=q.hard_project(j,p_a,p_b)
assert np.allclose(repaired.sum(axis=2),p_a,atol=1e-7)
assert np.allclose(repaired.sum(axis=1),p_b,atol=1e-7)
assert np.allclose(repaired.sum(axis=(1,2)),1,atol=1e-8)
print('PASS: probability normalization, hard marginal preservation')
'''
])

notebook('NB36 — Real TabICLv2 Four-View Pilot', '''**Purpose:** generate one authentic four-view TabICLv2 cache on Car with frozen validation/test IDs. The notebook **skips cleanly** on CPU; it never labels a skipped GPU run as a completed experiment. Model weights are downloaded only in a supported runtime.''',[
'''DATASET='car';SEED=23;FOLD=0;TRAIN_CAP=64
RUN_REAL_MODEL=True
ALLOW_CPU_SLOW=False # explicit override only if prepared for slow CPU inference
state=q.cuda_status()
if RUN_REAL_MODEL and (state['cuda'] or ALLOW_CPU_SLOW):
    import subprocess
    subprocess.run([sys.executable,'-m','pip','install','-q','tabicl==2.2.0'],check=True)
    print('TabICL package:',q.version('tabicl'),'checkpoint:',q.CHECKPOINTS['tabiclv2'])
    try:
        result=q.run_case(DATASET,'tabiclv2',SEED,FOLD,TRAIN_CAP,root=ROOT,allow_cpu=ALLOW_CPU_SLOW)
    except Exception as ex:
        result={'status':'FAILED_REAL_MODEL_INFERENCE','error':repr(ex),'model':'tabiclv2'}
        q.put_json(ROOT/'failures'/'NB36_REAL_MODEL_FAILURE.json',result)
        print('REVIEW FAILURE JSON; this is NOT a successful experiment')
else:
    result={'status':'SKIPPED_NO_GPU','reason':'Change runtime to T4/L4 GPU; NO MODEL RUN WAS EXECUTED'}
print('NB36 RESULT:',result)
''',
'''# Required evidence must be explicit; absence is NOT a successful GPU inference.
paths=sorted((ROOT/'real_caches').glob('car__tabiclv2__seed23__fold0__train64.npz'))
if paths:
    data,meta=q.read_cache(paths[0])
    assert meta['artifact_kind']=='real_tfm'
    print('REAL FOUR-VIEW CACHE VALIDATED:',paths[0],meta['package_version'],data['a_given_b_test'].shape)
else:
    print('NOT_RUN_REAL_INFERENCE: no matching verified cache')
'''
])

notebook('NB37 — Real TabPFN-3 Four-View Pilot', '''**Purpose:** generate one TabPFN-3 cache using **explicit** `ModelVersion.V3`, not the evolving package default (which may point to TabPFN-3.5). Prior Labs checkpoint access/terms may need attention before inference. On CPU the experiment is safely skipped.''',[
'''DATASET='car';SEED=23;FOLD=0;TRAIN_CAP=64
RUN_REAL_MODEL=True
ALLOW_CPU_SLOW=False
state=q.cuda_status()
if RUN_REAL_MODEL and (state['cuda'] or ALLOW_CPU_SLOW):
    import subprocess
    subprocess.run([sys.executable,'-m','pip','install','-q','tabpfn==9.2.0'],check=True)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    assert hasattr(ModelVersion,'V3'), 'Installed version lacks explicit TabPFN-3'
    assert hasattr(TabPFNClassifier,'create_default_for_version')
    print('TabPFN package:',q.version('tabpfn'),'selected:',ModelVersion.V3)
    try:
        result=q.run_case(DATASET,'tabpfn3',SEED,FOLD,TRAIN_CAP,root=ROOT,allow_cpu=ALLOW_CPU_SLOW)
    except Exception as ex:
        result={'status':'FAILED_REAL_MODEL_INFERENCE','error':repr(ex),'model':'tabpfn3'}
        q.put_json(ROOT/'failures'/'NB37_REAL_MODEL_FAILURE.json',result)
        print('REVIEW FAILURE JSON; this is NOT a successful experiment')
else:
    result={'status':'SKIPPED_NO_GPU','reason':'Change runtime to GPU; NO MODEL RUN WAS EXECUTED'}
print('NB37 RESULT:',result)
''',
'''paths=sorted((ROOT/'real_caches').glob('car__tabpfn3__seed23__fold0__train64.npz'))
if paths:
    data,meta=q.read_cache(paths[0])
    print('REAL FOUR-VIEW CACHE VALIDATED:',paths[0],'classes B=',meta['classes_b'])
else:
    print('NOT_RUN_REAL_INFERENCE: no matching verified cache')
'''
])

notebook('NB38 — TabFM Third-Model License and API Preflight', '''**Purpose:** non-invasive preflight. TabFM weights have a **separate non-commercial license**. Nothing downloads model weights or asserts a third-model result by default. Read https://github.com/google-research/tabfm and its weight terms before enabling.''',[
'''import json, importlib.util, inspect
ALLOW_NONCOMMERCIAL_WEIGHTS=False # change only after actually reviewing model-weight terms
INSTALL_OPTIONAL_TABFM=False # third-model setup is optional, expensive, and may conflict with Torch
report={'timestamp_utc':q.utcnow(),'source':'https://github.com/google-research/tabfm',
        'weight_license':'tabfm-non-commercial-v1.0','reviewed_terms':ALLOW_NONCOMMERCIAL_WEIGHTS,
        'module_installed':bool(importlib.util.find_spec('tabfm')),
        'status':'NOT_RUN_LICENSE_CONFIRMATION_REQUIRED' if not ALLOW_NONCOMMERCIAL_WEIGHTS else 'READY_FOR_OPTIONAL_INSTALL'}
if ALLOW_NONCOMMERCIAL_WEIGHTS and INSTALL_OPTIONAL_TABFM:
    import subprocess
    subprocess.run([sys.executable,'-m','pip','install','git+https://github.com/google-research/tabfm.git#egg=tabfm[pytorch]'],check=True)
    from tabfm import TabFMClassifier
    from tabfm import tabfm_v1_0_0_pytorch
    report.update({'module_installed':True,'classifier_signature':str(inspect.signature(TabFMClassifier)),
                   'load_attribute_available':callable(tabfm_v1_0_0_pytorch.load),'status':'API_PREFLIGHT_ONLY_NO_CHECKPOINT_INFERENCE'})
q.put_json(ROOT/'reports'/'tabfm_preflight.json',report)
print(json.dumps(report,indent=2))
'''
])

notebook('NB39 — One-Job Resumable Multi-Dataset GPU Queue', '''**Purpose:** broaden evidence without long single cells. **One cache/job maximum per execution**; repeat this notebook after each completed job. Existing authentic caches are verified and skipped. Runs a two-model, three-dataset, three-training-budget crossed matrix over two seeds when repeatedly invoked.''',[
'''# Change only the experiment matrix before beginning a named research run.
DATASETS=['car','credit','wine']
MODELS=['tabiclv2','tabpfn3']
SEEDS=[23,42]
FOLDS=[0]
TRAIN_CAPS=[64,128,256]
MAX_NEW_JOBS_THIS_RUN=1
RUN_REAL_QUEUE=True
state=q.cuda_status()
if RUN_REAL_QUEUE and state['cuda']:
    import subprocess
    subprocess.run([sys.executable,'-m','pip','install','-q','tabicl==2.2.0','tabpfn'],check=True)
    candidates=[(dataset,model,seed,fold,cap) for dataset in DATASETS for model in MODELS for seed in SEEDS for fold in FOLDS for cap in TRAIN_CAPS]
    print('PLANNED_CELLS:',len(candidates),'RUN_LIMIT_THIS_EXECUTION:',MAX_NEW_JOBS_THIS_RUN)
    completed=0; statuses=[]
    for dataset,model,seed,fold,cap in candidates:
        if completed>=MAX_NEW_JOBS_THIS_RUN:break
        key=f'{dataset}__{model}__seed{seed}__fold{fold}__train{cap}'
        p=ROOT/'real_caches'/(key+'.npz')
        if p.exists():
            try:
                _,m=q.read_cache(p)
                if m['model']==model and m['train_cap']==cap:
                    continue
            except Exception as ex:
                raise RuntimeError(f'Existing cache invalid; fail closed: {p}: {ex}')
        try:
            result=q.run_case(dataset,model,seed,fold,cap,root=ROOT)
            print('JOB:',(dataset,model,seed,fold,cap),result)
            statuses.append(result)
            if result['status']=='REAL_MODEL_CACHE_WRITTEN':completed+=1
            elif result['status'].startswith('SKIPPED'):continue
            else:break
        except Exception as ex:
            err={'timestamp_utc':q.utcnow(),'job':[dataset,model,seed,fold,cap],'error':repr(ex)}
            q.put_json(ROOT/'failures'/f'{key}.json',err)
            print('FAILED_JOB_SAVED:',err)
            break
    print('NEW_JOBS_COMPLETED:',completed)
else:
    print('SKIPPED_NO_GPU - queue not run. Mount GPU and restart; existing caches remain intact.')
'''
])

notebook('NB40 — Fixed-Test Two-Factor Selection and Reliability', '''**Purpose:** analyze authentic four-view caches using a **fixed test set** while crossing independent training caps and validation budgets. Calculates full-family, pool-only and predeclared regularized selection, plus NLL, Brier, view diagnostics and dataset-blocked uncertainty. CPU-only, but **requires real caches**.''',[
'''BUDGETS=[16,32,52,103,205]
result=q.analyze_real(ROOT,budgets=BUDGETS)
print(result)
if result['status']=='ANALYZED_REAL_CACHES':
    import pandas as pd
    display(pd.read_csv(ROOT/'reports'/'dataset_blocked_uncertainty.csv').head(25))
else:
    print('NOT_RUN_REAL_ANALYSIS: no authenticated four-view model caches; rerun after NB36/37/39')
'''
])

notebook('NB41 — Evidence Integrity Gate and Honest Results ZIP', '''**Purpose:** validate cache checksums, mark evidence **partial** unless a modest pilot matrix is verified, and ZIP splits, real caches, reports and manifests. Never label a partial artifact as paper-ready. Run again after new experiments.''',[
'''result=q.final_archive(ROOT)
print('STATUS:',result['status'])
print('AUTHENTIC MODEL CACHES:',result['valid_cache_count'],'DATASETS:',result['datasets'],'MODELS:',result['models'])
print('ZIP:',result['zip_path'])
print('Full paper submission ready:',result['full_paper_submission_ready'])
'''
])

notebook('NB42 — Real End-to-End Model Cost and Overhead', '''**Purpose:** analyze measured estimator fit/predict times captured *during real TFM execution*. CPU-only aggregation; requires caches and does not pretend to produce clean kernel-only latency. True GPU inference already occurred in NB36/37/39.''',[
'''result=q.summarize_runtime(ROOT)
print(result)
if result['status']=='RUNTIME_SUMMARY':
    import pandas as pd
    d=pd.read_csv(result['path'])
    print('Note: wall timings may include checkpoint loading/download and model initialization.')
    display(d.head(20))
else:
    print('NOT_RUN_REAL_RUNTIME: run GPU model pilots first')
'''
])

notebook('NB43 — Controlled Known-Truth Shift and Safety Study', '''**Purpose:** a second experimental layer that runs entirely on CPU and exercises opposing marginal-reliability regimes, finite validation budgets and a controlled validation/test mismatch. **Synthetic results only**, not evidence that a real TFM improved.''',[
'''RUN_CONTROLLED=True
REPEATS_PER_RUN=8  # at most 8 repeats per condition; rerun for additional evidence
csv_path=ROOT/'reports'/'SYNTHETIC_only_stress.csv'
if csv_path.exists():
    import pandas as pd
    prior=pd.read_csv(csv_path)
    start_repeat=int(prior['repeat'].max())+1 if not prior.empty else 0
else: start_repeat=0
if RUN_CONTROLLED:
    result=q.controlled_stress(ROOT,repeats=REPEATS_PER_RUN,start_repeat=start_repeat,
                               validation_sizes=[8,16,32,64,128,256])
    print(result)
    import pandas as pd
    d=pd.read_csv(result['path'])
    display(d.groupby(['regime','n_validation'],as_index=False).agg(mean_delta=('selected_minus_pool_nll','mean'),
               wins=('selected_minus_pool_nll',lambda x:int((x<0).sum())),count=('repeat','count')))
'''
])

notebook('NB44 — Independent Evidence and Manuscript Claim Audit', '''**Purpose:** final CPU-only audit of experimental completeness, model representation, per-cache provenance, and publishable claim restrictions. This notebook is safe before GPU work and will state that evidence is incomplete.''',[
'''import json
from pathlib import Path
paths=sorted((ROOT/'real_caches').glob('*.npz'))
rows=[];invalid=[]
for p in paths:
    try:
        _,m=q.read_cache(p)
        rows.append({k:m[k] for k in ['dataset','model','seed','fold','train_cap','package_version','checkpoint_identifier','checkpoint_revision_confirmed','npz_sha256','created_utc']})
    except Exception as e:invalid.append({'path':str(p),'error':repr(e)})
summary={'protocol':q.PROTOCOL,'audited_at_utc':q.utcnow(),'validated_real_cache_count':len(rows),
         'invalid_cache_count':len(invalid),'models':sorted({v['model'] for v in rows}),
         'datasets':sorted({v['dataset'] for v in rows}),
         'all_checkpoint_hashes_confirmed':bool(rows and all(v['checkpoint_revision_confirmed'] for v in rows)),
         'claim_policy':'Do not assert Q1/real-model improvements from smoke/synthetic outputs or model-inference skips.',
         'draft_headline_support':'REQUIRES_MORE_REAL_MODEL_DATA',
         'invalid':invalid}
q.put_json(ROOT/'reports'/'INDEPENDENT_CLAIM_AUDIT.json',summary)
print(json.dumps(summary,indent=2))
if rows:
    import pandas as pd
    display(pd.DataFrame(rows))
'''
])
