"""Owned, bounded backfill tasks. No job control or production artifact writes."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from filelock import FileLock

from .pilot import run_cell, CODE_FILES
from .cell import validate_published


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True));os.replace(tmp,path)


def task_list(out,fields):
    """Cell ledgers are SCC-independent; preserve all consuming field names."""
    tasks={}
    for field in fields:
        inv=json.loads((out/'inventory'/f'{field}.json').read_text())
        for record in inv['cells']:
            key=(record['cell'],record['fingerprint'])
            if key not in tasks:tasks[key]=dict(field=field,cell=record['cell'],fingerprint=record['fingerprint'],fields=[])
            tasks[key]['fields'].append(field)
    return list(tasks.values())


def update_status(out,task,status,**extra):
    state=out/'campaign';state.mkdir(parents=True,exist_ok=True)
    record=dict(task,status=status,time_unix=time.time(),**extra)
    key=task.get('state_key',task['cell']+':'+task.get('fingerprint',''))
    atomic_json(state/'states'/f"{key}.json",record)
    with FileLock(str(state/'summary.lock')):
        p=state/'summary.json'
        allrows=json.loads(p.read_text()) if p.exists() else {}
        allrows[key]=record;atomic_json(p,allrows)
        totals={}
        for r in allrows.values():totals[r['status']]=totals.get(r['status'],0)+1
        expected=len(json.loads((state/'tasks.json').read_text())) if (state/'tasks.json').exists() else 0
        failures=[r for r in allrows.values() if r['status'].endswith('failed')]
        page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PS1 ledger backfill progress</title><style>body{max-width:1000px;margin:40px auto;padding:20px;font:17px/1.6 system-ui}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}</style>'
        page+='<p><a href="index.html">Implementation and pilot evidence</a></p><h1>PS1 ledger backfill</h1>'
        page+=f'<p>{expected} unique cell versions scheduled. '+html.escape(json.dumps(totals))+'.</p>'
        page+='<p>Source accounting only; production images unchanged. Completed cell tasks are not equivalent to final-template/seam certification. The campaign stops after repeated failures or low storage.</p>'
        if failures:
            page+='<h2>Failures requiring review</h2><ul>'
            for r in failures:page+='<li>'+html.escape(r['cell']+': '+r.get('error',''))+'</li>'
            page+='</ul>'
        page+='<p><a href="campaign/summary.json">Detailed task states</a> · <a href="campaign/tasks.json">Input scope</a></p>'
        tmp=out/f'.progress-{os.getpid()}.html';tmp.write_text(page);os.replace(tmp,out/'progress.html')
        (out/'index.html').touch(exist_ok=True)


def worker(out,index,expected_sha):
    here=Path(__file__).resolve().parents[4]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True).strip()
    if sha!=expected_sha:raise ValueError('Worker code pin mismatch')
    subprocess.run(['git','diff','--exit-code','--','syndiff_pipeline/template_creation/processing'],cwd=here,check=True,stdout=subprocess.DEVNULL)
    tasks=json.loads((out/'campaign/tasks.json').read_text());task=tasks[index]
    states=out/'campaign/summary.json'
    if states.exists():
        failed=sum(v['status'].endswith('failed') for v in json.loads(states.read_text()).values())
        if failed>=3:
            update_status(out,task,'deferred',error='Circuit breaker: at least three failures need review')
            return 42
    if shutil.disk_usage(out).free < 150*1024**3:
        update_status(out,task,'deferred',error='Free /astro storage below 150 GiB safety floor')
        return 42
    update_status(out,task,'running',code_sha=sha,pid=os.getpid())
    try:
        prior=out/'cell_versions'/task['cell']/task['fingerprint']/'result.json'
        done=False
        if prior.exists():
            result=json.loads(prior.read_text());manifest=validate_published(result['ledger'])
            done=(manifest['identity']['combined_fingerprint']==task['fingerprint']
                  and manifest.get('explicit_deleted_signal_saved',False)
                  and manifest.get('source_accounting_status')=='catalogue_scoped'
                  and manifest.get('metadata',{}).get('code_file_sha256')==CODE_FILES)
        if not done:run_cell(task['field'],task['cell'],out,download_raw=True)
        result=json.loads(prior.read_text());manifest=validate_published(result['ledger'])
        if manifest['identity']['combined_fingerprint']!=task['fingerprint']:raise ValueError('Wrong combined-cell generation')
        update_status(out,task,'complete',code_sha=sha,ledger=result['ledger'],replay_exact=result['replay_exact'],
                      seconds=result['seconds'],source_rows=result['source_rows'])
        return 0
    except Exception as exc:
        update_status(out,task,'failed',code_sha=sha,error=str(exc),traceback=traceback.format_exc())
        raise


def prepare(out,fields,code_pin,expected_sha,*,limit=None,max_jobs=8,with_validation=True):
    tasks=task_list(out,fields)
    if limit is not None:tasks=tasks[:limit]
    dest=out/'campaign';dest.mkdir(parents=True,exist_ok=True);(dest/'condor').mkdir(exist_ok=True)
    if (dest/'submission.json').exists():raise ValueError('Submitted campaigns are immutable; prepare a new campaign root')
    if (dest/'provenance.json').exists():
        previous=json.loads((dest/'provenance.json').read_text())
        if previous['code_sha']!=expected_sha or previous['code_pin']!=str(code_pin):
            raise ValueError('Refusing to repoint prepared jobs to another code pin')
    path=dest/'tasks.json'
    if path.exists() and json.loads(path.read_text())!=tasks:raise ValueError('Refusing to replace a different campaign task list')
    atomic_json(path,tasks)
    atomic_json(dest/'provenance.json',dict(code_pin=str(code_pin),code_sha=expected_sha,fields=fields,max_jobs=max_jobs,
        cpus_per_job=2,memory_mb=8192,measurement='streamed 2528.004 pilot: peak RSS 4038164 KiB; wall 300 s',
        scope='cell ledger backfill; separate final-template validation follows'))
    script=dest/'job.sh'
    script.write_text(f'''#!/bin/bash
set -euo pipefail
cd '{code_pin}'
export PYTHONPATH='{code_pin}'
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR='{out}/.matplotlib'
export XDG_CACHE_HOME='{out}/.cache'
exec python -m syndiff_pipeline.template_creation.processing.removal_ledger.campaign worker --out '{out}' --index "$1" --expected-sha '{expected_sha}'
''');script.chmod(0o755)
    submit=dest/'cell.sub'
    submit.write_text(f'''universe = vanilla
executable = {code_pin}/syndiff_pipeline/common/orchestration/condor_wrapper.sh
arguments = /bin/bash {script} $(task_index)
initialdir = {code_pin}
getenv = false
should_transfer_files = NO
request_cpus = 2
request_memory = 8192MB
batch_name = ps1_ledger_fix_20261002
output = {dest}/condor/cell_$(task_index).out
error = {dest}/condor/cell_$(task_index).err
log = {dest}/condor/cell_$(task_index).log
queue 1
''')
    lines=[f'MAXJOBS ledger {max_jobs}', 'MAXJOBS validation 2']
    for i,task in enumerate(tasks):
        node=f'CELL{i:05d}'
        lines += [f'JOB {node} {submit}',f'VARS {node} task_index="{i}"',f'CATEGORY {node} ledger',f'ABORT-DAG-ON {node} 42 RETURN 42']
    if with_validation:
        if limit is not None:raise ValueError('Full validation cannot be attached to a truncated task list')
        from .field_validation import historical_canonical
        from .transport import required_inputs
        import pandas as pd
        old,_=historical_canonical()
        nodes={(r['cell'],r['fingerprint']):f'CELL{i:05d}' for i,r in enumerate(tasks)}
        vscript=dest/'validation.sh'
        vscript.write_text(f'''#!/bin/bash
set -euo pipefail
cd '{code_pin}'
export PYTHONPATH='{code_pin}'
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR='{out}/.matplotlib'
export XDG_CACHE_HOME='{out}/.cache'
exec python -m syndiff_pipeline.template_creation.processing.removal_ledger.campaign validate --out '{out}' --field "$1" --cell "$2" --expected-sha '{expected_sha}'
''');vscript.chmod(0o755)
        vsub=dest/'validation.sub'
        vsub.write_text(f'''universe = vanilla
executable = {code_pin}/syndiff_pipeline/common/orchestration/condor_wrapper.sh
arguments = /bin/bash {vscript} $(field) $(cell)
initialdir = {code_pin}
getenv = false
should_transfer_files = NO
request_cpus = 2
request_memory = 16384MB
batch_name = ps1_ledger_fix_20261002
output = {dest}/condor/validate_$(field)_$(cell).out
error = {dest}/condor/validate_$(field)_$(cell).err
log = {dest}/condor/validate_$(field)_$(cell).log
queue 1
''')
        for field in fields:
            inv=json.loads((out/'inventory'/f'{field}.json').read_text());ctx=inv['original_inputs']
            table=next(Path(ctx['mapping']).parent.glob('*master_skycells_list_os4.csv'))
            mapping=pd.read_csv(table).set_index('NAME',drop=False)
            rec={r['cell']:r for r in inv['cells']};validation_nodes=[]
            for i,row in enumerate(inv['cells']):
                name=row['cell'];md=old.metadata_for_cell(mapping,name)
                deps=[nodes[(n,rec[n]['fingerprint'])] for n in required_inputs(name,md,mapping)]
                node=f'VAL_{field}_{i:05d}';validation_nodes.append(node)
                lines += [f'JOB {node} {vsub}',f'VARS {node} field="{field}" cell="{name}"',
                          f'CATEGORY {node} validation',f'PARENT {" ".join(sorted(set(deps)))} CHILD {node}',
                          f'ABORT-DAG-ON {node} 42 RETURN 42']
            node=f'REDUCE_{field}'
            lines += [f'JOB {node} {vsub}',f'VARS {node} field="{field}" cell="REDUCE"',
                      f'CATEGORY {node} validation',f'PARENT {" ".join(validation_nodes)} CHILD {node}']
    (dest/'backfill.dag').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(tasks=len(tasks),dag=str(dest/'backfill.dag'),max_jobs=max_jobs),indent=2))


def validation_worker(out,field,cell,expected_sha):
    here=Path(__file__).resolve().parents[4]
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True).strip()!=expected_sha:
        raise ValueError('Validation code pin mismatch')
    subprocess.run(['git','diff','--exit-code','--','syndiff_pipeline/template_creation/processing'],cwd=here,check=True,stdout=subprocess.DEVNULL)
    task=dict(field=field,cell=cell,state_key=f'validation:{field}:{cell}',kind='validation')
    state=out/'campaign/summary.json'
    if state.exists() and sum(v['status'].endswith('failed') for v in json.loads(state.read_text()).values())>=3:
        update_status(out,task,'validation_deferred',error='Failure circuit breaker');return 42
    if shutil.disk_usage(out).free<150*1024**3:
        update_status(out,task,'validation_deferred',error='Storage floor');return 42
    update_status(out,task,'validation_running',code_sha=expected_sha)
    try:
        from .field_validation import validate_recipient,reduce_field
        result=reduce_field(out,field) if cell=='REDUCE' else validate_recipient(out,field,cell)
        update_status(out,task,'validation_complete',result_status=result['status'],code_sha=expected_sha)
        return 0
    except Exception as exc:
        update_status(out,task,'validation_failed',error=str(exc),traceback=traceback.format_exc(),code_sha=expected_sha)
        raise


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','worker','validate']:
        q=sub.add_parser(name);q.add_argument('--out',type=Path,required=True);q.add_argument('--expected-sha',required=True)
        if name=='prepare':
            q.add_argument('--fields',nargs='+',default=['C4']);q.add_argument('--code-pin',type=Path,required=True)
            q.add_argument('--limit',type=int);q.add_argument('--max-jobs',type=int,default=8)
            q.add_argument('--no-validation',action='store_true')
        elif name=='worker':q.add_argument('--index',type=int,required=True)
        else:q.add_argument('--field',required=True);q.add_argument('--cell',required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.out,a.fields,a.code_pin,a.expected_sha,limit=a.limit,max_jobs=a.max_jobs,with_validation=not a.no_validation)
    elif a.command=='worker':raise SystemExit(worker(a.out,a.index,a.expected_sha))
    else:raise SystemExit(validation_worker(a.out,a.field,a.cell,a.expected_sha))


if __name__=='__main__':main()
