"""Prepare new recovery inputs and submit files. Never submit or change an old job."""
import argparse,json,shutil,subprocess
from pathlib import Path
import yaml
from common import CODE,OLD,V3
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('/astro/armin/koji/syndiff/dev_runs/paper1_recovery_20261002'))
    p.add_argument('--apply',action='store_true');a=p.parse_args();out=a.out.resolve()
    assert out.is_relative_to(Path('/astro/armin/koji/syndiff/dev_runs')) and not out.is_relative_to(OLD)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--','syndiff_pipeline','tools/paper1_recovery'],cwd=CODE,text=True)
    if a.apply and dirty:raise SystemExit('Commit owned recovery source before preparing pinned inputs')
    for f in [OLD/'configs/F1.yaml',OLD/'F1/perband/cells.json',OLD/'F1/perband/publisher_lists.json',OLD/'F1/perband/band_cells',OLD/'F1/mapping',V3/'data_root',V3/'C1/runs/pp_os1/config.yaml',V3/'C1/runs/lane_template/config.yaml']:
        assert f.exists(),f
    plan=dict(code_root=str(CODE),code_sha=sha,output=str(out),C1='OS4 cache check; OS1 variants; native and F4 templates',F1='three sequential contributions on old fitted geometry: diagnostic only',science_hold='No background selection, hp_d rebuild, fitting or final campaign submit')
    print(json.dumps(plan,indent=2))
    if not a.apply:return
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'README.md').write_text('# Paper 1 recovery\n\nStatus: prepared, not submitted.\n\nC1 resumes only after the old job is absent globally. F1 is a resource and numerical-closure probe using old fitted geometry, not a publication product. Background and removal decisions remain with the user.\n\nCode/configuration: plan.json. Activate syndiff; submit generated job.sub files to an explicitly recorded scheduler after global inventory. No automatic submission.\n\nConclusion: pending execution.\n')
    for name in ['pp_os1','lane_template']:
        src=V3/'C1/runs'/name;dst=out/'C1/runs'/name;dst.mkdir(parents=True)
        cfg=yaml.safe_load((src/'config.yaml').read_text())
        cfg.update(workspace_root=str(out/'C1/workspace'),runs_root=str(out/'C1/runs'),state_db_path=str(out/'C1/workspace/control/pipeline_state.sqlite'),skycell_wcs_csv=str(CODE/'syndiff_pipeline/resources/skycell_wcs.csv'))
        (dst/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        shutil.copyfile(src/'targets.csv',dst/'targets.csv')
        meta=json.loads((src/'run_meta.json').read_text())
        meta.update(source_config_path=str(dst/'config.yaml'),note=f'C1 resume on {sha}; v3 store retained')
        (dst/'run_meta.json').write_text(json.dumps(meta,indent=2))
    f=out/'F1_probe';(f/'perband').mkdir(parents=True)
    (f/'mapping').symlink_to(OLD/'F1/mapping',target_is_directory=True)
    for name in ['cells.json','publisher_lists.json']:shutil.copyfile(OLD/'F1/perband'/name,f/'perband'/name)
    cfg=yaml.safe_load((OLD/'configs/F1.yaml').read_text())
    cfg.update(data_root=str(V3/'data_root'),out_root=str(f))
    cfg['code']=dict(sha=sha,forward_model_root=str(CODE))
    cfg['inputs']['band_cells']=str(OLD/'F1/perband/band_cells')
    (f/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    for label,script,cpus,memory in [('C1','resume_c1.py',24,400000),('F1_probe','probe_f1.py',2,32000)]:
        logs=out/label/'condor';logs.mkdir(parents=True)
        lines=['universe = vanilla',f'executable = {CODE}/syndiff_pipeline/common/orchestration/condor_wrapper.sh',
               f'arguments = python {CODE}/tools/paper1_recovery/{script} --out {out} --sha {sha}',
               f'initialdir = {CODE}','getenv = false','should_transfer_files = NO',
               f'environment = "PYTHONPATH={CODE} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR={out}/{label}/mpl"',
               f'request_cpus = {cpus}',f'request_memory = {memory}',f'batch_name = paper1_recovery_{label}',
               f'output = {logs}/job.out',f'error = {logs}/job.err',f'log = {logs}/job.log','queue 1']
        (logs/'job.sub').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
