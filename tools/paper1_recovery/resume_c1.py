"""Resume C1 OS1 and templates after proving the old job absent across schedulers."""
import jax
import argparse,datetime as dt,json,subprocess,sys
from pathlib import Path
from common import CODE,V3,check_code,inventory
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--sha',required=True);a=p.parse_args()
    check_code(a.sha)
    q=subprocess.run(['condor_q','-global','-constraint','Owner == "kshukawa" && JobBatchName == "template_v3_C1"','-af','GlobalJobId'],text=True,capture_output=True,timeout=60)
    if q.returncode or q.stderr.strip() or q.stdout.strip():
        raise SystemExit(f'Cannot establish old C1 job absent globally: {q.stdout} {q.stderr}')
    out=a.out/'C1';before=inventory(4)
    (out/'os4_before.json').write_text(json.dumps(before,indent=2));assert not before['missing'],before
    (out/'os1_before.json').write_text(json.dumps(inventory(1),indent=2))
    for name,stage in [('pp_os1','ps1_process'),('lane_template','downsample')]:
        token='codex-recovery-'+name+'-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')
        subprocess.run([sys.executable,'-m','syndiff_pipeline.common.orchestration.run_stage','--run-id',name,'--stage',stage,'--run-dir',str(out/'runs'/name),'--target-label','s0024_c1_k2','--launch-token',token],cwd=CODE,check=True)
        if name=='pp_os1':
            after=inventory(1);(out/'os1_after.json').write_text(json.dumps(after,indent=2));assert not after['missing'],after
    from syndiff_pipeline.forward_model.chain.bootstrap import step_template
    scc=V3/'data_root/s0024/c1/k2';ffi=list((scc/'ffi').glob('tess2020118235919-s0024-1-2-*-s_ffic.fits*'));assert len(ffi)==1,ffi
    result=step_template(sector=24,camera=1,ccd=2,ffi_path=ffi[0].resolve(),mapping_dir=scc/'mapping/oversampling_4',data_root=V3/'data_root',work=out/'bootstrap',band_weights={'r':0.254,'i':0.4368,'z':0.1654,'y':0.1438},n_jobs=16)
    (out/'recovery_result.json').write_text(json.dumps(result,indent=2,default=str))
    assert not result.get('error'),result
    assert list((out/'bootstrap/templates/oversampling_4/fits').glob('*.fits*'))
    print('C1 template stages finished; final inventory and visual checks still required.')
if __name__=='__main__':main()
