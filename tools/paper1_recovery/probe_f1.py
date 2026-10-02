"""Bounded single-worker contribution probe, not a final F1 science run."""
import jax
import argparse,json,resource,time
from pathlib import Path
from common import check_code
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--sha',required=True);a=p.parse_args()
    check_code(a.sha)
    from syndiff_pipeline.forward_model.chain.config import load_config
    from syndiff_pipeline.forward_model.chain.perband.f03_band_contrib import one
    f=a.out/'F1_probe';cfg=load_config(f/'config.yaml')
    cells=sorted(json.loads((f/'perband/cells.json').read_text())['cells'])
    selected=list(dict.fromkeys(cells[i] for i in [0,len(cells)//2,len(cells)-1]));results=[]
    for name in selected:
        t=time.monotonic();_,check=one(cfg,name)
        results.append(dict(cell=name,seconds=time.monotonic()-t,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,check=check))
        (f/'probe_results.json').write_text(json.dumps(dict(purpose='memory/closure only; old fitted geometry, v3 store',code_sha=a.sha,selected=selected,results=results),indent=2))
        assert 'error' not in check,check
        assert check['nan_mismatch']==0,check
        assert check['max_rel_to_peak']<=1e-5,check
    print('Three sequential contributions passed; full-array memory and final science remain unvalidated.')
if __name__=='__main__':main()
