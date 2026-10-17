"""Create a local metadata view of a completed, hash-synced independent GPU fit."""
from pathlib import Path
import argparse,json,hashlib,shutil
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
def main(frame,run):
 job=Path('/astro/armin/koji/syndiff/output/colab/jobs')/run;name=f'full_f{frame:04d}';src=job/'artifacts'/name
 assert (src/'DONE').exists() and (src/'execution.json').exists(),src
 cache=json.loads((job/'sync_manifest.json').read_text());required=['params.npz','flux_solved.npz','fit_meta.json','DONE','execution.json']
 hashes={}
 for key in required:
  digest=hashlib.sha256((src/key).read_bytes()).hexdigest();assert cache.get(name+'/'+key)==digest,(key,'not hash-synced');hashes[key]=digest
 out=R/'frames'/f'f{frame:04d}'/'full_fit';out.mkdir(exist_ok=True)
 for p in src.iterdir():
  if p.name=='fit_meta.json' or p.is_dir():continue
  dest=out/p.name
  if not dest.exists():dest.symlink_to(p)
 meta=json.loads((src/'fit_meta.json').read_text());meta['scene_dir']=str(out.parent/'scene');meta['colour_file']='/astro/armin/koji/syndiff/dev_runs/chroma_v2_20260928/colour/F1_s24c2k2_u.csv';meta['local_metadata_view']=dict(source=str(src),original_sha256=hashes['fit_meta.json'])
 (out/'fit_meta.json').write_text(json.dumps(meta,indent=1));(out/'provenance.json').write_text(json.dumps(dict(run=run,source=str(src),verified_hashes=hashes,stage_outcomes=meta.get('stage_outcomes'),execution=json.loads((src/'execution.json').read_text())),indent=2));print(out,meta.get('stage_outcomes'),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('frame',type=int);a.add_argument('run');v=a.parse_args();main(v.frame,v.run)
