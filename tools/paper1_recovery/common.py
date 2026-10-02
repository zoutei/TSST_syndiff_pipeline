"""Fail-closed code and recipe checks for the recovery entry points."""
from pathlib import Path
import subprocess
import yaml
CODE=Path(__file__).resolve().parents[2]
OLD=Path('/astro/armin/koji/syndiff/dev_runs/paper_dataset_20261001')
V3=OLD/'template_v3'
def check_code(sha):
    import syndiff_pipeline
    assert Path(syndiff_pipeline.__file__).resolve().is_relative_to(CODE)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert actual==sha,(actual,sha)
    dirty=subprocess.check_output(['git','status','--porcelain','--','syndiff_pipeline','tools/paper1_recovery'],cwd=CODE,text=True)
    assert not dirty,dirty
    from syndiff_pipeline.template_creation.processing.band_utils import REMOVAL_CONVENTION
    assert REMOVAL_CONVENTION=='footprint_v1'
def inventory(osn):
    from syndiff_pipeline.template_creation.processing import combined_store as B,convolved_store as C
    from syndiff_pipeline.template_creation.processing.csv_utils import load_csv_data
    from syndiff_pipeline.template_creation.processing.ps1_process import extract_projection_metadata
    cfg=yaml.safe_load((V3/'C1/runs/pp_os4/config.yaml').read_text())
    root=Path(cfg['data_root']);cr=B.production_combined_recipe(cfg['stages']['ps1_process'])
    vr=C.convolved_recipe(cfg['stages']['ps1_process'])
    assert B.combined_recipe_id(cr)=='e17a198a4942aa2d'
    assert C.convolved_recipe_id(vr)=='f4d8a7b322fb8cb5'
    files=list((root/f's0024/c1/k2/mapping/oversampling_{osn}').glob('*master_skycells*.csv'))
    assert len(files)==1,files
    df=load_csv_data(str(files[0]));missing=[];total=0
    for proj in df['projection'].astype(str).unique():
        md=extract_projection_metadata(df,proj)
        for name in df.loc[df['projection'].astype(str)==proj,'NAME'].drop_duplicates():
            projection,cell=name.rsplit('.',1);total+=1
            if not C.skycell_already_canonical(root,projection,cell,cr,vr,metadata=md):missing.append(name)
    return dict(oversampling=osn,total=total,complete=total-len(missing),missing=missing)
