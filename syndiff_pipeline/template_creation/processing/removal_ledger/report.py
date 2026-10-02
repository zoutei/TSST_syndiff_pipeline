"""Pilot evidence, with actual images beside ledger counts. No science edits."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    rows=[]
    for file in sorted((out/'pilot').glob('*/result.json')):
        result=json.loads(file.read_text());root=Path(result['ledger'])
        sources=pd.read_parquet(root/'sources.parquet');ass=pd.read_parquet(root/'associations.parquet')
        groups=sources.groupby(['catalogue','centre_status']).size().unstack(fill_value=0)
        counts=groups.reset_index().to_dict('records')
        result['counts']=counts;rows.append(result)
        cols=[c for c in ['source_key','canonical_entity_key','gaia_id','ps1_obj_id','ps1_detection_id','ra','dec','pixel_x','pixel_y','centre_status','identity_status','position_status','rPSFMag','iPSFMag','zPSFMag','yPSFMag'] if c in sources]
        sources[cols].to_csv(file.parent/'sources.csv',index=False);ass.to_csv(file.parent/'associations.csv',index=False)
    cell='skycell.2528.005';r=next(x for x in rows if x['cell']==cell);root=Path(r['ledger'])
    s=pd.read_parquet(root/'sources.parquet')
    cache=out.parent/f'epsf_atlas_20261001/dip/stagetrace/cells/{cell}.npz'
    with np.load(cache) as z:raw=z['I0'];before=z['Ibk'];after=z['I1']
    ps=s[(s.catalogue=='ps1_dr2_stack')&(s.centre_status=='centre_removed')&(~s.canonical_entity_key.str.startswith('gaia:'))].copy()
    ps=ps.drop_duplicates('ps1_obj_id')
    ps=ps[(ps.pixel_x>130)&(ps.pixel_y>130)&(ps.pixel_x<raw.shape[1]-130)&(ps.pixel_y<raw.shape[0]-130)]
    if 'rPSFMag' in ps:ps=ps[(pd.to_numeric(ps.rPSFMag,errors='coerce')>18)&(pd.to_numeric(ps.rPSFMag,errors='coerce')<23)].sort_values('rPSFMag')
    selected=[]
    for row in ps.itertuples():
        if all(np.hypot(row.pixel_x-o.pixel_x,row.pixel_y-o.pixel_y)>150 for o in selected):selected.append(row)
        if len(selected)==4:break
    fig,axs=plt.subplots(len(selected),4,figsize=(13,3.1*len(selected)),squeeze=False)
    for i,row in enumerate(selected):
        x,y=round(row.pixel_x),round(row.pixel_y);rad=100;sl=np.s_[y-rad:y+rad+1,x-rad:x+rad+1];ext=[x-rad,x+rad,y-rad,y+rad]
        planes=[raw[sl],before[sl],after[sl],np.nan_to_num(before[sl])-np.nan_to_num(after[sl])]
        for j,(plane,title) in enumerate(zip(planes,['Original PS1','Before explicit removal','Published after removal','Explicit deletion'])):
            ax=axs[i,j];im=ax.imshow(plane,origin='lower',extent=ext,cmap='magma',norm=SymLogNorm(linthresh=.02,vmin=-.1,vmax=10));ax.set_facecolor('grey')
            gaia=s[(s.catalogue=='gaia_dr3')&(abs(s.pixel_x-x)<rad)&(abs(s.pixel_y-y)<rad)]
            ax.scatter(gaia.pixel_x,gaia.pixel_y,c='lime',s=35,marker='+');ax.plot(row.pixel_x,row.pixel_y,'cx',ms=9)
            if i==0:ax.set_title(title,fontsize=10)
        axs[i,0].set_ylabel(f'PS1 {row.ps1_obj_id}\nr={row.rPSFMag:.2f}; no confirmed Gaia ID',fontsize=8)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.6,label='Combined PS1 flux / pixel; shared scale')
    fig.suptitle('New ledger: PS1 detections at explicitly deleted centre pixels\nCyan = PS1 detection; green = deeper Gaia. Catalogue detections are not automatically certified stars.')
    (out/'figs').mkdir(exist_ok=True)
    fig.savefig(out/'figs/ps1_only_examples.png',dpi=135,bbox_inches='tight');plt.close(fig)
    transport=out/'transport'/cell/'report.json'
    if transport.exists():
        summary=json.loads(transport.read_text());z=np.load(transport.parent/'maps.npz')
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        for ax,k,title in zip(axs,['before','after','transported','residual'],['Restored source cell through seams','Published reconstruction','Transported deletion','Before − after − deletion']):
            res=k=='residual';v=z[k]
            im=ax.imshow(v,origin='lower',cmap='RdBu_r' if res else 'magma',norm=None if res else SymLogNorm(linthresh=.02,vmin=0,vmax=10),vmin=-1e-6 if res else None,vmax=1e-6 if res else None)
            ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,fraction=.045)
        fig.suptitle('C4 2528.005: original v2 ownership + cross-projection full-halo correction\nOne-cell intervention, all neighbouring cells fixed; this is not whole-template certification.')
        fig.tight_layout();fig.savefig(out/'figs/seam_closure.png',dpi=140);plt.close(fig)
    (out/'pilot_summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps([dict(cell=r['cell'],replay_exact=r['replay_exact'],source_rows=r['source_rows']) for r in rows],indent=2))


if __name__=='__main__':main()
