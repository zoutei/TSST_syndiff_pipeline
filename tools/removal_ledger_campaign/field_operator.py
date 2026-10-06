"""Reuse frozen L4 hybrid seam assignments and the real sparse L5 binning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.template_creation.processing.field_remap import load_gid_epoch_index,resolve_l4a_epoch_id,_find_regmap
from syndiff_pipeline.template_creation.processing.field_abutting import abutting_undirected_pairs,l4a_exact_path,load_l4b_rim_side
from syndiff_pipeline.template_creation.processing.field_hybrid_exact import compose_group_hybrid_assignment,shared_abutting_border_tess_ids
from syndiff_pipeline.template_creation.processing.hybrid_regmaps import abutting_rim_ps1_mask
from syndiff_pipeline.template_creation.processing.field_downsample import _as_tess_pixel_ids,_bin_skycell_contrib,_neighbours_by_skycell_id
from syndiff_pipeline.template_creation.processing.removal_ledger.cell import file_digest


class FrozenFieldOperator:
    def __init__(self,context,group_id=0):
        self.context=context;self.gid=int(group_id)
        self.store=Path(context['template']).parent.parent
        path=self.store/'field_mode_assembly.json'
        self.assembly=json.loads(path.read_text())
        self.grid=MappingGrid.from_mapping_dict(self.assembly['mapping_grid'])
        self.remap=Path(self.assembly['remap_root'])
        shifts=pd.read_parquet(self.remap/'template_group_shifts.parquet')
        part=shifts[shifts.group_id==self.gid]
        if part.empty:raise ValueError('No frozen shifts for this group')
        self.shifts={r.skycell:(int(r.sx_int),int(r.sy_int)) for r in part.itertuples()}
        self.epochs=load_gid_epoch_index(self.remap/'gid_epoch_index.npz')
        with fits.open(context['mapping']) as h:
            self.master=np.asarray(h[1].data).copy()
            self.name_to_id={str(name).strip():int(i) for name,i in h[2].data}
        self.id_to_name={i:n for n,i in self.name_to_id.items()}
        self.pairs=abutting_undirected_pairs(self.master)
        self.neighbours=_neighbours_by_skycell_id(self.pairs)
        self.provenance=dict(assembly_sha256=file_digest(path),master_sha256=file_digest(context['mapping']),
            shifts_sha256=file_digest(self.remap/'template_group_shifts.parquet'),
            epoch_index_sha256=file_digest(self.remap/'gid_epoch_index.npz'),group_id=self.gid)
        recorded=context.get('input_sha256',{})
        if recorded.get('mapping') and recorded['mapping']!=self.provenance['master_sha256']:
            raise ValueError('Selected mapping changed since the frozen input inventory')

    def assignment(self,cell):
        c=self.context;sx,sy=self.shifts[cell]
        p=_find_regmap(Path(c['mapping']).parent,int(c['sector']),int(c['camera']),int(c['ccd']),cell,
                       oversampling_factor=int(self.assembly['oversampling_factor']))
        frozen=_as_tess_pixel_ids(fits.getdata(p,1))
        apply_intra=bool(self.assembly['apply_intra_skycell'])
        if sx==0 and sy==0 or not apply_intra:intra=self.remap/'exact_cache_l4a/_unused_roll0_exact.npz'
        else:
            epoch=resolve_l4a_epoch_id(self.epochs,skycell=cell,group_id=self.gid,sx_int=sx,sy_int=sy)
            intra=l4a_exact_path(self.remap/'exact_cache_l4a',cell,epoch,sx,sy)
        used_rims={}
        neighbour_ids=self.neighbours.get(self.name_to_id[cell],[])
        borders={nb:shared_abutting_border_tess_ids(self.master,self.name_to_id[cell],nb)[0] for nb in neighbour_ids}
        class RimMasks(dict):
            def __missing__(self,key):
                value=abutting_rim_ps1_mask(frozen,borders[key]);self[key]=value;return value
        def load_rim(path,*,skycell_id):
            used_rims[str(path)]=file_digest(path)
            return load_l4b_rim_side(path,skycell_id=skycell_id)
        hybrid,meta=compose_group_hybrid_assignment(frozen,skycell=cell,skycell_id=self.name_to_id[cell],
            sx_int=sx,sy_int=sy,master=self.master,group_shifts=self.shifts,name_to_id=self.name_to_id,
            l4a_cache_path=intra,l4b_cache_dir=self.remap/'exact_cache_l4b',group_id=self.gid,epoch_index=self.epochs,
            hybrid_R=int(self.assembly['intra_skycell_R']),apply_intra_skycell=apply_intra,
            apply_inter_skycell=bool(self.assembly['apply_inter_skycell']),
            require_intra_skycell_cache=apply_intra and (sx!=0 or sy!=0),
            require_inter_skycell_cache=False,  # reproduce recorded pipeline fallback, verify against payload below
            pair_ids=self.pairs,id_to_name=self.id_to_name,neighbour_ids=neighbour_ids,border_ids_by_neighbour=borders,
            rim_mask_base_by_neighbour=RimMasks(),rim_cache_loader=load_rim)
        return hybrid,dict(meta,regmap_sha256=file_digest(p),rim_cache_sha256=used_rims,
                           intra_cache_sha256=file_digest(intra) if intra.exists() else None)

    def bin(self,assignment,image,mask):
        return _bin_skycell_contrib(assignment=assignment,ps1_data=image,ps1_mask=mask,sx_int=0,sy_int=0,
            base_tess_shape=tuple(self.assembly['base_tess_shape']),roi_bounds=(0,0,0,0),
            ignore_mask=int(self.assembly['ignore_mask']),mapping_grid=self.grid)

    def verify_published_contribution(self,cell,binned):
        sx,sy=self.shifts[cell]
        p=self.store/'contribs'/f'{cell}_sx{sx:+d}_sy{sy:+d}_gid{self.gid}.npz'
        if not p.exists():raise ValueError(f'Missing frozen contribution {p}')
        with np.load(p) as z:
            indices=z['indices'];flux=z['flux_sum'];count=z['count']
        if binned is None:
            if len(indices):raise ValueError('Reconstructed contribution is empty')
            return dict(exact=True,max_flux_difference=0.,file_sha256=file_digest(p))
        idx,values,counts,_=binned
        if not np.array_equal(idx,indices) or not np.array_equal(counts,count):
            raise ValueError('Hybrid mapping/counts differ from published contribution')
        err=abs(values-flux)
        return dict(exact=bool(np.array_equal(values,flux)),max_flux_difference=float(err.max()) if len(err) else 0.,
                    file_sha256=file_digest(p),pixels=len(idx))
