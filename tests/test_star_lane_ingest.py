from types import SimpleNamespace

import numpy as np
from syndiff_pipeline.star.star_segments import lane_offsets, _mapping_origin
from syndiff_pipeline.star.diff_runner import _resolve_frame_fits_path


def test_scc_names_and_manifest_groups(tmp_path):
    lane = tmp_path/'hp_d'
    lane.mkdir()
    target = lane/'tess123-s0026-2-2_hp_d.fits.fz'
    target.touch()
    assert _resolve_frame_fits_path(str(lane),'tess123',(26,2,2)) == str(target)
    (tmp_path/'frames.csv').write_text('group_id,group_dx,group_dy\n2,.1,-.3\n2,.1,-.3\n3,0,0\n')
    ctx = SimpleNamespace(cluster_job={}, cluster_job_path=tmp_path/'diff_job.json')
    np.testing.assert_allclose(lane_offsets(ctx), [[.1,-.3],[0,0]])


def test_master_origin_precedes_fit_grid():
    ctx=SimpleNamespace(mapping_grid=SimpleNamespace(ffi_xmin=36,ffi_ymin=-8))
    assert _mapping_origin({'XMIN':44,'YMIN':-8},ctx)==(44,-8)


def test_tic_and_local_catalogue_preserve_gaia_integer(tmp_path):
    from astropy.table import Table

    from syndiff_pipeline.star.hosts import StarHostRequest
    from syndiff_pipeline.star.identifiers import _tic_gaia_source_id, resolve_host

    sid = 1408917159066407808
    table = Table({"GAIA": [str(sid)]})
    assert _tic_gaia_source_id(table) == sid
    assert _tic_gaia_source_id(Table({"GAIA": [float(sid)]})) is None
    path = tmp_path / "gaia.csv"
    path.write_text(f"source_id,ra,dec,pmra,pmdec\n{sid},1.2,3.4,5.6,7.8\n")
    host = resolve_host(
        StarHostRequest(gaia_source_id=sid, tic_id=None, label=None),
        gaia_catalog_path=str(path), allow_remote=False,
    )
    assert host.gaia_source_id == sid
    assert (host.pmra, host.pmdec) == (5.6, 7.8)
