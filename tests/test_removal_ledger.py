import json
import numpy as np
import pandas as pd
import pytest
from syndiff_pipeline.template_creation.processing.removal_ledger.cell import replay_cell, exact_id, validate_published
from syndiff_pipeline.template_creation.processing import band_utils as bu
from syndiff_pipeline.template_creation.processing.removal_ledger.cell import recover_segmentation_union


IDENTITY=dict(cell='skycell.0001.001',combined_fingerprint='fixture')


def cat(x=100.,sid=123):
    return pd.DataFrame(dict(source_id=pd.array([sid],dtype='Int64'),pixel_x=[x],pixel_y=[7.],
                             tess_mag=[10.],ra=[10.],dec=[20.]))


def test_cached_union_uses_no_sep_and_records_actual_cap(monkeypatch,tmp_path):
    raw=np.ones((15,1200),np.float32);raw[0,0]=np.nan
    union=np.zeros_like(raw,bool);union[6:9,20:1100]=True
    monkeypatch.setattr(bu,'build_sep_background_segmentation',lambda *a,**k:pytest.fail('SEP rerun'))
    result,legacy,ledger=replay_cell(raw,None,np.zeros(raw.shape,np.uint16),cat(),IDENTITY,cached_union=union)
    assert result[7,100]==0 and result[7,900]==1
    assert ledger.changed_stage[7,100]==2 and ledger.changed_stage[7,900]==0
    assert ledger.regions[0]['nonfinite_pixels']==1
    sources=pd.DataFrame(dict(source_key=['gaia:123','gaia:456'],pixel_x=[100.,900.],pixel_y=[7.,7.]))
    source,links=ledger.associate_centres(sources)
    assert source.centre_status.tolist()==['centre_removed','unaffected']
    assert not ((links.source_key=='gaia:456') & (links.reason=='catalog')).any()
    path=ledger.publish(tmp_path,sources=source,associations=links)
    assert validate_published(path)['status']=='complete_cell_pixels'
    assert ledger.publish(tmp_path,sources=source,associations=links)==path
    with open(path/'regions.parquet','ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):validate_published(path)


def test_overlapping_bright_triggers_charge_flux_once():
    raw=np.ones((20,60),np.float32);union=np.ones_like(raw,bool)
    catalogue=pd.concat([cat(10,1),cat(20,2)],ignore_index=True)
    result,_,ledger=replay_cell(raw,None,None,catalogue,IDENTITY,cached_union=union)
    ops=[r for r in ledger.regions if r['reason']=='catalog']
    assert len(ops)==2 and sum(r['changed_pixels'] for r in ops)==raw.size
    assert sum(r['signed_flux_change'] for r in ops)==raw.sum()
    assert ops[1]['changed_pixels']==0 and ops[1]['support_pixels']==raw.size
    assert set(r['trigger_gaia_id'] for r in ops)=={'1','2'}


def test_saturation_only_no_catalogue_and_zero_component():
    raw=np.zeros((12,12),np.float32);raw[3:5,3:5]=1
    mask=np.zeros_like(raw,np.uint16);mask[3,3]=0x1020;mask[9,9]=0x1020
    result,_,ledger=replay_cell(raw,None,mask,None,IDENTITY,cached_union=raw>0)
    assert result.sum()==0
    sats=[r for r in ledger.regions if r['reason']=='saturation']
    assert len(sats)==2 and sorted(r['changed_pixels'] for r in sats)==[0,4]


def test_outside_centre_not_clipped_and_partial_candidate_preserved():
    raw=np.ones((12,12),np.float32);mask=np.full_like(raw,0x1020,dtype=np.uint16)
    _,_,ledger=replay_cell(raw,None,mask,None,IDENTITY,cached_union=np.ones_like(raw,bool))
    sources=pd.DataFrame(dict(source_key=['ps1det:999'],pixel_x=[-1.],pixel_y=[5.]))
    out,links=ledger.associate_centres(sources,support_radius_px=3)
    assert out.centre_status.iloc[0]=='outside'
    assert not links.centre_in_support.any()
    assert links.candidate_overlap_pixels.sum()>0
    assert links.stellar_flux_change.isna().all()


def test_replay_fails_closed_and_preserves_input():
    raw=np.ones((12,12),np.float32);before=raw.copy()
    with pytest.raises(ValueError,match='Replay differs'):
        replay_cell(raw,None,None,None,IDENTITY,cached_union=np.ones_like(raw,bool),expected=np.zeros_like(raw))
    np.testing.assert_array_equal(raw,before)


def test_ids_never_round_trip_through_float():
    identifier=2150034897230546432
    assert exact_id(str(identifier))==str(identifier)
    assert exact_id(np.int64(identifier))==str(identifier)
    with pytest.raises(ValueError,match='lossy'):exact_id(float(identifier))
    assert exact_id(-1) is None and exact_id(pd.NA) is None


def test_live_sep_and_cached_path_identical():
    y,x=np.mgrid[:120,:120]
    raw=(200*np.exp(-((x-50)**2+(y-50)**2)/30)+np.random.default_rng(1).normal(0,.1,(120,120))).astype(np.float32)
    uncert=np.full_like(raw,.1);mask=np.zeros(raw.shape,np.uint16)
    catalogue=cat(50);catalogue['pixel_y']=50.
    expected,_=bu.remove_background(raw.copy(),uncert,mask=mask,gaia_catalog_pixels=catalogue,convention='footprint_v1')
    result,_,ledger=replay_cell(raw,uncert,mask,catalogue,IDENTITY,expected=expected)
    np.testing.assert_array_equal(result,expected)
    sep=bu.build_sep_background_segmentation(raw,uncert,close_bright_mask=True)
    other,_,_=replay_cell(raw,None,mask,catalogue,IDENTITY,cached_union=(sep.segmap>0)|sep.mask_bright_stars,expected=expected)
    np.testing.assert_array_equal(other,expected)


def test_background_cache_union_recovery_is_identifiable_or_rejected():
    raw=np.array([[1.,np.nan],[2.,np.nan]])
    union=np.array([[True,True],[False,False]])
    np.testing.assert_array_equal(recover_segmentation_union(raw,np.where(union,raw,0)),union)
    with pytest.raises(ValueError,match='ambiguous'):
        recover_segmentation_union(np.array([[0.]]),np.array([[0.]]))
    with pytest.raises(ValueError,match='pure background'):
        recover_segmentation_union(np.array([[1.]]),np.array([[2.]]))
