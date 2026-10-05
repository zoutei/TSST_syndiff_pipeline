import numpy as np
import pandas as pd
import pytest
from syndiff_pipeline.template_creation.processing import canonical_cell as cc
from syndiff_pipeline.template_creation.processing.removal_ledger.transport import transport_cell,bin_regmap,fixed_domain


def fixture():
    w=h=1000
    names=['skycell.9999.000','skycell.9999.001','skycell.9999.010','skycell.9999.011']
    md=dict(projection='9999',rows={0:[(names[0],0),(names[1],1)],1:[(names[2],0),(names[3],1)]},
            cell_width=w,cell_height=h,starting_x=0,max_cells_per_row=2,span_cells=2,
            cell_dimensions={n:(w,h) for n in names})
    mapping=pd.DataFrame(dict(NAME=names,projection=['skycell.9999']*4)).set_index('NAME')
    before={n:np.ones((h,w),np.float32) for n in names}
    after={n:a.copy() for n,a in before.items()}
    return names,md,mapping,before,after


@pytest.mark.parametrize('donor',[0,1,2,3])
def test_same_projection_owner_and_overlap_closure(donor):
    names,md,mapping,b,a=fixture()
    # Deliberately contradictory cell versions; receiver's overlap must use its
    # owner, not a union of the four deletion masks.
    a[names[donor]][:, :]=0
    r=transport_cell(names[3],md,mapping,b.get,a.get,sigma=2.,radius=8,canonical_renderer=cc.canonical_cell_image)
    assert r['report']['closure_passed']
    assert r['transported'].min()>=-1e-6
    if donor==3:
        assert r['transported'][100,100]==0  # both overlaps owned elsewhere
        assert r['transported'][800,800]>.99


def test_unchanged_neighbour_is_not_omitted():
    names,md,mapping,b,a=fixture();del b[names[0]]
    with pytest.raises(ValueError,match='Missing paired'):transport_cell(names[3],md,mapping,b.get,a.get,sigma=2.,radius=8,canonical_renderer=cc.canonical_cell_image)


def test_fixed_nan_domain_and_assignment_mask():
    b=np.array([[np.nan,4.],[2.,8.]],np.float32);a=np.array([[0.,0.],[2.,np.nan]],np.float32)
    before,after,d=fixed_domain(b,a)
    assert before[0,0]==0 and np.isnan(before[1,1])
    reg=np.array([[0.,0.],[1.,1.]])
    mask=np.array([[0,4096],[0,0]],np.uint16)
    np.testing.assert_array_equal(bin_regmap(d,reg,(1,2),quality_mask=mask),[[0.,0.]])
    np.testing.assert_array_equal(bin_regmap(d,reg,(1,2)),[[4.,0.]])
    with pytest.raises(ValueError,match='Noninteger'):bin_regmap(d,reg+.2,(1,2))
