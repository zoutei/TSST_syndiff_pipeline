import numpy as np
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.wcs.gwcs_adapter import build_fixed_time_gwcs
from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import TemporalChebWcs


def _model():
    w = WCS(naxis=2)
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cd = [[-0.001, 0.0], [0.0, 0.001]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    model = TemporalChebWcs.from_reference_wcs(
        w, center=[100, 100], half_extents=[100, 100]
    )
    model.coeff_matrix[0, :] = 0.2
    model.coeff_matrix[model.n_terms, :] = -0.1
    return model


def test_gwcs_fixed_time_round_trip_matches_custom_model():
    model = _model()
    btjd = 0.5
    gw = build_fixed_time_gwcs(model, btjd)
    x = np.linspace(0.0, 200.0, 17)
    y = np.linspace(200.0, 0.0, 17)
    ra, dec = gw.pixel_to_world_values(x, y)
    got_x, got_y = gw.world_to_pixel_values(ra, dec)
    assert np.max(np.hypot(got_x - x, got_y - y)) < 1e-9
    direct_ra, direct_dec = model.pixel_to_world(x, y, btjd)
    assert np.max(np.hypot(ra - direct_ra, dec - direct_dec)) < 1e-12


def test_gwcs_inverse_is_explicit_and_pickle_safe():
    import pickle

    gw = build_fixed_time_gwcs(_model(), 0.5)
    assert gw.forward_transform.inverse is not None
    restored = pickle.loads(pickle.dumps(gw))
    ra, dec = restored.pixel_to_world_values(np.array([0.0]), np.array([0.0]))
    x, y = restored.world_to_pixel_values(ra, dec)
    assert np.max(np.hypot(x, y)) < 1e-9
