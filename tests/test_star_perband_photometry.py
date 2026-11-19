import numpy as np

from syndiff_pipeline.difference_imaging.stages.gridded_epsf import load_gridded_psf_model
from syndiff_pipeline.star.perband_photometry import circular_aperture, joint_projection


def test_fractional_circular_aperture_and_clipped_sky():
    y, x = np.indices((35, 35))
    xy = (17.3, 16.55)
    r = np.hypot(x - xy[0], y - xy[1])
    data = np.full((5, 35, 35), 2.0)
    data[1:] = np.arange(1, 5)[:, None, None]
    data[0, r <= 6] += 1.3
    data[0, 3, 17] = 4000
    good = np.ones(x.shape, bool)
    noise = np.ones(x.shape)
    values, error, sky, retained = circular_aperture(data, noise, good, xy, 6, (12, 16))
    np.testing.assert_allclose(values, [1.3 * (r <= 6).sum(), *(r <= 6).sum()*np.arange(1, 5)])
    assert sky == 2 and error > 0
    assert retained < np.count_nonzero((r >= 12) & (r <= 16))
    good[17, 17] = False
    assert np.isnan(circular_aperture(data, noise, good, xy, 6, (12, 16))[0]).all()


def test_joint_projection_preserves_signed_flux_and_rejects_neighbour():
    y, x = np.indices((35, 35))
    xy = (17.1, 17.4)
    target = np.exp(-((x-xy[0])**2+(y-xy[1])**2)/3)
    target /= target.sum()
    neighbour = np.roll(target, 2, axis=1)
    good = np.ones(x.shape, bool)
    good[18, 18] = False
    amplitudes = np.array([-33.0, 1.4, 2.0, 3.0, 4.8])
    planes = amplitudes[:, None, None] * target + 7 * neighbour + 5
    measured, error = joint_projection(planes, 1+x/50, good, [target, neighbour], xy)
    np.testing.assert_allclose(measured, amplitudes, atol=1e-12)
    assert error > 0
    assert np.isnan(joint_projection(planes, 1+x/50, good, [target, target], xy)[0]).all()


def test_normalize_cropped_grid_before_interpolation(tmp_path):
    y, x = np.mgrid[-8:9, -8:9]
    grid_xy = [(10, 10), (40, 10), (10, 40), (40, 40)]
    stack = np.array([np.exp(-(x*x+y*y)/s) for s in (4, 5, 6, 7)])
    stack *= np.array([.71, .8, .9, .91])[:, None, None] * 4 / stack.sum(axis=(1, 2))[:, None, None]
    path = tmp_path/"grid.npz"
    np.savez(path, data=stack, grid_xypos=grid_xy, oversampling=2)
    model = load_gridded_psf_model(path, window_origin=(11, 31), normalize=True)
    np.testing.assert_allclose(model.data.sum(axis=(1, 2))/4, 1, atol=1e-14)
    np.testing.assert_array_equal(model.grid_xypos, np.asarray(grid_xy) - [11, 31])
