"""Hotpants target-only convolution on globally phased blocks of a padded fit grid."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from hotpants.pure.convolution import jit_make_kernel
from hotpants.pure.kernel import calculate_kernel_basis


class FrameKernel:
    def __init__(self, solution, config, fit_shape, oversample=1):
        self.factor = int(oversample)
        self.shape = tuple(fit_shape)
        self.radius = int(config.rkernel)
        self.hw = self.radius * self.factor
        self.step = int(getattr(config, "kc_step", None) or (2 * self.radius + 1)) * self.factor
        self.order = int(config.ko)
        width = 2 * self.hw + 1
        if config.use_pca:
            raise ValueError("The saved PCA kernel requires its fitted basis images")
        self.basis = np.asarray(calculate_kernel_basis(
            (width, width), [s / self.factor**2 for s in config.sigma_gauss], config.deg_fixe
        ))
        self.sol = np.asarray(solution, dtype=float).ravel()
        terms = (self.order + 1) * (self.order + 2) // 2
        if len(self.sol) < 2 + (len(self.basis) - 1) * terms:
            raise ValueError("Truncated saved Hotpants solution")
        self.blocks = {}

    def _block(self, bx, by):
        key = (by, bx)
        if key not in self.blocks:
            self.blocks[key] = jit_make_kernel(
                self.sol, self.step, self.hw, float(self.shape[1]), float(self.shape[0]),
                len(self.basis), self.order, self.basis,
                (bx * self.step + 2 * self.hw) / self.factor,
                (by * self.step + 2 * self.hw) / self.factor,
            )[::-1, ::-1]
        return self.blocks[key]

    def apply(self, image, x0, y0):
        """Drop one kernel radius per side. Origin is native padded-grid (x,y)."""
        windows = sliding_window_view(
            np.asarray(image, dtype="f4"), (2 * self.hw + 1,) * 2, axis=(-2, -1)
        )
        ny, nx = windows.shape[-4:-2]
        result = np.zeros((*windows.shape[:-4], ny, nx), dtype="f4")
        x = np.arange(nx) + x0 * self.factor
        y = np.arange(ny) + y0 * self.factor
        # Hotpants starts writing at radius r. A window index is its top-left
        # input, so its global output block starts at floor(index/step).
        valid_x = (x >= 0) & (x + 2 * self.hw < self.shape[1] * self.factor)
        valid_y = (y >= 0) & (y + 2 * self.hw < self.shape[0] * self.factor)
        for by in np.unique(y[valid_y] // self.step):
            ys = np.flatnonzero(valid_y & (y // self.step == by))
            sy = slice(ys[0], ys[-1] + 1)
            for bx in np.unique(x[valid_x] // self.step):
                xs = np.flatnonzero(valid_x & (x // self.step == bx))
                sx = slice(xs[0], xs[-1] + 1)
                result[..., sy, sx] = np.einsum(
                    "...ij,ij->...", windows[..., sy, sx, :, :], self._block(bx, by)
                )
        if self.factor > 1:
            f = self.factor
            result = result.reshape(*result.shape[:-2], ny // f, f, nx // f, f).sum(axis=(-3, -1))
        return result
