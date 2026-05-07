"""
Adaptive median-filter background smoother for TESS FFI background cubes.

Vendored verbatim from:
  TESSreduce/tessreduce/adaptive_background.py
(SynDiff does not depend on the tessreduce package; this file is copied from the
in-repo TESSreduce subtree.)

Public API
----------
AdaptiveBackground(data, time, sector, camera, data_path=None)
    Main class. Fetches Earth/Moon angle vectors from TESSVectors (local or
    remote) and exposes a .smooth() method.

get_tessvectors(sector, camera, data_path=None)
    Retrieve a TESSVectors CSV as a pandas DataFrame, checking data_path
    before falling back to the remote HEASARC server.

adaptive_medfilt_3d(data, time, ...)
    Low-level 3-D adaptive median filter. Can be used directly with
    pre-fetched angle arrays or without angles.
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, percentile_filter, uniform_filter
from scipy.signal import savgol_filter
from joblib import Parallel, delayed


# ── TESSVectors helpers ────────────────────────────────────────────────────────

_TESSVECTORS_REMOTE = (
    "https://heasarc.gsfc.nasa.gov/docs/tess/data/TESSVectors/Vectors/"
    "FFI_Cadence/TessVectors_S{sector:03d}_C{camera}_FFI.csv"
)
_TESSVECTORS_FNAME = "TessVectors_S{sector:03d}_C{camera}_FFI.csv"


def get_tessvectors(sector, camera, data_path=None):
    """Return a TESSVectors DataFrame for *sector* / *camera*.

    Parameters
    ----------
    sector : int
    camera : int  (1–4)
    data_path : str or None
        Local directory to check for pre-downloaded TESSVectors CSV files
        before attempting a remote download.

    Returns
    -------
    pandas.DataFrame
        Columns include ``MidTime`` (BTJD), ``Earth_Camera_Angle``,
        ``Moon_Camera_Angle``.
    """
    fname = _TESSVECTORS_FNAME.format(sector=sector, camera=camera)

    if data_path is not None:
        local = os.path.join(data_path, fname)
        if os.path.isfile(local):
            return pd.read_csv(local, comment='#', index_col=False)

    url = _TESSVECTORS_REMOTE.format(sector=sector, camera=camera)
    try:
        df = pd.read_csv(url, comment='#', index_col=False)
    except:
        df = None
    return df


def _interpolate_angles(time_mjd, df):
    """Interpolate Earth/Moon camera angles onto *time_mjd* (MJD).

    TESSVectors MidTime is BTJD (BJD − 2457000); MJD = BTJD + 57000.
    """
    btjd = time_mjd - 56999.5
    vec_t = df['MidTime'].values
    earth = np.interp(btjd, vec_t, df['Earth_Camera_Angle'].values)
    moon = np.interp(btjd, vec_t, df['Moon_Camera_Angle'].values)
    return earth, moon


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_odd(x):
    x = max(int(x), 3)
    return x if x % 2 == 1 else x + 1


def _get_segments(time, gap_thresh):
    dt = np.diff(time)
    median_dt = np.median(dt)
    breaks = np.where(dt > gap_thresh * median_dt)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(time)]])
    return list(zip(starts.tolist(), ends.tolist()))


def _block_reduce(arr, bs):
    """Block-average (T, X, Y) spatially by factor *bs*, discarding edge pixels
    that don't fill a complete block."""
    T, X, Y = arr.shape
    Xd, Yd = X // bs, Y // bs
    return (arr[:, :Xd * bs, :Yd * bs]
            .reshape(T, Xd, bs, Yd, bs)
            .mean(axis=(2, 4))
            .astype(arr.dtype))


def _upsample_nearest(arr, X, Y, bs):
    """Nearest-neighbour upsample (T, Xd, Yd) back to (T, X, Y)."""
    up = np.repeat(np.repeat(arr, bs, axis=1), bs, axis=2)
    if up.shape[1] < X:
        up = np.concatenate([up, up[:, -1:, :]], axis=1)
    if up.shape[2] < Y:
        up = np.concatenate([up, up[:, :, -1:]], axis=2)
    return up[:, :X, :Y]


# ── Core adaptive filter ───────────────────────────────────────────────────────

def adaptive_medfilt_3d(
    data,
    time=None,
    gap_thresh=3.0,
    w_min=3,
    w_max=51,
    grad_smooth_window=11,
    low_pct=5,
    high_pct=80,
    per_pixel_norm=True,
    n_levels=7,
    n_jobs=1,
    metric='deviation',
    coarse_windows=(11, 21, 51, 101),
    combined_weight=0.5,
    local_std_window=21,
    local_norm_window=201,
    brightness_sigma=2.0,
    window_smooth_size=1,
    earth_angle=None,
    moon_angle=None,
    scatter_angle_thresh=50.0,
    block_size=1,
    sigma_clip=5.0,
):
    """Adaptive median filter for a (T, X, Y) background cube.

    Parameters
    ----------
    data : array (T, X, Y)
    time : array (T,), optional
        Frame mid-times. Defaults to integer frame indices.
    gap_thresh : float
        Gap declared where ``dt > gap_thresh * median_dt``. Smoothing never
        crosses a gap boundary.
    w_min, w_max : int
        Min / max (odd) filter window sizes along the time axis.
    metric : {'deviation', 'gradient', 'local_std', 'combined'}
    coarse_windows : tuple of int
        Baseline window sizes for the deviation metric.
    brightness_sigma : float
        Sigma for sigma-clipping the frame-mean distribution to define the
        background brightness level.
    earth_angle, moon_angle : array (T,) or None
        Camera angles (degrees) for scattered-light masking.
    scatter_angle_thresh : float
        Frames with ``min(earth, moon) < scatter_angle_thresh`` are treated as
        scatter-contaminated; all deviation scales are zeroed (→ w_max) in
        scatter-free faint frames when scatter is detected.
    n_jobs : int
        Parallel jobs. ``-1`` uses all available cores.

    Returns
    -------
    smoothed : array (T, X, Y)
    windows : array (T, X, Y), int
    variability : array (T, X, Y), float
    windows_pre_smooth : array (T, X, Y), int
    """
    data = np.asarray(data, dtype=np.float32)
    T, X, Y = data.shape

    if time is None:
        time = np.arange(T, dtype=float)
    time = np.asarray(time, dtype=float)

    segments = _get_segments(time, gap_thresh)

    nan_mask = ~np.isfinite(data)
    data_filled = data.copy()
    if nan_mask.any():
        flat = data_filled.reshape(T, -1)
        for s, e in segments:
            t_seg = time[s:e]
            seg_flat = flat[s:e]
            bad = ~np.isfinite(seg_flat)
            if not bad.any():
                continue
            all_bad = bad.all(axis=0)
            seg_flat[:, all_bad] = 0.0
            partial = bad.any(axis=0) & ~all_bad
            for j in np.where(partial)[0]:
                ts = seg_flat[:, j]
                m = np.isfinite(ts)
                seg_flat[:, j] = np.interp(t_seg, t_seg[m], ts[m])

    gw = _make_odd(grad_smooth_window)

    if metric in ('gradient', 'combined'):
        grad = np.gradient(data_filled, time, axis=0)
        for s, e in segments:
            if s > 0:
                grad[s] = 0.0
            if e < T:
                grad[e - 1] = 0.0
        grad_var = np.zeros_like(grad)
        for s, e in segments:
            seg = grad[s:e]
            if e - s >= gw:
                grad_var[s:e] = np.abs(median_filter(seg, size=(gw, 1, 1), mode='reflect'))
            else:
                grad_var[s:e] = np.abs(seg)

    if metric in ('deviation', 'combined'):
        scales = [coarse_windows] if isinstance(coarse_windows, int) else list(coarse_windows)

        data_metric = _block_reduce(data_filled, block_size) if block_size > 1 else data_filled

        def _compute_dev(cw, data_metric=data_metric, segments=segments, gw=gw):
            dev = np.zeros_like(data_metric)
            for s, e in segments:
                seg = data_metric[s:e]
                n = e - s
                w = _make_odd(min(cw, n if n % 2 == 1 else n - 1))
                coarse_seg = median_filter(seg, size=(w, 1, 1), mode='reflect')
                diff = np.abs(seg - coarse_seg)
                dw = _make_odd(min(gw, n if n % 2 == 1 else n - 1))
                dev[s:e] = median_filter(diff, size=(dw, 1, 1), mode='reflect')
            return dev

        all_devs = Parallel(n_jobs=n_jobs)(delayed(_compute_dev)(cw) for cw in scales)

        _frame_mean = data_metric.mean(axis=(1, 2))
        _clipped = _frame_mean[np.isfinite(_frame_mean)]
        for _ in range(5):
            _med = np.median(_clipped)
            _std = np.std(_clipped)
            if _std == 0:
                break
            _clipped = _clipped[np.abs(_clipped - _med) < brightness_sigma * _std]
        brightness_threshold = float(np.mean(_clipped)) + brightness_sigma * float(np.std(_clipped))
        bright_mask = (_frame_mean > brightness_threshold).astype(float)[:, np.newaxis, np.newaxis]

        _frame_active = _frame_mean > brightness_threshold
        _use_stable_mask = False
        if earth_angle is not None or moon_angle is not None:
            _ea = np.asarray(earth_angle, dtype=float) if earth_angle is not None else np.full(T, np.inf)
            _ma = np.asarray(moon_angle, dtype=float) if moon_angle is not None else np.full(T, np.inf)
            _scatter_free = np.minimum(_ea, _ma) > scatter_angle_thresh
            if (~_scatter_free).any():
                _stable_frame = _scatter_free & ~_frame_active
                _stable_mask = _stable_frame[:, np.newaxis, np.newaxis].astype(float)
                _use_stable_mask = True

        lnw = _make_odd(local_norm_window)

        def _compute_norm(i_dev, bright_mask=bright_mask, segments=segments,
                          lnw=lnw, low_pct=low_pct, high_pct=high_pct):
            i, dev = i_dev
            scale_floor = max(np.nanpercentile(dev, 75), 1e-10)
            g_lo = np.zeros_like(dev)
            g_hi = np.zeros_like(dev)
            for s, e in segments:
                seg_v = dev[s:e]
                n = e - s
                lw = _make_odd(min(lnw, n if n % 2 == 1 else n - 1))
                g_lo[s:e] = percentile_filter(seg_v, low_pct, size=(lw, 1, 1), mode='reflect')
                g_hi[s:e] = percentile_filter(seg_v, high_pct, size=(lw, 1, 1), mode='reflect')
            dg = np.maximum(g_hi - g_lo, scale_floor)
            norm_scale = np.clip((dev - g_lo) / dg, 0.0, 1.0)
            if i == 0:
                norm_scale = norm_scale * bright_mask
            return norm_scale

        scale_norms = Parallel(n_jobs=n_jobs)(
            delayed(_compute_norm)((i, dev)) for i, dev in enumerate(all_devs)
        )

        if _use_stable_mask:
            scale_norms = [ns * (1.0 - _stable_mask) for ns in scale_norms]

        dev_var = np.max(np.stack(scale_norms), axis=0)

    if metric in ('local_std', 'combined'):
        lsw = _make_odd(local_std_window)
        lstd_var = np.zeros_like(data_filled)
        for s, e in segments:
            seg = data_filled[s:e]
            n = e - s
            w = _make_odd(min(lsw, n if n % 2 == 1 else n - 1))
            m = uniform_filter(seg, size=(w, 1, 1), mode='reflect')
            m2 = uniform_filter(seg ** 2, size=(w, 1, 1), mode='reflect')
            lstd_var[s:e] = np.sqrt(np.maximum(m2 - m ** 2, 0))

    def _norm01(x):
        lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        return np.clip((x - lo) / (hi - lo + 1e-30), 0, 1)

    if metric == 'gradient':
        variability = grad_var
    elif metric == 'local_std':
        variability = lstd_var
    elif metric == 'combined':
        variability = (1 - combined_weight) * _norm01(grad_var) + combined_weight * _norm01(lstd_var)
    elif metric == 'deviation':
        variability = dev_var
    else:
        raise ValueError(
            f"metric must be 'gradient', 'deviation', 'local_std', or 'combined', got '{metric}'"
        )

    if metric == 'deviation':
        norm = dev_var
    elif local_norm_window is not None:
        lnw = _make_odd(local_norm_window)
        g_lo = np.zeros_like(variability)
        g_hi = np.zeros_like(variability)
        for s, e in segments:
            seg_var = variability[s:e]
            n = e - s
            w = _make_odd(min(lnw, n if n % 2 == 1 else n - 1))
            g_lo[s:e] = percentile_filter(seg_var, low_pct, size=(w, 1, 1))
            g_hi[s:e] = percentile_filter(seg_var, high_pct, size=(w, 1, 1))
        dg = np.where((g_hi - g_lo) > 0, g_hi - g_lo, 1.0)
        norm = np.clip((variability - g_lo) / dg, 0.0, 1.0)
    elif per_pixel_norm:
        g_lo = np.nanpercentile(variability, low_pct, axis=0, keepdims=True)
        g_hi = np.nanpercentile(variability, high_pct, axis=0, keepdims=True)
        dg = np.where((g_hi - g_lo) > 0, g_hi - g_lo, 1.0)
        norm = np.clip((variability - g_lo) / dg, 0.0, 1.0)
    else:
        g_lo = np.nanpercentile(variability, low_pct)
        g_hi = np.nanpercentile(variability, high_pct)
        dg = np.where((g_hi - g_lo) > 0, g_hi - g_lo, 1.0)
        norm = np.clip((variability - g_lo) / dg, 0.0, 1.0)

    raw_w = w_max - norm * (w_max - w_min)
    windows = np.round(raw_w).astype(int)
    windows += (1 - windows % 2)
    windows = np.clip(windows, _make_odd(w_min), _make_odd(w_max))

    windows_pre_smooth = windows.copy()

    if window_smooth_size > 1:
        wsz = _make_odd(window_smooth_size)
        smoothed_win = np.empty_like(windows, dtype=float)
        for s, e in segments:
            n = e - s
            w = _make_odd(min(wsz, n if n % 2 == 1 else n - 1))
            smoothed_win[s:e] = median_filter(windows[s:e].astype(float), size=(w, 1, 1), mode='reflect')
        windows = np.round(smoothed_win).astype(int)
        windows += (1 - windows % 2)
        windows = np.clip(windows, _make_odd(w_min), _make_odd(w_max))

    if block_size > 1:
        variability = _upsample_nearest(variability, X, Y, block_size)
        windows_pre_smooth = _upsample_nearest(windows_pre_smooth, X, Y, block_size)
        windows = _upsample_nearest(windows, X, Y, block_size)

    levels = np.unique([_make_odd(int(round(w))) for w in np.linspace(w_min, w_max, n_levels)])
    windows_quantized = levels[np.argmin(np.abs(windows[..., np.newaxis] - levels), axis=-1)]

    result = np.empty((T, X, Y), dtype=np.float32)
    for s, e in segments:
        seg_data = data_filled[s:e]
        seg_wins = windows_quantized[s:e]
        seg_levels = np.unique(seg_wins)

        def _smooth_seg(w, seg=seg_data):
            return w, median_filter(seg, size=(w, 1, 1), mode='reflect')

        for w, smoothed_w in Parallel(n_jobs=n_jobs)(delayed(_smooth_seg)(w) for w in seg_levels):
            result[s:e][seg_wins == w] = smoothed_w[seg_wins == w]

    if sigma_clip is not None:
        frame_resid = np.nanmedian(np.abs(result - data_filled), axis=(1, 2))
        typical = np.nanmedian(frame_resid)
        mad_frame = np.nanmedian(np.abs(frame_resid - typical))
        frame_outlier = frame_resid > typical + sigma_clip * 1.4826 * mad_frame
        result[frame_outlier] = data_filled[frame_outlier]

    result[nan_mask] = np.nan
    return result, windows, variability, windows_pre_smooth


# ── Savitzky-Golay smoother ────────────────────────────────────────────────────

def savgol_smooth_3d(data, time=None, gap_thresh=3.0, window_length=None, polyorder=2, sigma_clip=5.0):
    """Apply a Savitzky-Golay filter along the time axis of a (T, X, Y) cube.

    Smoothing is applied independently per segment (gaps are not crossed).
    Per-pixel temporal outliers are sigma-clipped and interpolated over before
    filtering, preventing transient signals (e.g. asteroids) from biasing the
    smooth background estimate. NaNs are handled the same way.

    Parameters
    ----------
    data : array (T, X, Y)
    time : array (T,), optional
    gap_thresh : float
    window_length : int or None
        Must be odd; reduced automatically if shorter than a segment.
        If None (default), computed from the cadence to span 6 hours.
    polyorder : int
    sigma_clip : float
        Per-pixel frames more than sigma_clip * MAD above the median are
        replaced by interpolation before smoothing. Set to None to disable.

    Returns
    -------
    smoothed : array (T, X, Y)
    """
    data = np.asarray(data, dtype=np.float32)
    T, X, Y = data.shape

    if time is None:
        time = np.arange(T, dtype=float)
    time = np.asarray(time, dtype=float)

    segments = _get_segments(time, gap_thresh)

    nan_mask = ~np.isfinite(data)
    data_filled = data.copy()

    if nan_mask.any():
        flat = data_filled.reshape(T, -1)
        for s, e in segments:
            t_seg = time[s:e]
            seg_flat = flat[s:e]
            bad = ~np.isfinite(seg_flat)
            if not bad.any():
                continue
            all_bad = bad.all(axis=0)
            seg_flat[:, all_bad] = 0.0
            for j in np.where(bad.any(axis=0) & ~all_bad)[0]:
                ts = seg_flat[:, j]
                m = np.isfinite(ts)
                seg_flat[:, j] = np.interp(t_seg, t_seg[m], ts[m])

    if window_length is None:
        cadence = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
        n_frames = max(3, int(round(0.25 / cadence)))  # 6 hours = 0.25 days
        window_length = n_frames if n_frames % 2 == 1 else n_frames + 1
    wl = window_length if window_length % 2 == 1 else window_length + 1

    def _apply_savgol(arr):
        out = arr.copy()
        for s, e in segments:
            n = e - s
            w = wl
            while w >= n:
                w -= 2
            if w < polyorder + 1:
                continue
            out[s:e] = savgol_filter(arr[s:e], window_length=w, polyorder=polyorder, axis=0)
        return out

    # First pass
    first_pass = _apply_savgol(data_filled)

    # Identify outliers from first-pass residuals and interpolate over them
    if sigma_clip is not None:
        resid = data_filled - first_pass
        resid_flat = resid.reshape(T, -1)
        mad = np.nanmedian(np.abs(resid_flat - np.nanmedian(resid_flat, axis=0)), axis=0)
        robust_std = 1.4826 * mad
        # clip both positive and negative outliers
        outlier = np.abs(resid_flat) > sigma_clip * robust_std
        data_filled2 = data_filled.copy().reshape(T, -1)
        data_filled2[outlier] = np.nan
        for s, e in segments:
            t_seg = time[s:e]
            seg_flat = data_filled2[s:e]
            bad = ~np.isfinite(seg_flat)
            if not bad.any():
                continue
            all_bad = bad.all(axis=0)
            seg_flat[:, all_bad] = 0.0
            for j in np.where(bad.any(axis=0) & ~all_bad)[0]:
                ts = seg_flat[:, j]
                m = np.isfinite(ts)
                seg_flat[:, j] = np.interp(t_seg, t_seg[m], ts[m])
        data_filled2 = data_filled2.reshape(T, X, Y)
        result = _apply_savgol(data_filled2)
    else:
        result = first_pass

    # Per-frame fallback: if the whole frame deviates significantly from the
    # smooth (i.e. the smooth has smeared a sharp scattered-light transition),
    # restore the original unsmoothed value for that frame.
    frame_resid = np.nanmedian(np.abs(result - data_filled), axis=(1, 2))
    typical = np.nanmedian(frame_resid)
    mad_frame = np.nanmedian(np.abs(frame_resid - typical))
    frame_outlier = frame_resid > typical + sigma_clip * 1.4826 * mad_frame
    result[frame_outlier] = data_filled[frame_outlier]

    result[nan_mask] = np.nan
    return result


# ── Main class ─────────────────────────────────────────────────────────────────

class AdaptiveBackground:
    """Adaptive median-filter smoother for a TESS background cube.

    Parameters
    ----------
    data : array (T, X, Y)
        Background flux cube.
    time : array (T,)
        Frame mid-times in MJD (BJD − 2400000.5).
    sector : int
        TESS sector number. Used to fetch TESSVectors angle data.
    camera : int
        TESS camera number (1–4).
    data_path : str or None
        Local directory containing TESSVectors CSV files. Checked before
        downloading from the remote HEASARC server. If ``None``, always
        downloads.

    Attributes
    ----------
    smoothed : array (T, X, Y) or None
        Smoothed background, populated after calling :meth:`smooth`.
    windows : array (T, X, Y) or None
        Adaptive window sizes, populated after :meth:`smooth`.
    variability : array (T, X, Y) or None
        Normalised variability metric, populated after :meth:`smooth`.
    earth_angle, moon_angle : array (T,)
        Camera angles interpolated from TESSVectors (available after init).

    Examples
    --------
    >>> ab = AdaptiveBackground(bkg_cube, time_mjd, sector=34, camera=1)
    >>> ab.smooth()
    >>> smoothed = ab.smoothed
    """

    def __init__(self, data, time, sector, camera, data_path=None, n_jobs=-1, block_size=5):
        self.data = np.asarray(data, dtype=np.float32)
        self.time = np.asarray(time, dtype=float)
        self.sector = int(sector)
        self.camera = int(camera)
        self.data_path = data_path
        self.n_jobs = n_jobs
        self.block_size = int(block_size)

        self.smoothed = None
        self.windows = None
        self.variability = None
        self._windows_pre_smooth = None
        self.earth_angle = None
        self.moon_angle = None

        self._df = get_tessvectors(sector, camera, data_path=data_path)
        if self._df is not None:
            self.earth_angle, self.moon_angle = _interpolate_angles(self.time, self._df)

    def smooth(
        self,
        method='savgol',
        savgol_window=None,
        savgol_polyorder=2,
        gap_thresh=3.0,
        w_min=3,
        w_max=51,
        grad_smooth_window=11,
        low_pct=5,
        high_pct=80,
        per_pixel_norm=True,
        n_levels=7,
        n_jobs=None,
        metric='deviation',
        coarse_windows=(11, 21, 51, 101),
        combined_weight=0.5,
        local_std_window=21,
        local_norm_window=201,
        brightness_sigma=2.0,
        window_smooth_size=1,
        scatter_angle_thresh=50.0,
        block_size=None,
        sigma_clip=5.0,
    ):
        """Run background smoothing and store results on the instance.

        Parameters
        ----------
        method : {'savgol', 'adaptive'}
            Smoothing method. ``'savgol'`` applies a Savitzky-Golay filter;
            ``'adaptive'`` uses the adaptive median filter.
        savgol_window : int
            Window length for the Savitzky-Golay filter (must be odd).
        savgol_polyorder : int
            Polynomial order for the Savitzky-Golay filter.

        All remaining parameters are passed to :func:`adaptive_medfilt_3d`
        when ``method='adaptive'``. Returns ``self`` for method chaining.
        """
        if method == 'savgol':
            self.smoothed = savgol_smooth_3d(
                self.data,
                time=self.time,
                gap_thresh=gap_thresh,
                window_length=savgol_window,
                polyorder=savgol_polyorder,
            )
            self.windows = None
            self.variability = None
            self._windows_pre_smooth = None
        else:
            if n_jobs is None:
                n_jobs = self.n_jobs
            if block_size is None:
                block_size = self.block_size
            self.smoothed, self.windows, self.variability, self._windows_pre_smooth = (
                adaptive_medfilt_3d(
                    self.data,
                    time=self.time,
                    gap_thresh=gap_thresh,
                    w_min=w_min,
                    w_max=w_max,
                    grad_smooth_window=grad_smooth_window,
                    low_pct=low_pct,
                    high_pct=high_pct,
                    per_pixel_norm=per_pixel_norm,
                    n_levels=n_levels,
                    n_jobs=n_jobs,
                    metric=metric,
                    coarse_windows=coarse_windows,
                    combined_weight=combined_weight,
                    local_std_window=local_std_window,
                    local_norm_window=local_norm_window,
                    brightness_sigma=brightness_sigma,
                    window_smooth_size=window_smooth_size,
                    earth_angle=self.earth_angle,
                    moon_angle=self.moon_angle,
                    scatter_angle_thresh=scatter_angle_thresh,
                    block_size=block_size,
                    sigma_clip=sigma_clip,
                )
            )
        return self
