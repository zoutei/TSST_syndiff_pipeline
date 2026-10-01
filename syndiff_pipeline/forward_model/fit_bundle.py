# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Serialize / load a fully prepared fit (stamps + static model) for Adam-only runs.

Prep (hp_d, centroids, Gaia, PRF) happens once via ``export_fit_bundle``.
``run_fit --from-bundle`` never opens those artifacts.

Two on-disk formats, selected by ``bundle_version``:

- ``BUNDLE_VERSION = 1`` (dense/legacy): ``data``/``noise``/``weight`` are one
  ``(G, T, S, S)`` (square) or ``(G, T, P)`` (packed) array, padded to a
  single global ``P`` for packed bundles. This is what ``save_fit_bundle``
  still writes for every bundle built the normal way (square, or packed via
  ``attach_packed_pixels``) -- fully unchanged from before.
- ``BUNDLE_VERSION_PACKED_TIERS = 2`` (tier-segmented packed): stamp arrays
  are stored per ``(K_tier, P_tier)`` bucket (``PackedTier``), each sized to
  its own occupancy instead of the global max ``P``. Written only when a
  ``FitBundle`` is explicitly constructed with ``packed_tiers`` set (see
  ``scripts/transcode_bundle_tiered.py``). ``weight`` is uint8 on disk and in
  each ``PackedTier`` -- cast to float32 only where it enters a jnp array.

``FitBundle.data``/``noise``/``weight``/``pix_x``/``pix_y``/``pix_valid``
remain readable as dense arrays on *any* bundle (backward compatibility for
every existing call site): for tier-segmented bundles they are
``functools.cached_property`` views that reconstruct the dense
global-max-padded form on first access, byte-identical to what version-1
storage would have held. New code that wants the actual memory win should
avoid touching them and use ``packed_bucket_plan()`` + ``packed_tiers``
directly instead -- see ``diagnostics/measure_full_step.py`` for the pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import cheb_wcs as CW
from . import epsf_model as EM
from .groups import GroupSet


BUNDLE_VERSION = 1
BUNDLE_VERSION_PACKED_TIERS = 2


@dataclass
class PackedTier:
    """One ``(K_tier, P_tier)`` bucket's packed arrays, sized to its own
    occupancy -- no padding to the bundle's global ``P_max``.

    ``group_idx`` maps each row back to the bundle's global group axis:
    ``bundle.members[group_idx[i]]``, ``bundle.stamp_center_x[group_idx[i]]``,
    etc. are the source-of-truth per-group fields for row ``i`` of this tier
    (those global arrays are small -- ``(G, K<=8)`` / ``(G,)`` -- and are not
    themselves tiered).
    """

    data: np.ndarray  # (n_g, T, p_tier) float32
    noise: np.ndarray  # (n_g, T, p_tier) float32
    weight_u8: np.ndarray  # (n_g, T, p_tier) uint8, 0/1
    pix_x: np.ndarray  # (n_g, p_tier) float32
    pix_y: np.ndarray  # (n_g, p_tier) float32
    pix_valid: np.ndarray  # (n_g, p_tier) float32, 0/1
    group_idx: np.ndarray  # (n_g,) int32, index into the bundle's global group axis
    k_tier: int
    p_tier: int
    # v2.1-compatible optional redundancy.  The authoritative array remains
    # FitBundle.is_epsf_contributor[group_idx]; storing it here makes a packed
    # tier self-describing for consumers that never materialize global rows.
    is_epsf_contributor: np.ndarray | None = None  # (n_g,) bool

    @property
    def n_groups(self) -> int:
        return int(self.data.shape[0])

    @property
    def weight_f32(self) -> np.ndarray:
        """Cast weight to float32. Call only at the point data enters a jnp array."""
        return self.weight_u8.astype(np.float32)


def validate_packed_tiers(
    packed_tiers: list[PackedTier], *, n_groups: int, n_frames: int
) -> None:
    """Validate a complete, non-overlapping tier partition.

    This is intentionally called both when constructing and saving v2 bundles:
    corrupt group maps otherwise tend to surface much later as wrong training
    rows, rather than as a useful bundle error.
    """
    if not packed_tiers:
        raise ValueError("packed_tiers must be non-empty")
    seen = np.zeros(int(n_groups), dtype=np.uint8)
    for i, tier in enumerate(packed_tiers):
        ng, pt = tier.n_groups, int(tier.p_tier)
        expected3 = (ng, int(n_frames), pt)
        expected2 = (ng, pt)
        for name in ("data", "noise", "weight_u8"):
            if np.asarray(getattr(tier, name)).shape != expected3:
                raise ValueError(f"packed tier {i} {name} shape must be {expected3}")
        for name in ("pix_x", "pix_y", "pix_valid"):
            if np.asarray(getattr(tier, name)).shape != expected2:
                raise ValueError(f"packed tier {i} {name} shape must be {expected2}")
        gi = np.asarray(tier.group_idx)
        if gi.shape != (ng,) or not np.issubdtype(gi.dtype, np.integer):
            raise ValueError(f"packed tier {i} group_idx must be integer ({ng},)")
        if pt <= 0 or int(tier.k_tier) <= 0:
            raise ValueError(f"packed tier {i} has non-positive K/P tier")
        if np.any((gi < 0) | (gi >= int(n_groups))):
            raise ValueError(f"packed tier {i} group_idx out of range")
        if np.any(seen[gi]):
            raise ValueError(f"packed tier {i} overlaps an earlier group_idx")
        seen[gi] = 1
        wu8 = np.asarray(tier.weight_u8)
        if wu8.dtype != np.uint8 or np.any((wu8 != 0) & (wu8 != 1)):
            raise ValueError(f"packed tier {i} weight_u8 must contain only uint8 0/1")
        pv = np.asarray(tier.pix_valid)
        if np.any((pv != 0) & (pv != 1)):
            raise ValueError(f"packed tier {i} pix_valid must contain only 0/1")
        if tier.is_epsf_contributor is not None:
            contrib = np.asarray(tier.is_epsf_contributor)
            if contrib.shape != (ng,) or contrib.dtype != np.bool_:
                raise ValueError(f"packed tier {i} is_epsf_contributor must be bool ({ng},)")
    missing = np.flatnonzero(seen == 0)
    if missing.size:
        raise ValueError(f"packed tiers do not cover {missing.size} group(s)")


class FitBundle:
    """Everything needed to build FitData / run Adam without workspace I/O."""

    def __init__(
        self,
        *,
        data: np.ndarray | None = None,
        noise: np.ndarray | None = None,
        weight: np.ndarray | None = None,
        stamp_center_x: np.ndarray,
        stamp_center_y: np.ndarray,
        mask_active: np.ndarray,  # (G, T)
        ra: np.ndarray,
        dec: np.ndarray,
        x_lin: np.ndarray,  # (n_stars,) baked linear WCS
        y_lin: np.ndarray,
        cheb_basis: np.ndarray,  # (n_stars, n_terms)
        members: np.ndarray,  # (G, K)
        valid: np.ndarray,  # (G, K) bool
        kept_star_mask: np.ndarray,
        max_group_size: int,
        stamp_snr_weight: np.ndarray,
        fit_radius_stage1: np.ndarray,
        fit_radius_stage23: np.ndarray,
        cheb_static: CW.ChebWcsStatic,
        epsf_grid: EM.EpsfGridStatic,
        wcs_frame_basis: np.ndarray,  # (T, n_wcs)
        w_frame_basis: np.ndarray,  # (T, n_w)
        epsf_base: np.ndarray,
        epsf_modes: np.ndarray,
        params0: dict[str, np.ndarray],  # optimizer leaves after stage-0 warmstart
        t_exp_sec: float,
        stamp_physical: int,
        k_tiers: tuple[int, ...],
        # Packed irregular supports (None / P=0 -> square path)
        pix_x: np.ndarray | None = None,  # (G, P)
        pix_y: np.ndarray | None = None,
        pix_valid: np.ndarray | None = None,
        p_tiers: tuple[int, ...] = (),
        is_epsf_contributor: np.ndarray | None = None,  # (G,) bool
        bp_rp: np.ndarray | None = None,  # (n_stars,) Gaia BP-RP, NaN where unknown
        meta: dict | None = None,
        # Tier-segmented packed storage (bundle_version=2). When set, `data`/
        # `noise`/`weight`/`pix_x`/`pix_y`/`pix_valid` above should be left
        # None -- they become lazy dense reconstructions (see the cached
        # properties below) instead of being held eagerly.
        packed_tiers: list[PackedTier] | None = None,
    ) -> None:
        if packed_tiers is not None and not packed_tiers:
            raise ValueError("packed_tiers must be None or non-empty")
        if data is None and packed_tiers is None:
            raise ValueError("FitBundle needs either `data` or `packed_tiers`")

        # Pre-empt the cached_property descriptors below: a plain instance
        # attribute of the same name takes priority over a non-data
        # descriptor, so this costs nothing extra for the direct/dense path
        # (square bundles, legacy packed bundles, hand-built test bundles)
        # and the cached_property is simply never invoked for them.
        if data is not None:
            self.data = data
        if noise is not None:
            self.noise = noise
        if weight is not None:
            self.weight = weight
        if pix_x is not None:
            self.pix_x = pix_x
        if pix_y is not None:
            self.pix_y = pix_y
        if pix_valid is not None:
            self.pix_valid = pix_valid

        self.stamp_center_x = stamp_center_x
        self.stamp_center_y = stamp_center_y
        self.mask_active = mask_active
        self.ra = ra
        self.dec = dec
        self.x_lin = x_lin
        self.y_lin = y_lin
        self.cheb_basis = cheb_basis
        self.members = members
        self.valid = valid
        self.kept_star_mask = kept_star_mask
        self.max_group_size = max_group_size
        self.stamp_snr_weight = stamp_snr_weight
        self.fit_radius_stage1 = fit_radius_stage1
        self.fit_radius_stage23 = fit_radius_stage23
        self.cheb_static = cheb_static
        self.epsf_grid = epsf_grid
        self.wcs_frame_basis = wcs_frame_basis
        self.w_frame_basis = w_frame_basis
        self.epsf_base = epsf_base
        self.epsf_modes = epsf_modes
        self.params0 = params0
        self.t_exp_sec = t_exp_sec
        self.stamp_physical = stamp_physical
        self.k_tiers = k_tiers
        self.p_tiers = p_tiers
        if is_epsf_contributor is None:
            is_epsf_contributor = np.ones(np.asarray(members).shape[0], dtype=bool)
        self.is_epsf_contributor = np.asarray(is_epsf_contributor, dtype=bool)
        # Per-STAR Gaia colour, on the same axis as ra/dec/x_lin. None means the
        # bundle predates the chromatic term; NaN entries mean the colour is unknown
        # for that star and are turned into a zero offset at context build.
        if bp_rp is None:
            self.bp_rp = None
        else:
            arr = np.asarray(bp_rp, dtype=np.float64)
            if arr.shape != (np.asarray(ra).shape[0],):
                raise ValueError(
                    f"bp_rp must be per-star with shape {(np.asarray(ra).shape[0],)}, "
                    f"got {arr.shape}"
                )
            self.bp_rp = arr
        self.meta = meta if meta is not None else {}
        self.packed_tiers = packed_tiers

        self._is_packed = bool(packed_tiers) or (
            pix_x is not None and int(np.asarray(pix_x).shape[-1]) > 0
        )
        # Known without ever touching `data` -- the whole point for
        # tier-segmented bundles (avoids forcing a dense reconstruction just
        # to answer "how many groups/frames").
        self._n_groups = int(np.asarray(members).shape[0])
        self._n_frames = int(np.asarray(wcs_frame_basis).shape[0])
        if self.is_epsf_contributor.shape != (self._n_groups,):
            raise ValueError(
                "is_epsf_contributor must have shape "
                f"({self._n_groups},), got {self.is_epsf_contributor.shape}"
            )
        if packed_tiers is not None:
            validate_packed_tiers(
                packed_tiers, n_groups=self._n_groups, n_frames=self._n_frames
            )

    @property
    def n_groups(self) -> int:
        return self._n_groups

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def is_packed(self) -> bool:
        return self._is_packed

    def group_set(self) -> GroupSet:
        return GroupSet(
            n_groups=self.n_groups,
            max_group_size=int(self.max_group_size),
            members=np.asarray(self.members, dtype=int),
            valid=np.asarray(self.valid, dtype=bool),
            kept_star_mask=np.asarray(self.kept_star_mask, dtype=bool),
            dropped_oversized=0,
        )

    # -- backward-compat dense views (tier-segmented bundles only) ---------

    def _dense_stamp_array(self, key: str, frame_indices=None) -> np.ndarray:
        """Dense (G, T, p_max) reconstruction, optionally bounded to just
        ``frame_indices`` frame columns (default: all ``self._n_frames``).

        Bounding this is the difference between a few tens of MB and tens of
        GB at full-orbit scale for callers that only need a handful of
        frames -- see ``diagnostics/export_fits.py``'s ``_fd_from_bundle``.
        """
        if self.packed_tiers is None:
            raise AttributeError(
                f"FitBundle.{key} unavailable: bundle has neither a literal "
                f"`{key}` array nor `packed_tiers` (malformed bundle)"
            )
        g = self._n_groups
        t = self._n_frames if frame_indices is None else len(frame_indices)
        p_max = max(int(tier.p_tier) for tier in self.packed_tiers)
        fill = 1.0 if key == "noise" else 0.0  # matches packed_support.concat_packed_batches
        out = np.full((g, t, p_max), fill, dtype=np.float32)
        for tier in self.packed_tiers:
            src = tier.weight_f32 if key == "weight" else getattr(tier, key)
            if frame_indices is not None:
                src = src[:, frame_indices]
            out[tier.group_idx, :, : tier.p_tier] = src
        return out

    def _dense_pix_array(self, key: str) -> np.ndarray | None:
        if self.packed_tiers is None:
            return None
        g = self._n_groups
        p_max = max(int(tier.p_tier) for tier in self.packed_tiers)
        out = np.zeros((g, p_max), dtype=np.float32)
        for tier in self.packed_tiers:
            out[tier.group_idx, : tier.p_tier] = getattr(tier, key)
        return out

    @cached_property
    def data(self) -> np.ndarray:
        return self._dense_stamp_array("data")

    @cached_property
    def noise(self) -> np.ndarray:
        return self._dense_stamp_array("noise")

    @cached_property
    def weight(self) -> np.ndarray:
        return self._dense_stamp_array("weight")

    @cached_property
    def pix_x(self) -> np.ndarray | None:
        return self._dense_pix_array("pix_x")

    @cached_property
    def pix_y(self) -> np.ndarray | None:
        return self._dense_pix_array("pix_y")

    @cached_property
    def pix_valid(self) -> np.ndarray | None:
        return self._dense_pix_array("pix_valid")

    # -- efficient tier-native access (bypasses the dense reconstruction) --

    def packed_bucket_plan(
        self,
    ) -> list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray, int, int]]:
        """Bucket plan sourced directly from tier-segmented storage.

        Same return shape as ``packed_support.bucket_packed_by_kp``:
        ``(bucket_groups, cx, cy, group_idx, k_tier, p_tier)`` per tier, in
        the same order as ``self.packed_tiers`` (zip them 1:1 for the
        matching stamp/pixel arrays). Never touches ``.data``/``.pix_x``.
        """
        if self.packed_tiers is None:
            raise ValueError(
                "packed_bucket_plan() needs tier-segmented storage "
                "(bundle_version=2); this bundle has none -- use "
                "packed_support.bucket_packed_by_kp(bundle.group_set(), "
                "bundle.stamp_center_x, bundle.stamp_center_y, "
                "bundle.pix_valid, ...) for legacy dense/packed bundles"
            )
        members = np.asarray(self.members)
        valid = np.asarray(self.valid)
        cx = np.asarray(self.stamp_center_x)
        cy = np.asarray(self.stamp_center_y)
        n_stars = int(np.asarray(self.kept_star_mask).shape[0])
        out: list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
        for tier in self.packed_tiers:
            gi = np.asarray(tier.group_idx, dtype=np.int64)
            kt = int(tier.k_tier)
            m = members[gi][:, :kt].astype(np.int32)
            v = valid[gi][:, :kt].astype(bool)
            kept = np.zeros(n_stars, dtype=bool)
            flat = m[v]
            in_range = (flat >= 0) & (flat < n_stars)
            kept[flat[in_range]] = True
            bg = GroupSet(int(gi.size), kt, m, v, kept, 0)
            out.append((bg, cx[gi], cy[gi], gi.astype(np.int32), kt, int(tier.p_tier)))
        return out


def _cheb_to_arrays(static: CW.ChebWcsStatic) -> dict[str, np.ndarray]:
    exps = np.asarray(static.exponents, dtype=np.int32)
    return {
        "cheb_ra0_deg": np.asarray(static.ra0_deg, dtype=np.float64),
        "cheb_dec0_deg": np.asarray(static.dec0_deg, dtype=np.float64),
        "cheb_cd_inv": np.asarray(static.cd_inv, dtype=np.float64),
        "cheb_crpix": np.asarray(static.crpix, dtype=np.float64),
        "cheb_center": np.asarray(static.center, dtype=np.float64),
        "cheb_half_extents": np.asarray(static.half_extents, dtype=np.float64),
        "cheb_poly_degree": np.asarray(static.poly_degree, dtype=np.int32),
        "cheb_exponents": exps,
    }


def _cheb_from_arrays(d: dict) -> CW.ChebWcsStatic:
    exps = np.asarray(d["cheb_exponents"])
    exponents = tuple((int(a), int(b)) for a, b in exps.reshape(-1, 2))
    return CW.ChebWcsStatic(
        ra0_deg=float(np.asarray(d["cheb_ra0_deg"])),
        dec0_deg=float(np.asarray(d["cheb_dec0_deg"])),
        cd_inv=np.asarray(d["cheb_cd_inv"], dtype=float),
        crpix=np.asarray(d["cheb_crpix"], dtype=float),
        center=np.asarray(d["cheb_center"], dtype=float),
        half_extents=np.asarray(d["cheb_half_extents"], dtype=float),
        poly_degree=int(np.asarray(d["cheb_poly_degree"])),
        exponents=exponents,
    )


def _epsf_grid_to_arrays(grid: EM.EpsfGridStatic) -> dict[str, np.ndarray]:
    return {
        "epsf_node_x": np.asarray(grid.node_x, dtype=np.float64),
        "epsf_node_y": np.asarray(grid.node_y, dtype=np.float64),
        "epsf_node_col_ccd": np.asarray(grid.node_col_ccd, dtype=np.float64),
        "epsf_node_row_ccd": np.asarray(grid.node_row_ccd, dtype=np.float64),
    }


def _epsf_grid_from_arrays(d: dict) -> EM.EpsfGridStatic:
    return EM.EpsfGridStatic(
        node_x=np.asarray(d["epsf_node_x"], dtype=float),
        node_y=np.asarray(d["epsf_node_y"], dtype=float),
        node_col_ccd=np.asarray(d["epsf_node_col_ccd"], dtype=float),
        node_row_ccd=np.asarray(d["epsf_node_row_ccd"], dtype=float),
    )


def _static_model_arrays(bundle: FitBundle) -> dict[str, np.ndarray]:
    """Fields common to both bundle_version formats."""
    arrays: dict[str, np.ndarray] = {
        "stamp_center_x": np.asarray(bundle.stamp_center_x, dtype=np.float64),
        "stamp_center_y": np.asarray(bundle.stamp_center_y, dtype=np.float64),
        "mask_active": np.asarray(bundle.mask_active, dtype=np.float32),
        "ra": np.asarray(bundle.ra, dtype=np.float64),
        **({} if getattr(bundle, "bp_rp", None) is None
           else {"bp_rp": np.asarray(bundle.bp_rp, dtype=np.float64)}),
        "dec": np.asarray(bundle.dec, dtype=np.float64),
        "x_lin": np.asarray(bundle.x_lin, dtype=np.float32),
        "y_lin": np.asarray(bundle.y_lin, dtype=np.float32),
        "cheb_basis": np.asarray(bundle.cheb_basis, dtype=np.float32),
        "members": np.asarray(bundle.members, dtype=np.int32),
        "valid": np.asarray(bundle.valid, dtype=np.bool_),
        "is_epsf_contributor": np.asarray(bundle.is_epsf_contributor, dtype=np.bool_),
        "kept_star_mask": np.asarray(bundle.kept_star_mask, dtype=np.bool_),
        "max_group_size": np.asarray(bundle.max_group_size, dtype=np.int32),
        "stamp_snr_weight": np.asarray(bundle.stamp_snr_weight, dtype=np.float32),
        "fit_radius_stage1": np.asarray(bundle.fit_radius_stage1, dtype=np.float32),
        "fit_radius_stage23": np.asarray(bundle.fit_radius_stage23, dtype=np.float32),
        "wcs_frame_basis": np.asarray(bundle.wcs_frame_basis, dtype=np.float32),
        "w_frame_basis": np.asarray(bundle.w_frame_basis, dtype=np.float32),
        "epsf_base": np.asarray(bundle.epsf_base, dtype=np.float32),
        "epsf_modes": np.asarray(bundle.epsf_modes, dtype=np.float32),
        "epsf_repr": np.asarray(EM.EPSF_REPR),
        "t_exp_sec": np.asarray(bundle.t_exp_sec, dtype=np.float64),
        "stamp_physical": np.asarray(bundle.stamp_physical, dtype=np.int32),
        "k_tiers": np.asarray(bundle.k_tiers, dtype=np.int32),
    }
    arrays.update(_cheb_to_arrays(bundle.cheb_static))
    arrays.update(_epsf_grid_to_arrays(bundle.epsf_grid))
    for key, val in bundle.params0.items():
        arrays[f"params0_{key}"] = np.asarray(val)
    return arrays


def _dense_arrays(bundle: FitBundle) -> dict[str, np.ndarray]:
    """bundle_version=1 payload -- identical to the original implementation."""
    arrays = _static_model_arrays(bundle)
    arrays["data"] = np.asarray(bundle.data, dtype=np.float32)
    arrays["noise"] = np.asarray(bundle.noise, dtype=np.float32)
    arrays["weight"] = np.asarray(bundle.weight, dtype=np.float32)
    arrays["p_tiers"] = np.asarray(bundle.p_tiers if bundle.p_tiers else (), dtype=np.int32)
    if bundle.pix_x is not None:
        arrays["pix_x"] = np.asarray(bundle.pix_x, dtype=np.float32)
        arrays["pix_y"] = np.asarray(bundle.pix_y, dtype=np.float32)
        arrays["pix_valid"] = np.asarray(bundle.pix_valid, dtype=np.float32)
    return arrays


def _tier_segmented_arrays(bundle: FitBundle) -> dict[str, np.ndarray]:
    """bundle_version=2 payload: one array group per PackedTier, no global padding."""
    assert bundle.packed_tiers is not None
    validate_packed_tiers(
        bundle.packed_tiers, n_groups=bundle.n_groups, n_frames=bundle.n_frames
    )
    arrays = _static_model_arrays(bundle)
    tiers = bundle.packed_tiers
    arrays["n_packed_tiers"] = np.asarray(len(tiers), dtype=np.int32)
    p_tiers_sorted = tuple(sorted({int(t.p_tier) for t in tiers}))
    arrays["p_tiers"] = np.asarray(p_tiers_sorted, dtype=np.int32)
    for i, tier in enumerate(tiers):
        pfx = f"pt{i}_"
        arrays[pfx + "data"] = np.asarray(tier.data, dtype=np.float32)
        arrays[pfx + "noise"] = np.asarray(tier.noise, dtype=np.float32)
        arrays[pfx + "weight_u8"] = np.asarray(tier.weight_u8, dtype=np.uint8)
        arrays[pfx + "pix_x"] = np.asarray(tier.pix_x, dtype=np.float32)
        arrays[pfx + "pix_y"] = np.asarray(tier.pix_y, dtype=np.float32)
        arrays[pfx + "pix_valid"] = np.asarray(tier.pix_valid, dtype=np.float32)
        arrays[pfx + "group_idx"] = np.asarray(tier.group_idx, dtype=np.int32)
        arrays[pfx + "k_tier"] = np.asarray(tier.k_tier, dtype=np.int32)
        arrays[pfx + "p_tier"] = np.asarray(tier.p_tier, dtype=np.int32)
        arrays[pfx + "is_epsf_contributor"] = np.asarray(
            bundle.is_epsf_contributor[tier.group_idx]
            if tier.is_epsf_contributor is None else tier.is_epsf_contributor,
            dtype=np.bool_,
        )
    return arrays


def save_fit_bundle(path: Path, bundle: FitBundle) -> Path:
    """Write ``fit_bundle.npz`` (+ sibling ``fit_bundle_meta.json``).

    Writes ``bundle_version=2`` (tier-segmented) iff ``bundle.packed_tiers``
    is set; otherwise writes ``bundle_version=1`` (dense), exactly as before.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".npz":
        path = path / "fit_bundle.npz" if path.suffix == "" or path.is_dir() else path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    if bundle.packed_tiers is not None:
        arrays = _tier_segmented_arrays(bundle)
        version = BUNDLE_VERSION_PACKED_TIERS
    else:
        arrays = _dense_arrays(bundle)
        version = BUNDLE_VERSION
    arrays["bundle_version"] = np.asarray(version, dtype=np.int32)

    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)

    meta = dict(bundle.meta)
    meta["bundle_version"] = version
    meta["epsf_repr"] = EM.EPSF_REPR
    meta_path = path.with_name(path.stem + "_meta.json")
    if path.name == "fit_bundle.npz":
        meta_path = path.with_name("fit_bundle_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return path


def _load_dense_bundle(path: Path, raw: dict) -> FitBundle:
    """bundle_version=1 -- unchanged from the original implementation."""
    params0 = {}
    for key in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        k = f"params0_{key}"
        if k not in raw:
            raise KeyError(f"{path} missing {k}")
        params0[key] = np.asarray(raw[k])
    # Legacy (pre-representation-change) fit bundles stored 58-grid raw
    # leaves -- detect + convert (one warning) rather than silently
    # mis-rendering a sub-pixel grid as pixel-integrated.
    params0 = EM.convert_legacy_params_raw(params0, name=f"{path} params0")

    meta_path = path.with_name("fit_bundle_meta.json")
    if not meta_path.exists():
        meta_path = path.with_name(path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    pix_x = np.asarray(raw["pix_x"]) if "pix_x" in raw else None
    pix_y = np.asarray(raw["pix_y"]) if "pix_y" in raw else None
    pix_valid = np.asarray(raw["pix_valid"]) if "pix_valid" in raw else None
    p_tiers = tuple(int(x) for x in np.asarray(raw.get("p_tiers", [])).ravel())
    contrib = np.asarray(
        raw.get("is_epsf_contributor", np.ones(np.asarray(raw["members"]).shape[0], dtype=bool)),
        dtype=bool,
    )
    epsf_base = np.asarray(EM.convert_legacy_epsf_array(raw["epsf_base"], name=f"{path} epsf_base"))
    epsf_modes = np.asarray(EM.convert_legacy_epsf_array(raw["epsf_modes"], name=f"{path} epsf_modes"))

    return FitBundle(
        data=np.asarray(raw["data"]),
        noise=np.asarray(raw["noise"]),
        weight=np.asarray(raw["weight"]),
        stamp_center_x=np.asarray(raw["stamp_center_x"]),
        stamp_center_y=np.asarray(raw["stamp_center_y"]),
        mask_active=np.asarray(raw["mask_active"]),
        ra=np.asarray(raw["ra"]),
        bp_rp=(np.asarray(raw["bp_rp"]) if "bp_rp" in raw else None),
        dec=np.asarray(raw["dec"]),
        x_lin=np.asarray(raw["x_lin"]),
        y_lin=np.asarray(raw["y_lin"]),
        cheb_basis=np.asarray(raw["cheb_basis"]),
        members=np.asarray(raw["members"]),
        valid=np.asarray(raw["valid"]),
        kept_star_mask=np.asarray(raw["kept_star_mask"]),
        max_group_size=int(np.asarray(raw["max_group_size"])),
        stamp_snr_weight=np.asarray(raw["stamp_snr_weight"]),
        fit_radius_stage1=np.asarray(raw["fit_radius_stage1"]),
        fit_radius_stage23=np.asarray(raw["fit_radius_stage23"]),
        cheb_static=_cheb_from_arrays(raw),
        epsf_grid=_epsf_grid_from_arrays(raw),
        wcs_frame_basis=np.asarray(raw["wcs_frame_basis"]),
        w_frame_basis=np.asarray(raw["w_frame_basis"]),
        epsf_base=epsf_base,
        epsf_modes=epsf_modes,
        params0=params0,
        t_exp_sec=float(np.asarray(raw["t_exp_sec"])),
        stamp_physical=int(np.asarray(raw["stamp_physical"])),
        k_tiers=tuple(int(x) for x in np.asarray(raw["k_tiers"]).ravel()),
        meta=meta,
        pix_x=pix_x,
        pix_y=pix_y,
        pix_valid=pix_valid,
        p_tiers=p_tiers,
        is_epsf_contributor=contrib,
    )


def _load_tier_segmented_bundle(path: Path, raw: dict) -> FitBundle:
    """bundle_version=2 -- reconstruct PackedTier list, no dense arrays touched."""
    params0 = {}
    for key in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        k = f"params0_{key}"
        if k not in raw:
            raise KeyError(f"{path} missing {k}")
        params0[key] = np.asarray(raw[k])
    # Legacy (pre-representation-change) fit bundles stored 58-grid raw
    # leaves -- detect + convert (one warning) rather than silently
    # mis-rendering a sub-pixel grid as pixel-integrated.
    params0 = EM.convert_legacy_params_raw(params0, name=f"{path} params0")

    meta_path = path.with_name("fit_bundle_meta.json")
    if not meta_path.exists():
        meta_path = path.with_name(path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    n_tiers = int(np.asarray(raw["n_packed_tiers"]))
    if n_tiers <= 0:
        raise ValueError(f"{path}: bundle_version=2 but n_packed_tiers={n_tiers}")
    contrib = np.asarray(
        raw.get("is_epsf_contributor", np.ones(np.asarray(raw["members"]).shape[0], dtype=bool)),
        dtype=bool,
    )
    tiers: list[PackedTier] = []
    for i in range(n_tiers):
        pfx = f"pt{i}_"
        tiers.append(PackedTier(
            data=np.asarray(raw[pfx + "data"], dtype=np.float32),
            noise=np.asarray(raw[pfx + "noise"], dtype=np.float32),
            weight_u8=np.asarray(raw[pfx + "weight_u8"], dtype=np.uint8),
            pix_x=np.asarray(raw[pfx + "pix_x"], dtype=np.float32),
            pix_y=np.asarray(raw[pfx + "pix_y"], dtype=np.float32),
            pix_valid=np.asarray(raw[pfx + "pix_valid"], dtype=np.float32),
            group_idx=np.asarray(raw[pfx + "group_idx"], dtype=np.int32),
            k_tier=int(np.asarray(raw[pfx + "k_tier"])),
            p_tier=int(np.asarray(raw[pfx + "p_tier"])),
            is_epsf_contributor=np.asarray(
                raw.get(
                    pfx + "is_epsf_contributor",
                    contrib[np.asarray(raw[pfx + "group_idx"], dtype=int)],
                ),
                dtype=bool,
            ),
        ))
    p_tiers = tuple(int(x) for x in np.asarray(raw.get("p_tiers", [])).ravel())
    epsf_base = np.asarray(EM.convert_legacy_epsf_array(raw["epsf_base"], name=f"{path} epsf_base"))
    epsf_modes = np.asarray(EM.convert_legacy_epsf_array(raw["epsf_modes"], name=f"{path} epsf_modes"))

    return FitBundle(
        stamp_center_x=np.asarray(raw["stamp_center_x"]),
        stamp_center_y=np.asarray(raw["stamp_center_y"]),
        mask_active=np.asarray(raw["mask_active"]),
        ra=np.asarray(raw["ra"]),
        bp_rp=(np.asarray(raw["bp_rp"]) if "bp_rp" in raw else None),
        dec=np.asarray(raw["dec"]),
        x_lin=np.asarray(raw["x_lin"]),
        y_lin=np.asarray(raw["y_lin"]),
        cheb_basis=np.asarray(raw["cheb_basis"]),
        members=np.asarray(raw["members"]),
        valid=np.asarray(raw["valid"]),
        kept_star_mask=np.asarray(raw["kept_star_mask"]),
        max_group_size=int(np.asarray(raw["max_group_size"])),
        stamp_snr_weight=np.asarray(raw["stamp_snr_weight"]),
        fit_radius_stage1=np.asarray(raw["fit_radius_stage1"]),
        fit_radius_stage23=np.asarray(raw["fit_radius_stage23"]),
        cheb_static=_cheb_from_arrays(raw),
        epsf_grid=_epsf_grid_from_arrays(raw),
        wcs_frame_basis=np.asarray(raw["wcs_frame_basis"]),
        w_frame_basis=np.asarray(raw["w_frame_basis"]),
        epsf_base=epsf_base,
        epsf_modes=epsf_modes,
        params0=params0,
        t_exp_sec=float(np.asarray(raw["t_exp_sec"])),
        stamp_physical=int(np.asarray(raw["stamp_physical"])),
        k_tiers=tuple(int(x) for x in np.asarray(raw["k_tiers"]).ravel()),
        meta=meta,
        p_tiers=p_tiers,
        is_epsf_contributor=contrib,
        packed_tiers=tiers,
    )


def load_fit_bundle(path: Path) -> FitBundle:
    """Load a bundle written by ``save_fit_bundle`` (either format)."""
    path = Path(path)
    if path.is_dir():
        path = path / "fit_bundle.npz"
    raw = dict(np.load(path, allow_pickle=False))
    ver = int(np.asarray(raw.get("bundle_version", 0)))
    if ver == BUNDLE_VERSION:
        return _load_dense_bundle(path, raw)
    if ver == BUNDLE_VERSION_PACKED_TIERS:
        return _load_tier_segmented_bundle(path, raw)
    raise ValueError(
        f"{path}: unsupported bundle_version={ver} "
        f"(expected {BUNDLE_VERSION} or {BUNDLE_VERSION_PACKED_TIERS})"
    )


def params0_as_jnp(bundle: FitBundle) -> dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in bundle.params0.items()}


def attach_packed_pixels(
    bundle: FitBundle,
    *,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    pix_valid: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    weight: np.ndarray,
    p_tiers: tuple[int, ...] = (),
) -> FitBundle:
    """Return a copy of ``bundle`` with packed 1D supports (data shape ``G,T,P``).

    Always produces a dense (bundle_version=1) packed bundle -- the same
    format as before. Use ``scripts/transcode_bundle_tiered.py`` to convert
    an existing dense packed bundle to tier-segmented (bundle_version=2)
    storage.
    """
    if data.ndim != 3:
        raise ValueError(f"packed data must be (G,T,P), got shape {data.shape}")
    if pix_x.shape != pix_y.shape or pix_x.shape != pix_valid.shape:
        raise ValueError("pix_x/pix_y/pix_valid shape mismatch")
    if data.shape[0] != pix_x.shape[0] or data.shape[2] != pix_x.shape[1]:
        raise ValueError("data vs pix_* shape mismatch")
    meta = dict(bundle.meta)
    meta["packed"] = True
    meta["p_tier"] = int(pix_x.shape[1])
    return FitBundle(
        data=np.asarray(data, dtype=np.float32),
        noise=np.asarray(noise, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
        stamp_center_x=bundle.stamp_center_x,
        stamp_center_y=bundle.stamp_center_y,
        mask_active=bundle.mask_active,
        ra=bundle.ra,
        dec=bundle.dec,
        x_lin=bundle.x_lin,
        y_lin=bundle.y_lin,
        cheb_basis=bundle.cheb_basis,
        members=bundle.members,
        valid=bundle.valid,
        kept_star_mask=bundle.kept_star_mask,
        max_group_size=bundle.max_group_size,
        stamp_snr_weight=bundle.stamp_snr_weight,
        fit_radius_stage1=bundle.fit_radius_stage1,
        fit_radius_stage23=bundle.fit_radius_stage23,
        cheb_static=bundle.cheb_static,
        epsf_grid=bundle.epsf_grid,
        wcs_frame_basis=bundle.wcs_frame_basis,
        w_frame_basis=bundle.w_frame_basis,
        epsf_base=bundle.epsf_base,
        epsf_modes=bundle.epsf_modes,
        params0=bundle.params0,
        t_exp_sec=bundle.t_exp_sec,
        stamp_physical=bundle.stamp_physical,
        k_tiers=bundle.k_tiers,
        pix_x=np.asarray(pix_x, dtype=np.float32),
        pix_y=np.asarray(pix_y, dtype=np.float32),
        pix_valid=np.asarray(pix_valid, dtype=np.float32),
        p_tiers=tuple(p_tiers) if p_tiers else (int(pix_x.shape[1]),),
        is_epsf_contributor=bundle.is_epsf_contributor,
        meta=meta,
    )


def bundle_from_tiers(bundle: FitBundle, packed_tiers: list[PackedTier]) -> FitBundle:
    """Return a copy of a (dense, packed) ``bundle`` using tier-segmented
    storage instead -- i.e. drop the literal ``data``/``noise``/``weight``/
    ``pix_*`` arrays and let them become lazy dense views over
    ``packed_tiers``. Used by ``scripts/transcode_bundle_tiered.py``.
    """
    if not packed_tiers:
        raise ValueError("packed_tiers must be non-empty")
    p_tiers_sorted = tuple(sorted({int(t.p_tier) for t in packed_tiers}))
    return FitBundle(
        stamp_center_x=bundle.stamp_center_x,
        stamp_center_y=bundle.stamp_center_y,
        mask_active=bundle.mask_active,
        ra=bundle.ra,
        dec=bundle.dec,
        x_lin=bundle.x_lin,
        y_lin=bundle.y_lin,
        cheb_basis=bundle.cheb_basis,
        members=bundle.members,
        valid=bundle.valid,
        kept_star_mask=bundle.kept_star_mask,
        max_group_size=bundle.max_group_size,
        stamp_snr_weight=bundle.stamp_snr_weight,
        fit_radius_stage1=bundle.fit_radius_stage1,
        fit_radius_stage23=bundle.fit_radius_stage23,
        cheb_static=bundle.cheb_static,
        epsf_grid=bundle.epsf_grid,
        wcs_frame_basis=bundle.wcs_frame_basis,
        w_frame_basis=bundle.w_frame_basis,
        epsf_base=bundle.epsf_base,
        epsf_modes=bundle.epsf_modes,
        params0=bundle.params0,
        t_exp_sec=bundle.t_exp_sec,
        stamp_physical=bundle.stamp_physical,
        k_tiers=bundle.k_tiers,
        p_tiers=p_tiers_sorted,
        is_epsf_contributor=bundle.is_epsf_contributor,
        meta=dict(bundle.meta),
        packed_tiers=packed_tiers,
    )


def tiered_packed_bundle(
    bundle: FitBundle,
    *,
    k_tiers: tuple[int, ...] | None = None,
    p_tiers: tuple[int, ...] | None = None,
) -> FitBundle:
    """Convert a dense packed bundle to v2 without changing numerical rows.

    Already-tiered bundles are returned unchanged when no new ladder is
    requested.  The local import avoids a module import cycle.
    """
    if bundle.packed_tiers is not None and k_tiers is None and p_tiers is None:
        validate_packed_tiers(
            bundle.packed_tiers, n_groups=bundle.n_groups, n_frames=bundle.n_frames
        )
        return bundle
    if not bundle.is_packed or bundle.packed_tiers is not None:
        raise ValueError("tiered_packed_bundle requires a dense packed bundle")
    from . import packed_support as PS

    kt = tuple(k_tiers or bundle.k_tiers or PS.DEFAULT_K_TIERS)
    pt = tuple(p_tiers or bundle.p_tiers or PS.DEFAULT_P_TIERS)
    plan = PS.bucket_packed_by_kp(
        bundle.group_set(), bundle.stamp_center_x, bundle.stamp_center_y,
        np.asarray(bundle.pix_valid), k_tiers=kt, p_tiers=pt,
    )
    weight = np.asarray(bundle.weight)
    if np.any((weight != 0) & (weight != 1)):
        raise ValueError("packed weight must be binary for lossless uint8 v2 storage")
    tiers = [PackedTier(
        data=np.ascontiguousarray(bundle.data[gi, :, :p], dtype=np.float32),
        noise=np.ascontiguousarray(bundle.noise[gi, :, :p], dtype=np.float32),
        weight_u8=np.ascontiguousarray(weight[gi, :, :p], dtype=np.uint8),
        pix_x=np.ascontiguousarray(bundle.pix_x[gi, :p], dtype=np.float32),
        pix_y=np.ascontiguousarray(bundle.pix_y[gi, :p], dtype=np.float32),
        pix_valid=np.ascontiguousarray(bundle.pix_valid[gi, :p], dtype=np.float32),
        group_idx=np.asarray(gi, dtype=np.int32), k_tier=int(k), p_tier=int(p),
        is_epsf_contributor=np.asarray(bundle.is_epsf_contributor[gi], dtype=bool),
    ) for _bg, _cx, _cy, gi, k, p in plan]
    return bundle_from_tiers(bundle, tiers)
