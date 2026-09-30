# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Guard: --from-bundle import graph must not pull prep-only heavy deps."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


BLOCKED_ROOTS = {"astropy", "scipy", "pandas", "PRF", "syndiff_pipeline"}
BLOCKED_EXACT = {
    "data_io",
    "fit_wcs_from_centroids",
    "temporal_model",
    "cheb_poly_fit",
    "sip_poly_fit",
}


@pytest.mark.parametrize(
    "modname",
    [
        "syndiff_pipeline.forward_model.fit_bundle",
        "syndiff_pipeline.forward_model.train_loop",
        "syndiff_pipeline.forward_model.fit",
            "syndiff_pipeline.forward_model.run_fit",
            "syndiff_pipeline.forward_model.train_from_bundle",
            "syndiff_pipeline.forward_model.stamp_reject",
        "syndiff_pipeline.forward_model.loss",
        "syndiff_pipeline.forward_model.epsf_model",
        "syndiff_pipeline.forward_model.cheb_wcs",
        "syndiff_pipeline.forward_model.groups",
        "syndiff_pipeline.forward_model.temporal",
    ],
)
def test_from_bundle_modules_import_without_prep_stack(modname):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    # Ensure jax/numpy are already loaded before the guard.
    import jax  # noqa: F401
    import jax.numpy  # noqa: F401
    import numpy  # noqa: F401
    import optax  # noqa: F401

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in BLOCKED_ROOTS or name in BLOCKED_EXACT:
            raise ImportError(f"lean-train blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    # Drop previously loaded targets so the guarded import path is exercised.
    doomed = [
        key for key in sys.modules
        if key.startswith("syndiff_pipeline.forward_model")
        or key in BLOCKED_EXACT
        or key.split(".")[0] in BLOCKED_ROOTS
    ]
    # Snapshot the evicted modules and put them back in the finally below.
    # Evicting `astropy` (a BLOCKED_ROOT) without restoring it poisons every
    # later test in the same process: astropy's logger installs itself into
    # `warnings.showwarning` on first import, so a re-import after eviction
    # builds a *new* AstropyLogger while `warnings.showwarning` still points at
    # the orphaned old module's bound method. astropy then refuses to touch it
    # ("LoggingError: Cannot disable warnings logging: warnings.showwarning was
    # not set by AstropyLogger") and any test importing astropy dies at
    # collection. That took out 13 of 163 tests -- including 6 in the README's
    # own documented four-file gate -- purely on test ordering.
    evicted = {key: sys.modules[key] for key in doomed}
    for key in doomed:
        del sys.modules[key]

    builtins.__import__ = guarded
    try:
        importlib.import_module(modname)
    finally:
        builtins.__import__ = real_import
        # Discard whatever this test imported under the guard, then restore the
        # pre-test module table so the process is left exactly as we found it.
        for key in [k for k in sys.modules if k.startswith("syndiff_pipeline.forward_model")]:
            del sys.modules[key]
        sys.modules.update(evicted)
