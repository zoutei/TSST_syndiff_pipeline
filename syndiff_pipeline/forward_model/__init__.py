# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Forward-modelled ePSF + WCS scene fitting (Gaia-positioned stars on a TESS FFI).

Migrated from ``dev/forward_epsf_wcs``. Single-FFI path: ``scene_export`` -> ``scene_fit``; multi-frame
(Paper 2): ``scene_fit_multi``, ``run_fit``/``train_loop`` and the Colab/GPU runners; overfitting checks:
``crossfit``. Default recipe: ``recipes/``.

jax is imported first on purpose: pandas/pyarrow loaded before jax pulls in the system libstdc++ and the
first XLA CPU compile segfaults.
"""

try:
    import jax  # noqa: F401  (must precede pandas/pyarrow)
except ImportError as exc:  # pragma: no cover
    raise ImportError("syndiff_pipeline.forward_model needs jax + optax: pip install -e '.[forward]'") from exc
