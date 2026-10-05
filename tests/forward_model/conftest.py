"""Pin JAX's x64 mode for every test in this directory and restore it afterwards.

``syndiff_pipeline.forward_model.chain.kernels`` switches JAX to float64 at import time (each chain stage runs in its
own process, so in production this only affects that stage). Under pytest every test module is imported at
collection, so that switch leaked into the float32 forward-model tests, which then failed only in full-suite runs
(2026-10-05). The forward-model tests expect float32; the chain tests expect float64 (as their stages run).
"""
import pytest


@pytest.fixture(autouse=True)
def _jax_x64_mode():
    try:
        import jax
    except ImportError:
        yield
        return
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)
