"""Register the ``slow`` marker (parity tests that re-run stages on the e2e F1 products under /astro)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402  (before pandas/pyarrow, and compile once: pyarrow-before-XLA-init segfaults)
import jax.numpy as jnp  # noqa: E402

(jnp.zeros(2) + 1).block_until_ready()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: parity test against the e2e F1 products (skipped if /astro paths are absent)")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _jax_x64_mode():
    """Run every chain test with JAX x64 on, as the chain stages run (``chain.kernels`` / ``wcs_export`` switch it on
    in their own process), and restore the previous mode afterwards. Under pytest all test modules are imported at
    collection, so ``chain.kernels``'s import-time switch used to leak into the float32 forward-model tests, which then
    failed only in full-suite runs (2026-10-05); ``tests/forward_model/conftest.py`` pins those to float32."""
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)
