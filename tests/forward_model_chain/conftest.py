"""Register the ``slow`` marker (parity tests that re-run stages on the e2e F1 products under /astro)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402  (before pandas/pyarrow, and compile once: pyarrow-before-XLA-init segfaults)
import jax.numpy as jnp  # noqa: E402

(jnp.zeros(2) + 1).block_until_ready()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: parity test against the e2e F1 products (skipped if /astro paths are absent)")
