"""QA plots for shared-mask bits under ``debug_plots/masks/``."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from syndiff_pipeline.masking import bits
from syndiff_pipeline.masking.bits import epsf_reject_mask, full_mask_bool
from syndiff_pipeline.masking.catalog import MaskCatalog

log = logging.getLogger(__name__)


def _save_bit_png(path: Path, layer: np.ndarray, title: str, dpi: int = 120) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.debug("matplotlib missing; skip mask plot %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(layer.astype(float), origin="lower", cmap="gray_r", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x (crop)")
    ax.set_ylabel("y (crop)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info("  mask plot → %s", path)


def write_mask_debug_plots(
    catalog: MaskCatalog,
    plot_dir: str | Path,
    *,
    sample_cadence: int | None = None,
    dpi: int = 120,
) -> None:
    """Write bit-plane and consumer-predicate PNGs under *plot_dir*."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    static = catalog.static

    planes = [
        (bits.BRIGHT_CAT, "bit1_bright_cat.png", "bit 1 BRIGHT_CAT"),
        (bits.SAT_CROSS, "bit2_sat_cross.png", "bit 2 SAT_CROSS"),
        (bits.FAINT_CAT, "bit32_faint_cat.png", "bit 32 FAINT_CAT"),
        (bits.TNS, "bit64_tns.png", "bit 64 TNS"),
    ]
    for bit, name, title in planes:
        _save_bit_png(plot_dir / name, (static.astype(np.int64) & bit) != 0, title, dpi=dpi)

    _save_bit_png(
        plot_dir / "epsf_reject_mask.png",
        epsf_reject_mask(static),
        "ePSF reject (ignore bits 1|2)",
        dpi=dpi,
    )

    cad = sample_cadence
    if cad is None and catalog.asteroid_times is not None and not catalog.asteroid_times.empty:
        cad = int(catalog.asteroid_times["cadence"].iloc[len(catalog.asteroid_times) // 2])
    if cad is not None and catalog.has_temporal():
        temporal = catalog.mask_at(cad, which="temporal")
        _save_bit_png(
            plot_dir / "bit128_asteroid_sample.png",
            (temporal.astype(np.int64) & bits.ASTEROID) != 0,
            f"bit 128 ASTEROID cadence={cad}",
            dpi=dpi,
        )
        full = catalog.mask_at(cad, which="full")
        _save_bit_png(
            plot_dir / "hotpants_full_sample.png",
            full_mask_bool(full),
            f"Hotpants full mask cadence={cad}",
            dpi=dpi,
        )
    else:
        _save_bit_png(
            plot_dir / "hotpants_full_sample.png",
            full_mask_bool(static),
            "Hotpants full mask (static)",
            dpi=dpi,
        )
