# Reuse validity: can patched assignments be shared?

**Worktree:** `hybrid-boundary-recompute`  
**Script:** [`scripts/test_reuse_whole_ccd_gt_ffis.py`](scripts/test_reuse_whole_ccd_gt_ffis.py)  
**Outputs:** [`outputs/`](outputs/)  
**SCC:** s0020 / camera 3 / ccd 2 · **1044 skycells** · **3 GT FFIs** + **593-frame** reuse probe on micromap sites

## Verdict

**Do not reuse on integer `(sx, sy)` alone.**

You *can* amortize across ~900 templates, but the cache key must include **fractional PS1 phase**, not only the integer roll. Within one integer bin, frames can sit at opposite rounding corners (`frac_dist` up to ~√2); their nearest-pixel TESS ownership then differs by **~1–2%** of PS1 pixels (worst) / **~0.4%** (typical). That residual is almost entirely explained by fractional distance (`corr ≈ 0.97` with both inverse-WCS and pancakes GT).

**Recommended key for hybrid patch reuse:**

```text
key = (skycell, quantize(sx_f, q), quantize(sy_f, q))   with q = 0.25 PS1 px
```

At `q=0.25`, worst-pair disagree on the 4 GT skycells drops to **median ~0.3%, max ~0.6%**.

Integer-only reuse is only acceptable if you explicitly budget an extra ~1% ownership noise (comparable to the notebook Exact≠Linear floor after a good roll, ~0.4–0.8%, but worse in the worst bin).

---

## What was tested

| Part | Scope | Method |
|------|-------|--------|
| A | Whole CCD × 3 GT FFIs | Local TESS drift at every skycell center → PS1 float/int shift vs reference |
| B | 4 micromap skycells × 3 GT epochs | Pancakes GT: raw disagree vs after integer-align roll |
| C | Whole CCD × 3 GT FFI pairs | Inverse WCS nearest-pixel disagree vs `|Δint|` |
| D | 4 micromap sites × 3 GT epochs | Inverse WCS tid vs pancakes GT (validation of inverse) |
| E | 4 micromap sites × 593 frames | Same-int-bin reuse + `q=0.25` reuse |

GT FFIs:

- `tess2020004172923-s0020-3-2-0165-s_ffic.fits.gz` (reference)
- `tess2020013135923-s0020-3-2-0165-s_ffic.fits.gz`
- `tess2020020065923-s0020-3-2-0165-s_ffic.fits.gz`

---

## Results (measured)

### Whole CCD (3 GT FFIs)

- Pairs sharing the same int key: **9 / 3132 (0.29%)** — the three epochs are far apart in time, so almost every skycell has moved.
- Those 9 “same int” pairs still have **large fractional separation** (`frac_dist` 0.72–1.15) — they only match because rounding put them in the same bin from opposite sides.
- Inverse disagree scales cleanly with Manhattan `|Δsx|+|Δsy|`:

| `\|Δint\|` | n | disagree median | `d_tess` median |
|------------|---|-----------------|-----------------|
| 0 | 9 | 1.0% | 0.009 |
| 1 | 133 | 1.5% | 0.012 |
| 3 | 697 | 3.4% | 0.028 |
| 6 | 385 | 6.5% | 0.054 |
| 9 | 11 | 10.2% | 0.085 |

### Pancakes GT (roll factorization)

Integer-align roll of one exact regmap toward another cuts disagree a lot, but a **frac-dependent residual remains**:

| metric | value |
|--------|-------|
| raw disagree median | 3.7% |
| after int-align roll median | **0.75%** |
| after int-align roll max | **1.62%** (at `frac_dist≈0.91`) |
| `corr(aligned_disagree, frac_dist)` | **0.976** |

So: “skycell moved by N PS1 pixels” ≈ roll of the exact map, **plus** a residual set by the leftover fractional phase. That is exactly why int-only keys under-share geometry.

### Inverse ≡ GT on these sites

Nearest-pixel inverse WCS **agrees with pancakes GT tids at 100%** on 30k-pixel footprint samples for all 4 sites × 3 epochs. (Continuous residual ~0.38 TESS px is distance to pixel center, not a tid mismatch.)

### Same-int reuse across 593 frames (the real amortization question)

| | worst pair in bin | typical pair |
|--|-------------------|--------------|
| disagree median | **1.05%** | **0.39%** |
| disagree max | **2.05%** | 0.83% |
| `corr(worst_disagree, frac_dist)` | **0.974** | — |

Disagree vs fractional separation inside the int bin:

| `frac_dist` | n bins | worst disagree median | max |
|-------------|--------|----------------------|-----|
| 0–0.1 | 5 | 0.11% | 0.15% |
| 0.1–0.25 | 8 | 0.24% | 0.31% |
| 0.25–0.5 | 12 | 0.62% | 0.77% |
| 0.5–0.75 | 11 | 0.90% | 1.13% |
| 0.75–1.01 | 13 | 1.47% | 1.76% |

`q=0.25` multi-frame bins: worst-pair disagree **median ~0.30%, max ~0.60%**.

Unique int keys per micromap skycell over 593 frames: **~18–20** (not 593) — amortization is real; the question is only how fine to quantize.

---

## Decision for the hybrid pipeline

1. **Amortize** patch work by unique local phase keys, not by template count.
2. **Key = quantized float shift** `(skycell, q·round(sx_f/q), q·round(sy_f/q))` with default **`q=0.25`**.
3. Integer `(sx_i, sy_i)` alone is a coarser / cheaper tier (`q=1`), with known ~1% worst-case ownership error inside the bin — do not claim it is exact reuse.
4. Representative WCS for a key: any frame whose `(sx_f, sy_f)` falls in that quantum (Jacobian argument from the prior attempt still applies *within* a fine enough bin).
5. Inverse WCS is a valid patch engine for nearest-pixel ownership on these GT sites; keep the Phase-1 gate that patches must match pancakes on `needs_recompute` pixels.

## How to re-run

```bash
mamba activate syndiff
cd .claude/worktrees/hybrid-boundary-recompute
export PYTHONPATH="$PWD:$PYTHONPATH"
python -u dev/distortion_aware_template/reuse_validity/scripts/test_reuse_whole_ccd_gt_ffis.py
```

---

## Locations in this worktree

- Canonical doc (tracked): `doc/hybrid_boundary_reuse_validity.md` (this file)
- Runnable script (tracked): `tools/hybrid_reuse_validity/test_reuse_whole_ccd_gt_ffis.py`
- Local scratch copy + CSVs/plots: `dev/distortion_aware_template/reuse_validity/` (gitignored `dev/`)
