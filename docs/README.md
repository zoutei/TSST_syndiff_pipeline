# SynDiff documentation

- [Documentation index](markdown/README.md)
- [Field templates (default geometry)](markdown/field_geometry.md) — L0–L5, L4a/L4b, storage, ops
- [Coordinate frames and cropping](markdown/coordinate_frames_and_cropping.md) — full-FFI/crop-local temporal WCS boundary, MappingGrid, OS indexing, L5 completeness
- [Storage layout](markdown/storage_layout.md) (`workspace_root`/`data_root`, SCC + nested-event layout)
- **Published site:** https://zoutei.github.io/TSST_syndiff_pipeline/

## Build HTML locally

```bash
mamba activate syndiff
pip install -e ".[docs]"          # local: installs Sphinx + hotpants
# CI-style lightweight build (no hotpants):
# pip install "sphinx>=7.0" "myst-parser>=3.0" "pydata-sphinx-theme>=0.15" "sphinx-autodoc-typehints>=2.0"
# pip install -e . --no-deps
cd docs/sphinx && make html
```

The docs build mocks third-party imports (numpy, astropy, hotpants, …) so the full pipeline runtime is not required — same as GitHub Actions CI.

Output: `docs/_build/html/index.html` (user guide + API reference).

## Publish to GitHub Pages

Deployment is **manual only** — pushes to `main` do not auto-update the site.

1. GitHub → **Actions** → **docs** → **Run workflow**
2. Select branch `main` → **Run workflow**

Requires **Settings → Pages → Source: GitHub Actions** (one-time setup).

Layout:

- `markdown/` — narrative documentation (markdown source)
- `sphinx/` — Sphinx build config and API autodoc stubs
- `_build/` — generated HTML (gitignored)
