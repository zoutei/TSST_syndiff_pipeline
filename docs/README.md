# SynDiff documentation

- [Documentation index](markdown/README.md)
- **Published site:** https://zoutei.github.io/TSST_syndiff_pipeline/

## Build HTML locally

```bash
mamba activate syndiff
pip install -e ".[docs]" --no-deps
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
