"""Sphinx configuration for syndiff_pipeline documentation."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

# autodoc_mock_imports replaces packages with MagicMocks; warning categories like
# astropy.wcs.FITSFixedWarning are then not real classes and break filterwarnings
# at import time. Skip invalid categories during the docs build only.
_filterwarnings = warnings.filterwarnings


def _safe_filterwarnings(*args, **kwargs):
    category = kwargs.get("category")
    if category is not None and not isinstance(category, type):
        return
    _filterwarnings(*args, **kwargs)


warnings.filterwarnings = _safe_filterwarnings

project = "syndiff-pipeline"
copyright = "2026, SynDiff contributors"
author = "SynDiff contributors"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "SynDiff Pipeline"
html_baseurl = os.environ.get("DOCS_BASE_URL", "/")
html_theme_options = {
    "navigation_depth": 4,
    "show_toc_level": 2,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Mock heavy/unavailable runtime deps; numpy/scipy/pandas/matplotlib/astropy are
# installed in CI because autodoc imports modules that use them at import time.
autodoc_mock_imports = [
    "yaml",
    "zarr",
    "numba",
    "shapely",
    "joblib",
    "tqdm",
    "requests",
    "filelock",
    "dask",
    "dask_image",
    "psutil",
    "mocpy",
    "hotpants",
    "discord",
    "tglc",
    "sep",
    "regions",
    "healpy",
    "fitsio",
    "reproject",
    "astroquery",
    "photutils",
    "skimage",
    "PRF",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# Allow narrative markdown one directory up (docs/markdown/).
os.environ.setdefault("SPHINX_MULTIDOC", "1")
