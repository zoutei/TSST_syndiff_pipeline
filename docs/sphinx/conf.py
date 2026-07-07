"""Sphinx configuration for syndiff_pipeline documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

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
    "sphinx_autodoc_typehints",
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
autodoc_typehints = "description"
# Mock third-party imports so autodoc can build without the full pipeline runtime
# (numpy, astropy, hotpants, MOCPy, etc.). Docstrings are read from source only.
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "scipy",
    "astropy",
    "matplotlib",
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
