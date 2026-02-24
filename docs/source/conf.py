# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
from importlib.metadata import version as get_version
from pathlib import Path
import shutil

# -- Project information -----------------------------------------------------

project = 'xwmb'
copyright = '2026'
authors = 'Henri Drake and Graeme MacGilchrist'

release = get_version(project)
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon','IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive','jupyter_sphinx','nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
nbsphinx_execute = "never"

# -- Copy notebooks from repo root/examples into docs/source/examples --------

HERE = Path(__file__).resolve()
DOCS_SOURCE = HERE.parent                       # docs/source
REPO_ROOT = HERE.parents[2]                     # up two levels from conf.py
EXAMPLES_SRC = REPO_ROOT / "examples"
EXAMPLES_DST = DOCS_SOURCE / "examples"

def _sync_examples():
    if not EXAMPLES_SRC.exists():
        return

    # fresh copy so removed notebooks don't linger
    if EXAMPLES_DST.exists():
        shutil.rmtree(EXAMPLES_DST)
    EXAMPLES_DST.mkdir(parents=True, exist_ok=True)

    for nb in EXAMPLES_SRC.rglob("*.ipynb"):
        rel = nb.relative_to(EXAMPLES_SRC)
        out = EXAMPLES_DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nb, out)

_sync_examples()

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

#-- Turn off copyright notice
html_show_copyright = False

#-- Master document
master_doc = 'index'

#-- Build api
from sphinx.ext.apidoc import main
main(['-f', '-M', '-e', '-T', '../../xwmb', '-o', 'api' ])
