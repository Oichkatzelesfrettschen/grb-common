"""Sphinx configuration for grb-common documentation."""

import os
import sys

# Add source path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'grb-common'
copyright = '2025, OpenUniverse'
author = 'OpenUniverse Contributors'
version = '0.1.0'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
autodoc_typehints = 'description'
autosummary_generate = True

# Napoleon settings (for numpy/google docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Source files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'grb-common Documentation'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Suppress warnings for missing references to optional dependencies
suppress_warnings = ['ref.option']
