# -*- coding: utf-8 -*-

project = "spherely"
copyright = "2022, Benoit Bovy"
author = "Benoit Bovy"

# -- General configuration  ----------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "shapely": ("https://shapely.readthedocs.io/en/latest/", None),
}

# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

master_doc = "index"

exclude_patterns = [
    "**.ipynb_checkpoints",
    "build/**.ipynb",
]

templates_path = ["_templates"]

highlight_language = "python"

pygments_style = "sphinx"

# autodoc_docstring_signature = True

# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/benbovy/spherely",
}

html_static_path = ["_static"]

htmlhelp_basename = "spherelydoc"
