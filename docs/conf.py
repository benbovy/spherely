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
    "numpy": ("https://numpy.org/doc/stable", None),
    "shapely": ("https://shapely.readthedocs.io/en/latest/", None),
}


napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
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
