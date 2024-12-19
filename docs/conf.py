# -*- coding: utf-8 -*-

import spherely

project = "spherely"
author = "Spherely developers"
copyright = "2022, Spherely Developers"
# The short X.Y version.
version = spherely.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
release = spherely.__version__

# -- General configuration  ----------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "shapely": ("https://shapely.readthedocs.io/en/latest/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    # objects without namespace: spherely
    "EARTH_RADIUS_METERS": "spherely.EARTH_RADIUS_METERS",
    # objects without namespace: numpy
    "ndarray": "~numpy.ndarray",
    "array": ":term:`array`",
}

source_suffix = [".rst", ".md"]

master_doc = "index"

language = "en"

exclude_patterns = [
    "**.ipynb_checkpoints",
    "build/**.ipynb",
]

templates_path = ["_templates"]

highlight_language = "python"

pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_book_theme"
html_title = ""

html_theme_options = dict(
    repository_url="https://github.com/benbovy/spherely",
    repository_branch="main",
    path_to_docs="docs",
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
)

html_static_path = ["_static"]
html_logo = "_static/spherely_logo_noline.svg"
html_favicon = "_static/favicon.ico"
htmlhelp_basename = "spherelydoc"
