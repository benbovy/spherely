# -*- coding: utf-8 -*-

project = "spherely"
copyright = "2022, Spherely Developers"
author = "Benoit Bovy"

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
}

autodoc_typehints = "none"

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

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_book_theme"
html_title = ""

html_theme_options = dict(
    repository_url="https://github.com/benbovy/spherely",
    repository_branch="main",
    path_to_docs="doc",
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
    extra_navbar="",
    navbar_footer_text="",
)

html_static_path = ["_static"]
html_logo = "_static/spherely_logo_noline.svg"
html_favicon = "_static/favicon.ico"
htmlhelp_basename = "spherelydoc"
