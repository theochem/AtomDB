import importlib
import pathlib
import sys

#
# Configure path
#

sys.path.insert(0, str(pathlib.Path(__file__).parents[2].absolute()))

#
# Import module
#

module = importlib.import_module("atomdb")

#
# Module info
#

project = "AtomDB"

project_copyright = "2024, QC-Devs"

author = "QC-Devs"

version = getattr(module, "version")

release = version

#
# General configuration
#

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "nbsphinx",
]

bibtex_bibfiles = ["atomdb_cite.bib"]

templates_path = [
    "_templates",
]

exclude_patterns = [
    "_build",
    "_themes/*",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

pygments_style = "sphinx"

#
# Autodoc configuration
#

autoclass_content = "both"
autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}

#
# HTML configuration
#

html_show_sourcelink = False
html_style = "css/override.css"
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
}

html_static_path = [
    "_static",
]

#
# Nbsphinx configuration
#

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=200",
]

nbsphinx_execute = "never"

binder_url = (
    "https://mybinder.org/v2/gh/theochem/atomdb/master"
    "?filepath=docs%2Fnotebooks%2F{{ docname }}.ipynb"
)

binder_badge = "https://mybinder.org/badge_logo.svg"

binder_badge_sty = "vertical-align:text-bottom"

binder_html = (
    f'<a href="{binder_url}">'
    f'<img alt="Binder" src="{binder_badge}" style="{binder_badge_sty}">'
    f"</a>"
)

nbsphinx_prolog = f"""
{{% set docname = env.docname.split("/")[-1] %}}

.. raw:: html

.. role:: raw-html(raw)
  :format: html
.. nbinfo::
    The corresponding file can be obtained from:

    - Jupyter Notebook: :download:`{{{{docname+".ipynb"}}}}`
    - Interactive Jupyter Notebook: :raw-html:`{binder_html}`
"""
