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
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
]

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
