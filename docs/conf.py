import os
import sys

# Make src/ importable
sys.path.insert(0, os.path.abspath(".."))

project = "OOP for AI — Dataset Library"
author = "Raphael MARTIN"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

html_theme = "sphinx_rtd_theme"
html_static_path = []
