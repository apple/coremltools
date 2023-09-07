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
sys.path.insert(0, os.path.abspath('.'))
import pathlib


# -- Project information -----------------------------------------------------

project = 'coremltools API Reference'
copyright = '2021, Apple Inc'
author = 'Apple Inc'
release = '7.0'
version = '7.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx_gallery.gen_gallery",
]

numpydoc_show_class_members = False
napoleon_use_param = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_init_with_doc = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'display_version': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/norightmargin.css'
]

# -- Sphinx Gallery settings --------------------------------------------------

# Add tutorials here.
examples_dir = os.path.join(
    pathlib.Path(__file__).parent, "examples")
examples = [
    "optimize/torch/pruning",
    "optimize/torch/palettization",
    "optimize/torch/quantization",
]

sphinx_gallery_conf = {
    'examples_dirs': [os.path.join(examples_dir, e) for e in examples],
    'gallery_dirs': ['_examples'] * len(examples),
    'ignore_pattern': r'(__init__\.py)',
    'backreferences_dir': None,
}
