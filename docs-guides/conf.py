# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Core ML Tools Guide'
copyright = '2023, Apple Inc'
author = 'Apple'
release = '7.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_copybutton"
]

# MyST extensions setting
myst_enable_extensions = [
    "colon_fence",
    "html_admonition",
    "html_image",
    "smartquotes",
]

# MyST heading anchors and url schemes
myst_heading_anchors = 4
myst_url_schemes = ['http', 'https', 'mailto', 'ftp', 'phantom', 'adir']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = "Guide to Core ML Tools"

html_theme_options = {
    "repository_url": "https://github.com/apple/coremltools",
    "use_repository_button": True,
}

html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/imgstyle.css',
]
