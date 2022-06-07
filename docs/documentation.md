coremltools API Documentation
==============================

This document describes the following:

* [Viewing documentation](#viewing-documentation)
* [Updating the guides and examples](#updating-the-guides-and-examples)
* [How the API Reference is auto-generated](#how-the-api-reference-is-auto-generated)
* [Updating the API Reference](#updating-the-api-reference)
* [Generating the top-level RST files](#generating-the-top-level-rst-files)


## Viewing documentation

For coremltools documentation, see the following:

* Core ML Tools [README](https://github.com/apple/coremltools/blob/main/README.md) file for this repository
* [Guides and examples](https://coremltools.readme.io/) 
* [Release Notes](https://github.com/apple/coremltools/releases/) for the current release and previous releases
* [API Reference](https://apple.github.io/coremltools/index.html): Documentation describing the coremltools API, auto-generated from _docstrings_ in the [source code](https://github.com/apple/coremltools/tree/main/coremltools).
* [Core ML Format Specification](https://apple.github.io/coremltools/mlmodel/index.html): Documentation describing the Core ML model spec, auto-generated from the proto files located in [`coremltools/mlmodel/format`](https://github.com/apple/coremltools/tree/main/mlmodel/format).


## Updating the guides and examples

To make editing changes to [Guides and examples](https://coremltools.readme.io/), see [the Documentation section of Contributing](https://coremltools.readme.io/docs/how-to-contribute#documentation).


## How the API Reference is auto-generated
​
To automatically generate the content for the [API Reference](https://apple.github.io/coremltools/index.html), the Core ML Tools team uses the following process with [Sphinx](https://www.sphinx-doc.org) and [numpydoc](https://pypi.org/project/numpydoc/) to generate the HTML files from the _docstrings_ in the source code:

1. Set up the Sphinx configuration, and create and edit the `index.rst` file in the `coremltools/docs/` folder, and the other `.rst` files in the `coremltools/docs/source` folder, to establish the documentation's layout and navigation. 
   
   This step is performed once (and has already been performed by the team). You don't need to do it again unless you need to make changes to the Sphinx configuration, layout, or navigation, such as adding a new module. For details on how to create and edit these files, see [Generating the top-level RST files](#generating-the-top-level-rst-files).

2. Use Sphinx to generate the HTML files, and host the `html` folder on [GitHub](https://apple.github.io/coremltools/index.html).
   
   After updating the `docstrings` in the source files as described in the next section, perform this step to generate HTML from the source files, and to check the generated `html` folder into the `gh-pages` branch of the [coremltools repository](https://github.com/apple/coremltools).


## Updating the API Reference

To update the `docstrings` in the source files for the [API Reference](https://apple.github.io/coremltools/index.html), and then publish the changes, fork and clone the [coremltools repository](https://github.com/apple/coremltools) (if you haven't already done so), and make your editing changes to the source files. Then follow these general steps:

1. To generate the HTML and see a preview of your changes, run the following script from the root (`coremltools`) folder:
   
   ```shell
   zsh -i scripts/build_docs.sh
   ```
   
   The HTML files appear in the `coremltools/docs/_build/html` folder. Open `html/index.html` in your browser for a preview.

2. Push the changes to your fork, and create a pull request (PR) to merge them with the `main` branch of Apple's repo.

3. To make changes during PR review, switch to the `docs` folder, and run the following command to clean out the previous HTML before repeating the above steps:
   
   ```shell
   make clean 
   ```

After the PR is reviewed and merged, perform the following steps to generate the final HTML:

1. Run the following script from the root (`coremltools`) folder:
   
   ```
   zsh -i scripts/build_docs.sh
   ```
   
   The HTML files appear in the `coremltools/docs/_build/html` folder.

2. Switch to the root folder (`coremltools/`), check out the `main` branch, and then checkout the `gh-pages` branch from upstream:
    
	```shell
	git checkout main
	git checkout -b gh-pages upstream/gh-pages
	```

3. If the following folders appear in the root folder (or in a `git status` command), remove them:
    
	```
	coremltools.egg-info/
	coremltools/
	docs/
	envs/
    ```

4. Copy the entire contents of the  `html/` folder from your temporarily location to the root directory (`coremltools/`), overwriting any existing files and folders with the same names.

5. Commit and push your changes to *your* fork only, and submit a PR to merge them with the remote `gh-pages` branch.

At this point you can share the following URL to preview the new version, with your GitHub ID for `sample-user`: `https://sample-user.github.io/coremltools/`.


## Generating the top-level RST files

This section describes how to generate a new version of the [API Reference](https://apple.github.io/coremltools/index.html) from scratch. It also shows how to edit the Sphinx navigation and content selection elements.


### Set up your environment and Sphinx
​
You need to do this only once. Follow these steps:

1. In the `coremltools` root directory, create a Conda environment and install Sphinx, numpydoc, and the theme: 
   
   ```shell
   conda create coremltools-env
   conda activate coremltools-env
   conda install sphinx
   conda install numpydoc
   conda install sphinx_rtd_theme
   ```

2. Create a new `docs` folder in the root folder:
    
	```shell
	mkdir docs
	```

3. Switch to the `coremltools/docs` folder and start `sphinx-quickstart`:
    
	```shell
	cd docs
	sphinx-quickstart
	```

4. Accept the defaults for all options except the project name and author name:
    
	```
	> Project name: coremltools API Reference
	> Author name(s): Apple Inc
	```
    
	Sphinx finishes by adding the following to the `docs` folder:
    
	```
	_build
	_static
	_templates
	conf.py
	index.rst
	make.bat
	Makefile
	```

5. Change the Sphinx configuration by editing the `conf.py` file to un-comment the `sys.path` and add options and extensions, including the [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension and its options:
    
	```python
	...
	import os
	import sys
	sys.path.insert(0, os.path.abspath('.'))
	...
	extensions = [
	   "sphinx.ext.autodoc",
		"numpydoc",
		"sphinx.ext.napoleon",
		"sphinx.ext.coverage",
		"sphinx.ext.mathjax",
		"sphinx.ext.inheritance_diagram",
		"sphinx.ext.autosummary",
		"sphinx_rtd_theme",
	]

	numpydoc_show_class_members = False
	napoleon_use_param = False
	napoleon_numpy_docstring = True
	napoleon_include_private_with_doc = False
	napoleon_include_init_with_doc = True
	...
	html_theme = 'sphinx_rtd_theme'
	```

6. From the `docs` folder, run `sphinx-apidoc` to create the `source` folder and the initial set of `.rst` files:

	```shell
	sphinx-apidoc -o source/ ../coremltools
	```


### Edit the RST files

1. Edit the `.rst` files in the `source` folder, replacing their contents with the contents of the [existing versions in the repo](https://github.com/apple/coremltools/tree/main/docs/source):

	```
	coremltools.converters.rst
	coremltools.converters.libsvm.rst
	coremltools.converters.mil.rst
	coremltools.converters.mil.mil.ops.defs.rst
	coremltools.converters.sklearn.rst
	coremltools.converters.xgboost.rst
	coremltools.models.neural_network.rst
	coremltools.models.rst
	```
    
    Make editing changes to the above files as needed to change navigation and content.
    
2. Create `coremltools.converters.mil.input_types.rst` in the `source` folder with the [existing version in the repo](https://github.com/apple/coremltools/tree/main/docs/source/coremltools.converters.mil.input_types.rst).

3. Delete all other `.rst` files in your `source` folder. You should now have the following files in your `source` folder, along with any additional modules you may have added:

	```
	coremltools.converters.libsvm.rst
	coremltools.converters.mil.input_types.rst
	coremltools.converters.mil.mil.ops.defs.rst
	coremltools.converters.mil.rst
	coremltools.converters.rst
	coremltools.converters.sklearn.rst
	coremltools.converters.xgboost.rst
	coremltools.models.neural_network.rst
	coremltools.models.rst
	modules.rst
	```

4. Switch back to the `docs` folder, and edit the `index.rst` file as follows (or make changes as needed):

	```
	coremltools API
	================
	
	This the API Reference for coremltools. For guides, installation instructions, and examples, see `Guides <https://coremltools.readme.io/docs>`.

	.. toctree::
	   :maxdepth: 1
	   :caption: API Contents
   
	   source/coremltools.converters.rst
	   source/coremltools.models.rst
	   source/coremltools.converters.mil.input_types.rst
	   source/coremltools.converters.mil.mil.ops.defs.rst
   
	* :ref:`genindex`
	* :ref:`modindex`

	.. toctree::
	   :maxdepth: 1
	   :caption: Resources
   
   Guides <https://coremltools.readme.io/docs>
   Core ML Format Specification <https://apple.github.io/coremltools/mlmodel/index.html>
   GitHub <https://github.com/apple/coremltools>
   
	```

5. Follow the steps in [Updating the API Reference](#updating-the-api-reference).




