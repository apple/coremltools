coremltools API Documentation
==============================

This document describes the following:

* [Viewing documentation](#viewing-documentation)
* [Updating the guide and examples](#updating-the-guides-and-examples)
* [How the API Reference is auto-generated](#how-the-api-reference-is-auto-generated)
* [Updating the API Reference](#updating-the-api-reference)
* [Generating the top-level RST files](#generating-the-top-level-rst-files)



## Viewing documentation

For coremltools documentation, see the following:

* Core ML Tools [README](https://github.com/apple/coremltools/blob/main/README.md) file for this repository
* [Guide and examples](https://apple.github.io/coremltools/docs-guides/)
* [Release Notes](https://github.com/apple/coremltools/releases/) for the current release and previous releases
* [API Reference](https://apple.github.io/coremltools/index.html): Documentation describing the coremltools API, auto-generated from _docstrings_ in the source code.
* [Core ML Format Specification](https://apple.github.io/coremltools/mlmodel/index.html): Documentation describing the Core ML model spec, auto-generated from the proto files located in [`coremltools/mlmodel/format`](https://github.com/apple/coremltools/tree/main/mlmodel/format).


## Updating the guides and examples

To make editing changes to the [Guide and examples](https://apple.github.io/coremltools/docs-guides/index.html), Send a pull request as described in [Contributions](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#contributions) and add the **docs** label to it (see [Labels](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#labels) for details).


## How the API Reference is auto-generated

The Core ML Tools team uses the following process with [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) and [numpydoc](https://pypi.org/project/numpydoc/) to automatically generate the HTML files from the _docstrings_ in the source code:

1. Create and edit the `index.rst` file and other `.rst` files in the `coremltools/docs/source` folder to establish the documentation layout and navigation.

   This step has already been done. It does not need to be done again unless changes are required to the layout and navigation, such as adding a new class or method. For details on how to create and edit these files, see [Generating the top-level RST files](#generating-the-top-level-rst-files).

2. Use Sphinx to generate the HTML files, and host the `html` folder on [GitHub](https://apple.github.io/coremltools/index.html).

   After updating the `docstrings` in the source files as described in the next section, perform this step to check the generated `html` folder into the `gh-pages` branch of [this repository](https://github.com/apple/coremltools).



## Updating the API Reference

To update the `docstrings` in the source files for the [API Reference](https://apple.github.io/coremltools/index.html), follow these general steps:

1. [Fork and clone](#fork-and-clone) the coremltools repo.
2. [Install Sphinx and theme](#install-sphinx-and-theme) in the root directory (`coremltools`) if you have not already done so.
3. [Edit and generate HTML for preview](#edit-and-generate-html-for-preview).
4. [Copy the HTML to gh-pages](#copy-the-html-to-gh-pages).


### Fork and clone

1. With your GitHub ID for `sample-user`, fork and clone the `coremltools` repo, change to your fork clone root folder (`coremltools/`), and add the upstream repo to the list of remotes:

	```shell
	git clone git@github.com:sample-user/coremltools.git
	cd coremltools
	git remote add upstream https://github.com/apple/coremltools.git
	git remote -v
	```

2. Update your fork:

    ```shell
	git fetch upstream
	git checkout main
	git merge upstream/main
	```

### Install Sphinx and theme

In the `coremltools` clone root directory, install or upgrade Sphinx, the theme, and numpydoc for this environment if you have not already done so:

```shell
conda install sphinx
conda install numpydoc
conda install sphinx_rtd_theme
```

### Edit and generate HTML for preview

1. Create a new branch (`newfeature`) for changes:

    ```shell
	git checkout -b newfeature
	```

2. Make your editing changes. Check status to see the files that changed.

    ```shell
	git status
	```

3. Switch to the `docs` folder and run `make clean` to delete any old HTML files (no effect if there are no HTML files):

    ```shell
	cd docs
	make clean
	```

4. Change to the `coremltools` root folder (parent of `docs`), and run the following script to generate the HTML:

	```shell
	cd ..
	zsh ./scripts/build_docs.sh
	```

	*Warning*: Include the `./` or the script may not function correctly.

	The HTML files appear in the `docs/_build/html` folder.

5. Preview your changes by right-clicking the `index.html` file in the `html` folder and choosing **Safari** (or another browser).

6. To make more changes repeat Steps 2-4.

7. If satisfied with the preview from Step 5, add your changes for the commit.

    ```shell
    git add --all
    ```

8. Make the commit.

    ```shell
    git commit -m "Commit message"
    ```

9. Copy the `html` folder to a temporarily location on your computer (outside of the `coremltools` clone controlled by git).

10. Push the changes to your fork:

    ```shell
    git push
    ```

11. On your fork, request a Pull Request (PR) for your changes.

Once approved, these changes will eventually be merged from your `newfeature` branch to the `main` branch. Do not delete your local branch yet. You still need to [Copy the HTML to gh-pages](#copy-the-html-to-gh-pages).


### Copy the HTML to gh-pages

1. Be sure that you have a copy of the `html` folder in a temporarily location (outside of the `coremltools` clone controlled by git).

2. Switch to the root folder (`coremltools`), check out the `main` branch, and then checkout the `gh-pages` branch from upstream:

	```shell
	git checkout main
	git checkout -b gh-pages upstream/gh-pages
	```

3. If the following folders appear in the root folder (or in a `git status` command), remove them:

	```
	build
	coremltools
	coremltools.egg-info
	deps
	docs
	envs
    ```

   You should now have only the following files and folders in the root folder of the gh-pages branch:

   ```
   _modules
   _sources
   _static
   docs-api-version3
   genindex.html
   index.html
   mlmodel
   objects.inv
   py-modindex.html
   search.html
   searchindex.js
   source
   ```

4. Copy the entire contents of the  `html` folder to the root directory (`coremltools`), overwriting any existing files and folders with the same names.

5. Add and commit your changes, and push your changes to *your* fork:

	```shell
	git add --all
	git commit -m "Commit message for gh-pages"
	git push origin gh-pages
	```

	At this point you can share the following URL to preview the new version (substituting your user name for `sample-user`): `https://sample-user.github.io/coremltools/`.

6. On your fork, switch to the `gh-pages` branch, and request a Pull Request (PR) for your changes to merge them with the `gh-pages` branch on the Apple repo.

Once approved, these changes will eventually be merged from your `gh-pages` branch to the Apple repo's `gh-pages` branch.


## Generating a new API reference

This section is for generating a new version of the API Reference from scratch, and also shows how to edit the Sphinx navigation and content selection elements.


### Set up the environment

1. [Fork and clone](#fork-and-clone) the coremltools repo.

2. [Install Sphinx and theme](#install-sphinx-and-theme) in the root directory (`coremltools`) if you have not already done so.

3. From the `main` branch, do a build for coremltools:

	```shell
	zsh ./scripts/build.sh
	```

	*Warning*: Include the `./` or the script may not function correctly.

	The build files appear in the `build` folder.

    If you run into a `cmake` error (such as a warning that `cmake` was not found in your PATH), you may need to change your PATH. If you need help with this, see [Use environment variables in Terminal on Mac](https://support.apple.com/guide/terminal/use-environment-variables-apd382cc5fa-4f58-4449-b20a-41c53c006f8f/mac). You may also need to install cmake.

4. Create a new `docs` folder in the `coremltools` root folder:

   *Warning*: This step deletes the current version of the `docs` folder.

	```shell
	mkdir docs
	```

### Set the Sphinx options

1. Change to the `docs` folder and start `sphinx-quickstart`:

	```shell
	cd docs
	sphinx-quickstart
	```

2. Accept the defaults for all options except the project name and author name:

	```
	> Project name: coremltools API Reference
	> Author name(s): Apple Inc
	```

	This procedure finishes by adding the following to the `docs` folder:

	```
	_build
	_static
	_templates
	conf.py
	index.rst
	make.bat
	Makefile
	```

    You can delete `make.bat` unless you are using MS Windows.

3. Set the Sphinx configuration by editing the `conf.py` file:

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
		"sphinx.ext.viewcode",
		"sphinx_rtd_theme"
	]

	numpydoc_show_class_members = False
	napoleon_use_param = False
	napoleon_numpy_docstring = True
	napoleon_include_private_with_doc = False
	napoleon_include_init_with_doc = True
	...
	html_theme = 'sphinx_rtd_theme'
	html_static_path = ['_static']
    html_css_files = [
         'css/norightmargin.css'
    ]
	```

4. Create the `css` folder in the `_static` folder, and add the following as the file `norightmargin.css` in the `css` folder:

   ```
   .wy-nav-content {
    max-width: none;
   }
   ```


### Edit the RST files

1. From the `docs` folder, run `sphinx-apidoc` to create the `source` folder with the `.rst` files for Sphinx doc generation:

	```shell
	cd docs
	sphinx-apidoc -o source/ ../coremltools
	```

2. Edit the `.rst` files in your local `source` folder, replacing them with the contents of the [existing versions in the repo](https://github.com/apple/coremltools/tree/main/docs/source).

	*Note*: The existing versions replace the ones generated by Step 1, and at least one is new, such as `coremltools.converters.mil.input_types.rst`.

	Make editing changes to the above files as needed to change navigation and content.

3. Delete all other `.rst` files in your local `source` folder. As a result, you should have only these files, and they should match the ones in the [existing versions in the repo](https://github.com/apple/coremltools/tree/main/docs/source) (unless you made modifications):

	```
	coremltools.converters.convert.rst
	coremltools.converters.libsvm.rst
	coremltools.converters.mil.input_types.rst
	coremltools.converters.mil.mil.ops.defs.rst
 	coremltools.converters.mil.mil.passes.defs.rst
	coremltools.converters.mil.rst
	coremltools.converters.rst
	coremltools.converters.sklearn.rst
	coremltools.converters.xgboost.rst
	coremltools.models.ml_program.rst
	coremltools.models.neural_network.rst
	coremltools.models.rst
	```

4. Switch back to the `docs` folder, and edit the `index.rst` file as follows (or make changes as needed):

```
coremltools API
=====================

This is the API Reference for coremltools. For guides, installation instructions, and examples, see `Guides <https://coremltools.readme.io/docs>`_.

.. toctree::
   :maxdepth: 1
   :caption: API Contents

   source/coremltools.converters.rst
   source/coremltools.models.rst
   source/coremltools.converters.mil.rst
   source/coremltools.converters.mil.input_types.rst
   source/coremltools.converters.mil.mil.ops.defs.rst
   source/coremltools.converters.mil.mil.passes.defs.rst

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :maxdepth: 1
   :caption: Resources

   Guides and examples <https://coremltools.readme.io/docs>
   Core ML Format Specification <https://apple.github.io/coremltools/mlmodel/index.html>
   GitHub <https://github.com/apple/coremltools>
```


### Generate the HTML

1. Follow the steps in [Edit and generate HTML for preview](#edit-and-generate-html-for-preview).

2. Follow the steps in [Copy the HTML to gh-pages](#copy-the-html-to-gh-pages).
