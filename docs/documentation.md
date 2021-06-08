coremltools API Documentation
==============================

This document describes the following:

* [Viewing documentation](#viewing-documentation)
* [Updating the documentation](#updating-the-documentation)
* [Generating a new API reference](#generating-a-new-api-reference)


## Viewing documentation

For coremltools documentation, see the following:

* Core ML Tools [README](README.md) file for this repository
* [Guides and examples](https://coremltools.readme.io/) with installation and troubleshooting
* [API Reference](https://apple.github.io/coremltools/index.html)
* [Building from Source](https://github.com/apple/coremltools/blob/master/BUILDING.md)
* [Release Notes](https://github.com/apple/coremltools/releases/) for the current release and previous releases


## Updating the documentation

You can update the following:

### Updating the guides and examples

To make editing changes to [Guides and examples](https://coremltools.readme.io/), see [the Documentation section of Contributing](https://coremltools.readme.io/docs/how-to-contribute#documentation).

### Updating the API reference

To create a new version or update the HTML files for the API reference, follow these general steps:

1. Fork and clone the repo and edit the content as usual.
2. Generate the HTML files by running this script from the root folder: `zsh -i scripts/build_docs.sh`. The HTML files appear in the `docs/_build/html` folder.
3. Copy the `html` folder to a temporary location, and use it to preview your changes. Repeat 1-3 until finished.
4. Push the changes to your fork, and create a pull request (PR) to merge them with the `main` branch of Apple's repo.
5. Checkout the `gh-pages` branch from upstream.
6. Copy the contents of the `html` folder to the root folder, overwriting the existing contents.
7. Commit and push your changes to *your* fork, and submit a PR to merge them with the remote `gh-pages` branch.

At this point you can share the following URL to preview the new version: `https://tonybove-apple.github.io/coremltools/`.


#### Fork and clone

The following are the specific steps in git corresponding to the first part of general step 1:

1. With your GitHub ID for `sample-user`, fork and clone the `coremltools` repo, change to your fork clone's root folder (`coremltools/`), and add the upstream repo to the list of remotes:

	```shell
	# Clone your fork to your local machine.
	git clone git@github.com:sample-user/coremltools.git
	cd coremltools
	# Add 'upstream' repo to list of remotes.
	git remote add upstream https://github.com/apple/coremltools.git
	# Verify the new remote named 'upstream'
	git remote -v
	```
	
    You should see the following, with your GitHub ID for `sample-user`:
    
    ```shell
    origin   git@github.com:sample-user/coremltools.git (fetch)
	origin   git@github.com:sample-user/coremltools.git (push)
	upstream      https://github.com/apple/coremltools.git (fetch)
	upstream      https://github.com/apple/coremltools.git (push)
	```

2. Update your fork and create a new branch (`newfeature`) for changes:

    ```shell
	# Fetch from upstream remote.
	git fetch upstream
	# Checkout your master branch and merge upstream.
	git checkout master
	git merge upstream/master
	# Create a new branch named newfeature.
	git branch newfeature
	# Switch to your new branch.
	git checkout newfeature
	```

#### Install Sphinx and theme

In the `coremltools` clone root directory, install or upgrade Sphinx, the theme, and numpydoc for this environment if you haven’t already done so:

```shell
conda install sphinx
conda install numpydoc
conda install sphinx_rtd_theme
```


#### Edit the content as usual

The following are the specific steps in git corresponding to the last part of general step 1:

1. Make your editing changes, check status to see the files that changed, and add your changes for the commit.
    
    ```shell
    git status
    # You should see the files you changed. Add them all:
    git add --all
    ```

2. Make the commit.
    
    ```shell
    git commit -m "Commit message"
    ```

#### Generate and preview the HTML files

The following are the specific steps in git corresponding to general steps 2-4:

1. From the root folder (`coremltools/`), run the following script:

	```shell
	zsh -i scripts/build_docs.sh
	```

	The HTML files appear in the `docs/_build/html/` folder.

2. Copy the `html/` folder to a temporarily location on your computer (outside of the `coremltools` clone controlled by git).

3. Preview your changes by right-clicking the `index.html` file in the `html/` folder and choosing **Safari** (or another browser). 

4. Repeat all steps from [Edit the content as usual](#edit-the-content-as-usual) to this step until you are satisfied with the preview.

5. Push your changes to your fork, and on your fork, request a Pull Request (PR) for your changes. Once approved, these changes will eventually be merged from your `newfeature` branch to the `main` branch. Do not delete your local branch yet.


#### Copy the HTML to gh-pages

The following are the specific steps in git corresponding to general steps 5-7:

1. Switch to the root folder (`coremltools/`), check out the `master` branch, and then checkout the `gh-pages` branch from upstream:

	```shell
	git checkout master
	git checkout -b gh-pages upstream/gh-pages
	```

2. If the following folders appear in the root folder (or in a `git status` command), remove them using the Finder:

	```
	coremltools.egg-info/
	coremltools/
	docs/
	envs/
    ```

3. Copy the entire contents of the  `html/` folder from your temporarily location to the root directory (`coremltools/`), overwriting any existing files and folders with the same names.

4. Add and commit your changes, and push your changes to *your* fork: 

	```shell
	git add --all
	git commit -m "Commit message for gh-pages"
	git push origin gh-pages
	```

	At this point you can share the following URL to preview the new version (substituting your user name for `sample-user`): `https://sample-user.github.io/coremltools/`.

5. On your fork, request a Pull Request (PR) for your changes to merge them with the `gh-pages` branch on the Apple repo. Once approved, these changes will eventually be merged from your `gh-pages` branch to the Apple repo's `gh-pages` branch. 


## Generating a new API reference

This section is for generating a new version of the API Reference from scratch, and also shows how to edit the Sphinx navigation and content selection elements.


### Set up the environment

You need to do this only once.

1. Start by creating a Conda environment and installing Sphinx and the theme:

	```shell
	conda create coremltools-env
	conda activate coremltools-env
	```

2. Follow the steps in [Fork and clone](#fork-and-clone) including creating and checking out your `newfeature` branch.


3. Install the following in the root (`coremltools/`) folder:

	```shell
	conda install sphinx
	conda install numpydoc
	conda install sphinx_rtd_theme
	```

4. Create a new `docs/` folder in the root folder:

	```shell
	mkdir docs
	```


### Set the Sphinx options

1. Change to the `docs/` folder and start `sphinx-quickstart`:

	```shell
	cd docs
	sphinx-quickstart
	```

2. Accept the defaults for all options except the project name and author name:

	```
	> Project name: coremltools API Reference
	> Author name(s): Apple Inc
	```

	This finishes by adding the following to the `docs/` folder (including the hidden `.gitignore`, still there from before):

	```
	_build
	_static
	_templates
	conf.py
	index.rst
	make.bat
	Makefile
	```

3. Set the Sphinx configuration by editing the `docs/conf.py` file:

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

4. From the `docs/` folder, run `sphinx-apidoc` to create the `source/` folder with the `.rst` files for Sphinx doc generation:

	```shell
	sphinx-apidoc -o source/ ../coremltools
	```

### Edit the RST files {#edit-the-rst-files}

1. Edit the `.rst` files in your local `source/` folder, replacing them with the contents of the [existing versions in the repo](https://github.com/apple/coremltools/tree/master/docs/source):

	```
	coremltools.converters.rst
	coremltools.converters.keras.rst
	coremltools.converters.libsvm.rst
	coremltools.converters.mil.rst
	coremltools.converters.mil.mil.ops.defs.rst
	coremltools.converters.onnx.rst
	coremltools.converters.sklearn.rst
	coremltools.converters.xgboost.rst
	coremltools.models.neural_network.rst
	coremltools.models.rst
	```
    
    Make editing changes to the above files as needed to change navigation and content.
    
2. Create `coremltools.converters.mil.input_types.rst` in the `source/` folder with the [existing version in the repo](https://github.com/apple/coremltools/tree/master/docs/source/coremltools.converters.mil.input_types.rst).

3. Delete all other `.rst` files in your local `source/` folder. You should now have the following files in your local `source/` folder:

	```
	coremltools.converters.keras.rst
	coremltools.converters.libsvm.rst
	coremltools.converters.mil.input_types.rst
	coremltools.converters.mil.mil.ops.defs.rst
	coremltools.converters.mil.rst
	coremltools.converters.onnx.rst
	coremltools.converters.rst
	coremltools.converters.sklearn.rst
	coremltools.converters.xgboost.rst
	coremltools.models.neural_network.rst
	coremltools.models.rst
	modules.rst
	```

4. Switch back to the `docs/` folder, and edit the `index.rst` file as follows (or make changes as needed):

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
   Core ML Format Specification <https://mlmodel.readme.io/reference>
   GitHub <https://github.com/apple/coremltools>
   
	```

### Generate the HTML

1. From the root folder (`coremltools/`), run the following script:

	```shell
	zsh -i scripts/build_docs.sh
	```

	The HTML files appear in the `docs/_build/html/` folder.

2. Copy the `html/` folder to a temporarily location on your computer (outside of the `coremltools` clone controlled by git).

3. Preview your changes by right-clicking the `index.html` file in the `html/` folder and choosing **Safari** (or another browser). 


### Iterate on this process

To iterate on this process, switch to the `docs/` folder, run `make clean` to delete the old HTML files, and change back to the `coremltools/` root folder:

	```shell
	cd docs
	make clean
	cd ..
	```

Then repeat the steps in [Edit the RST files](#edit-the-rst-files).


### Finish the process

When finished with editing and previewing, follow these steps:

1. Check status to see the files that changed, and add your changes for the commit.
    
    ```shell
    git status
    # You should see the files you changed. Add them all:
    git add --all
    ```

2. Push your changes to your fork, and on your fork, request a Pull Request (PR) for your changes. Once approved, these changes will eventually be merged from your `newfeature` branch to the `main` branch. Do not delete your local branch yet.

3. Follow the steps in [Copy the HTML to gh-pages](#copy-the-html-to-gh-pages).






