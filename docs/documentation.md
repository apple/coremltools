Core ML Tools Documentation
===========================

Before updating any documentation, fork the [coremltools repository](https://github.com/apple/coremltools), clone your fork to your computer, and add the remote upstream:

```shell
git remote add upstream https://github.com/apple/coremltools.git
```

Create a branch for the pull request for your editing changes, and make your changes in that branch. For instructions, see [GitHub Standard Fork & Pull Request Workflow](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

Core ML Tools documentation is organized as follows:

* [Guide and Examples](https://apple.github.io/coremltools/docs-guides/) for learning Core ML Tools. To update, see [Updating the Guide and Examples Pages](#updating-the-guide-and-examples-pages).
* [API Reference](https://apple.github.io/coremltools/index.html) that describes the coremltools API. To update, see [Updating the API Reference](#updating-the-api-reference).
* [Core ML Format Specification](https://apple.github.io/coremltools/mlmodel/index.html) that describes the protobuf message definitions that comprise the Core ML model format. To update, see [Updating the Core ML Format Specification](#updating-the-core-ml-format-specification).

In addition, the coremltools repository includes the following:

* [Core ML Tools README](https://github.com/apple/coremltools/blob/main/README.md) file for this repository.
* [Release Notes](https://github.com/apple/coremltools/releases/) for the current release and previous releases.


## Updating the Guide and Examples Pages

The Markdown files for [Guide and Examples](https://apple.github.io/coremltools/docs-guides/index.html) pages are organized in the `docs-guides/source` folder. The `docs-guides/index.rst` file provides the table of contents and left-column navigation. To make editing changes, follow these general steps:

1. Fork and clone the repository and create a branch for your changes.

2. To preview the HTML, in the `docs-guides` folder enter `make clean` followed by `make html`. The Sphinx-generated HTML files appear in the `docs-guides/_build/html` folder. Open the preview by double-clicking `docs-guides/_build/html/index.html`.

3. When satisfied with your preview, copy the `html` folder to a temporarily location (outside of the `coremltools` clone controlled by git).

4. Commit and push your changes.

5. Create and submit a pull request as described in [Contributions](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#contributions) and add the **docs** label to it (see [Labels](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#labels) for details).

6. When the pull request is approved and merged, switch to the root folder (`coremltools`), check out the `main` branch, and then checkout the `gh-pages` branch from upstream:
    
    ```shell
    git checkout main
    git checkout -b gh-pages upstream/gh-pages
    ```

7. Copy the entire contents of your new  `html` folder from Step 3 to the Guide and Examples directory (`docs-guides`), overwriting any existing files and folders with the same names.

8. Add and commit your changes, and push your changes to *your* fork:
    
    ```shell
    git add --all
    git commit -m "Commit message for gh-pages"
    git push origin gh-pages
    ```
    
    At this point you can share the following URL to preview the new version (substituting your user name for `sample-user`): `https://sample-user.github.io/coremltools/docs-guides/`.

9. On your fork, switch to the `gh-pages` branch, and create a pull request to merge your changes with the `gh-pages` branch on the Apple repo.


## Updating the Core ML Format Specification

The ReStructured Text (`.rst`) files for the [Core ML Format Specification](https://apple.github.io/coremltools/mlmodel/index.html) are organized in the `mlmodel/docs/Format` folder. Synchronize your editing changes with the corresponding protobuf files in the `mlmodel/format` folder. The `mlmodel/docs/index.rst` file provides the table of contents and left-column navigation.

To make editing changes, follow these general steps:

1. Fork and clone the repository and create a branch for your changes.

2. To preview the HTML, in the `mlmodel/docs` folder enter `make clean` followed by `make html`. The Sphinx-generated HTML files appear in the `mlmodel/docs/_build/html` folder. Open the preview by double-clicking `mlmodel/docs/_build/html/index.html`.

3. When satisfied with your preview, copy the `html` folder to a temporarily location (outside of the `coremltools` clone controlled by git).

4. Commit and push your changes.

5. Create and submit a pull request as described in [Contributions](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#contributions) and add the **docs** label to it (see [Labels](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#labels) for details).

6. When the pull request is approved and merged, switch to the root folder (`coremltools`), check out the `main` branch, and then checkout the `gh-pages` branch from upstream:
    
    ```shell
    git checkout main
    git checkout -b gh-pages upstream/gh-pages
    ```

7. Copy the entire contents of your new  `html` folder from Step 3 to the Core ML Format Specification directory (`mlmodel/docs`), overwriting any existing files and folders with the same names.

8. Add and commit your changes, and push your changes to *your* fork:
    
    ```shell
    git add --all
    git commit -m "Commit message for gh-pages"
    git push origin gh-pages
    ```
    
    At this point you can share the following URL to preview the new version (substituting your user name for `sample-user`): `https://sample-user.github.io/coremltools/mlmodel/`.

9. On your fork, switch to the `gh-pages` branch, and create a pull request to merge your changes with the `gh-pages` branch on the Apple repo.


## Updating the API Reference

To update the `docstrings` in the source files for the [API Reference](https://apple.github.io/coremltools/index.html), and then generate the HTML for the documentation site from the `docstrings`, follow these general steps:

1. In your clone `coremltools` (root) directory, create a virtual environment or Miniconda environment (for instructions see [Installing Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/installing-coremltools.html)). Then do the following:

    a. Install the requirements for building coremltools, and for building the documentation:
    
    ```shell
    pip install -r reqs/build.pip
    pip install -r reqs/docs.pip
    ```
    
    b. Install coremltools from the repo:
    
    ```shell
    pip install -e .
    ```

2. Edit the `docstrings` in the API source files. You can also edit the `coremltools/docs/index.rst` file and other `.rst` files in the `coremltools/docs/source` folder to establish the documentation layout and navigation. 

3. [Generate HTML for a Preview](#generate-html-for-a-preview).

4. Commit and push your changes, and create a pull request for your changes as described in [Contributions](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#contributions). Submit your pull request when finished.

5. Create and submit another PR for the generated HTML files. See [Copy the HTML to gh-pages](#copy-the-html-to-gh-pages) for details.


### Generate HTML for a Preview

1. Navigate to the `docs` folder and run `make clean` to delete any old HTML files (no effect if there are no HTML files).

2. Switch to the `coremltools` root folder (parent of `docs`), and run the following script to generate the HTML:
    
    ```shell
    zsh ./scripts/build_docs.sh
    ```
    
    *Warning*: Include the `./` or the script may not function correctly.

3. The HTML files appear in the `docs/_build/html` folder. Preview your changes by double-clicking `docs/_build/html/index.html`.

4. When satisfied with your preview, copy the `html` folder to a temporarily location (outside of the `coremltools` clone controlled by git).


### Copy the HTML to gh-pages

Perform this step to check the generated `html` folder into the `gh-pages` branch of [this repository](https://github.com/apple/coremltools).

1. Switch to the root folder (`coremltools`), check out the `main` branch, and then checkout the `gh-pages` branch from upstream:
    
    ```shell
    git checkout main
    git checkout -b gh-pages upstream/gh-pages
    ```

2. If the following folders appear in the root folder (or in a `git status` command), remove them:

    ```
    build
    coremltools
    coremltools.egg-info
    deps
    docs
    envs
    ```

   You should now have only the following files and folders in the root folder of the `gh-pages` branch:

   ```
    _downloads
    _examples
    _images
    _modules
    _sources
    _static
    auto_examples
    docs-guides
    genindex.html
    index.html
    mlmodel
    objects.inv
    py-modindex.html
    search.html
    searchindex.js
    source
    v3.4
    v4.1
    v6.3
   ```

3. Copy the entire contents of your new  `html` folder to the root directory (`coremltools`), overwriting any existing files and folders with the same names.

4. Add and commit your changes, and push your changes to *your* fork:
    
    ```shell
    git add --all
    git commit -m "Commit message for gh-pages"
    git push origin gh-pages
    ```
    
    At this point you can share the following URL to preview the new version (substituting your user name for `sample-user`): `https://sample-user.github.io/coremltools/`.

5. On your fork, switch to the `gh-pages` branch, and create a pull request to merge your changes with the `gh-pages` branch on the Apple repo.

