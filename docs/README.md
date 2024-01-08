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

3. Commit and push your changes.

4. Create and submit a pull request as described in [Contributions](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#contributions) and add the **docs** label to it (see [Labels](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#labels) for details).

After the pull request is approved and merged, the changes will appear in the HTML pages as they are updated.


## Updating the Core ML Format Specification

The ReStructured Text (`.rst`) files for the [Core ML Format Specification](https://apple.github.io/coremltools/mlmodel/index.html) are organized in the `mlmodel/docs/Format` folder. Synchronize your editing changes with the corresponding protobuf files in the `mlmodel/format` folder. The `mlmodel/docs/index.rst` file provides the table of contents and left-column navigation.

To make editing changes, follow these general steps:

1. Fork and clone the repository and create a branch for your changes.

2. To preview the HTML, in the `mlmodel/docs` folder enter `make clean` followed by `make html`. The Sphinx-generated HTML files appear in the `mlmodel/docs/_build/html` folder. Open the preview by double-clicking `mlmodel/docs/_build/html/index.html`.

3. Commit and push your changes.

4. Create and submit a pull request as described in [Contributions](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#contributions) and add the **docs** label to it (see [Labels](https://apple.github.io/coremltools/docs-guides/source/how-to-contribute.html#labels) for details).

After the pull request is approved and merged, the changes will appear in the HTML pages as they are updated.


## Updating the API Reference

To update the `docstrings` in the source files for the [API Reference](https://apple.github.io/coremltools/index.html), follow these general steps:

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

After the pull request is approved and merged, the changes will appear in the HTML pages as they are updated.


### Generate HTML for a Preview

1. Navigate to the `docs` folder and run `make clean` to delete any old HTML files (no effect if there are no HTML files).

2. Switch to the `coremltools` root folder (parent of `docs`), and run the following script to generate the HTML:
    
    ```shell
    zsh ./scripts/build_docs.sh
    ```
    
    *Warning*: Include the `./` or the script may not function correctly.

3. The HTML files appear in the `docs/_build/html` folder. Preview your changes by double-clicking `docs/_build/html/index.html`.


