Core ML Tools Documentation
===========================

Changes to our hosted documentation are made by merging changes in the the [gh-pages](https://github.com/apple/coremltools/tree/gh-pages) branch. 

Core ML Tools documentation is organized into three parts:

**1** - [Guide and Examples](https://apple.github.io/coremltools/docs-guides/) for learning Core ML Tools. The source documents are under `docs-guides`. It gets deployed to the `docs-guides` folder of the `gh-pages` branch. 

**2** - [API Reference](https://apple.github.io/coremltools/index.html) that describes the coremltools API. The source documents are under `docs`, and they are deployed to the root of the `gh-pages` branch

**3** - [Core ML Format Specification](https://apple.github.io/coremltools/mlmodel/index.html) that describes the protobuf message definitions that comprise the Core ML model format. The source documents are under `mlmodel/docs`. It gets deployed to the `mlmodel` folder of the `gh-pages` branch.

In addition, the coremltools repository includes the following:

* [Core ML Tools README](https://github.com/apple/coremltools/blob/main/README.md) file for this repository.
* [Release Notes](https://github.com/apple/coremltools/releases/) for the current release and previous releases.


## Environment Setup

```shell
scripts/build.sh --python=3.10
source scripts/env_activate.sh --python=3.10
pip reqs/docs.pip
pip install -r reqs/docs.pip
pip install -e .
```

In order to preview your changes before merging, you need to be working from your own fork of the repository.

## Updating Documentation

After setting up your enviroment, go to the source folder for the part of the documentation you want to update:

Then run:
```
make clean
make html
```

Verify that things look good: `open _build/html/index.html`

Copy generated HTML out of the repository: `cp -r _build/html /tmp/`

Check out the `gh-pages` branch of your fork. Make sure it's up to date with `upstream/gh-pages`.

Copy the update HTML from `/tmp/html` to correct location depending on which of the three parts you are updating.

Add all new files that you copied over. Commit your changes and push.

Put up a pull request from the `gh-pages` of your fork to the `upstream/gh-pages`. Include a preview link for the part you are updating.

[Example pull request](https://github.com/apple/coremltools/pull/2581).


