### Building

This page describes building Core ML Tools (coremltools) from the source repository.

## Requirements

To build coremltools from source, you need the following:

* [CMake](https://cmake.org/)
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [Anaconda](https://www.anaconda.com/))
* [Zsh shell](http://zsh.sourceforge.net/) (the default shell for macOS 10.16+) installed in /usr/bin

## Build script

Follow these steps:

1. Fork and clone the GitHub [coremltools repository](https://github.com/apple/coremltools).
2. Run the [build.sh](scripts/build.sh) script to activate the appropriate release environment and build `coremltools`. The build.sh script uses Python 3.7, but you can include `--python=3.5` (or `3.6`, `3.8`, and so on) as a argument to change the Python version. The script creates a new `build` folder with the coremltools distribution, and a `dist` folder with Python wheel files.
3. Run the [test.sh](scripts/test.sh) script to test the build.

Both scripts automatically run the `env_create` and `env_activate` scripts. To examine them, see the following:

* [`env_create.sh`](scripts/env_create.sh): Reads the required packages in `coremltools/reqs` and installs them into a new environment. This new environment is located at `<repo root>/coremltoos/envs`. The environment is named after the `py` parameter. For example, a development environment with py 3.7 is named `coremltools-dev-py37`.
* [`env_activate.sh`](scripts/env_activate.sh): Activates the new environment with `conda activate`.

## Build targets

The following build targets help you configure the development environment. If you need to add packages, edit the `reqs/pip` files, and the auto-environment script installs them automatically.


* `build` | Build coremltools in debug mode (include symbols).
* `docs` | Build documentation.
* `clean` | Clean build dir.
* `clean_envs` | Delete all `envs` created by the scripts.
* `lint` | Linter.
* `proto` | Build coremltools and rebuild MLModel protobuf sources.
* `release` | Set up the package for release, but don’t upload to pypi. Include all wheels from `build/dist` in the built package.
* `style` | Style checking.
* `test` | Run all tests. Pass `TEST_PACKAGES="..."` to set which packages to test.
* `test_fast` | Run all fast tests.
* `test_slow` | Run all non-fast tests.
* `wheel` | Build wheels in release mode.

The script uses Python 3.7, but you can include `--python=3.5` (or `3.6`, `3.8`, and so on) as a argument to change the Python version.

## Resources

For more information, see the following:

* Core ML Tools [README](README.md) file for this repository
* [Release Notes](https://github.com/apple/coremltools/releases/) for the current release and previous releases
* [Guides and examples](https://coremltools.readme.io/) with installation and troubleshooting
* [API Reference](https://coremltools.readme.io/reference/convertersconvert)
* [Core ML Specification](https://mlmodel.readme.io/)
* [Contribution Guidelines](CONTRIBUTING.md) for reporting issues and making requests
* Third-party repositories: 
    * [onnx-coreml](https://github.com/onnx/onnx-coreml): Convert ONNX models into the Core ML format.
    * [tf-coreml](https://github.com/tf-coreml/tf-coreml): Convert from TensorFlow to CoreML.

