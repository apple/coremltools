Building from Source
====================

This page describes how to build Core ML Tools (coremltools) from the source repository.

## Requirements

To build coremltools from source, you need the following:

* [CMake](https://cmake.org/)
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [Anaconda](https://www.anaconda.com/))
* [Zsh shell](http://zsh.sourceforge.net/) (the default shell for macOS 10.16+) installed in /usr/bin
* A C++17 compatible compiler (if using GCC, need GCC 9.0 or later)

## Build script

Follow these steps:

1. Fork and clone the GitHub [coremltools repository](https://github.com/apple/coremltools).

2. Run the [build.sh](scripts/build.sh) script to build `coremltools`.
	* By default this script uses Python 3.7, but you can include `--python=3.8` (or `3.9`, `3.10`, `3.11`, `3.12`) as a argument to change the Python version.
	* The script creates a new `build` folder with the coremltools distribution, and a `dist` folder with Python wheel files.

3. Run the [test.sh](scripts/test.sh) script to test the build.

**Under the hood**: If an Anaconda or Miniconda environment doesn't already exist or is not up-to-date, the `build.sh` script automatically runs the [`env_create.sh`](scripts/env_create.sh) script to create the environment. It then uses [`env_activate.sh`](scripts/env_activate.sh) to activate the environment and set up the appropriate version of Python. The new environment is located at `<repo root>/coremltools/envs` and is named after the `py` parameter. For example, a development environment with py 3.7 is named `coremltools-dev-py37`.


## Build targets

The following build targets help you configure the development environment. If you need to add packages, edit the `reqs/pip` files, and the auto-environment script installs them automatically.


* `build` | Build coremltools in debug mode (include symbols).
* `docs` | Build documentation.
* `clean` | Clean build dir.
* `clean_envs` | Delete all `envs` created by the scripts.
* `lint` | Linter.
* `proto` | Build coremltools and rebuild MLModel protobuf sources.
* `release` | Set up the package for release, but don't upload to pypi. Include all wheels from `build/dist` in the built package.
* `style` | Style checking.
* `test` | Run all tests. Pass `TEST_PACKAGES="..."` to set which packages to test.
* `test_fast` | Run all fast tests.
* `test_slow` | Run all non-fast tests.
* `wheel` | Build wheels in release mode.

The script uses Python 3.7, but you can include `--python=3.8` (or `3.9`, `3.10`, `3.11`, `3.12`) as a argument to change the Python version.

## Resources

For more information, see the following:

* Core ML Tools [README](README.md) file for this repository
* [Release Notes](https://github.com/apple/coremltools/releases/) for the current release and previous releases
* [Guides and examples](https://coremltools.readme.io/) with installation and troubleshooting help
* [API Reference](https://apple.github.io/coremltools/index.html)
* [Core ML Specification](https://apple.github.io/coremltools/mlmodel/index.html)
* [Contribution Guidelines](CONTRIBUTING.md) for reporting issues and making pull requests
