### Makefile

CoreMLTools development is centered around a provided makefile.

To build the project, you need [CMake](https://cmake.org) and [Ananconda](https://www.anaconda.com/) (or miniconda) to configure the project.
*Our makefile & scripts require the **zsh** shell (now the default shell for MacOS) installed in `/usr/bin`.*

*The following targets will handle the development environment for you. If you need to add packages, edit the reqs/pip files and the auto-environment will install them automatically.*

* `build` | Build coremltools in *debug* mode (include symbols).
* `docs` | Build documentation.
* `clean` | Clean build dir.
* `clean_envs` | Delete all envs created by the scripts.
* `lint` | Linter.
* `proto` | Build coremltools and rebuild MLModel protobuf sources.
* `release` | Setup the package for release, but don’t upload to pypi. Include all wheels from build/dist in the built package.
* `style` | Style checking.
* `test` | Run all tests. Pass TEST_PACKAGES="..." to set which packages to test.
* `test_fast` | Run all fast tests.
* `test_slow` | Run all non-fast tests.
* `wheel` | Build wheels in *release* mode.

For many of the above, you can pass `python=3.7` (or 2.7 / 3.8 / etc.)  as a argument to change the env / build / wheel python version.

*Using an unmanaged developer environment*

Use `make env` to create an auto-set-up development environment with the correct package dependencies.
This env will not be changed by scripts after creation. However, provided scripts & makefiles
do not currenlty support custom development environments; rather, they will always auto-activate the managed environment.
 
## Where are generated conda environments stored?

Environments are generated and stored at `<repo root>/envs`. They're named as follows:
* environment automatically used in scripts: `envs/coremltools-py<version string>`
* `make env`-created environment: `envs/coremltools-dev-py<version string>`
