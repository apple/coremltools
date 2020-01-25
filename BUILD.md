### Building from Source

To build the project, you need [CMake](https://cmake.org) to configure the project.

```shell
./configure
```

When several python virtual environments are installed, it may be useful to
point to the correct intended version of python (see `./configure --help` for
options)

The configure script creates a virtual environment. You can source the right
environment variables using

```
source ./scripts/python_env.sh
```

Now you can build the package as follows:

```shell
cd build
cmake ../
```
after which you can use make to build the project.

```shell
make -j3
```

### Running Unit Tests

In order to run unit tests, you need `pytest`, `pandas`, and `h5py`.

```shell
pytest -rfs -m "not slow" coremltools/test
```

### Building Installable Wheel

To make a wheel/egg from scratch that you can distribute, you can do the following:

```shell
./scripts/make_wheel.sh
```
A python wheel is put into the `build/dist` folder.

### Building Documentation

First install all external dependencies.

```shell
pip install Sphinx==1.8.5 sphinx-rtd-theme==0.4.3 numpydoc==0.9.1
pip install -e git+git://github.com/michaeljones/sphinx-to-github.git#egg=sphinx-to-github
```

You also must have the *coremltools* package install, see the *Building* section.

Then from the root of the repository:

```shell
cd docs
make html
open _build/html/index.html
```
