Core ML Community Tools
=======================

Core ML community tools contains all supporting tools for CoreML model
conversion and validation. This includes Scikit Learn, LIBSVM, Caffe,
Keras and XGBoost.


We recommend using virtualenv to use, install, or build coremltools. Be
sure to install virtualenv using your system pip.

```
    pip install virtualenv
```

Installation
------------

The method for installing *coremltools* follows the
[standard python package installation steps](https://packaging.python.org/installing/).
Once you have set up a python environment, run::

    pip install -U coremltools

The package [documentation](https://apple.github.io/coremltools) contains
more details on how to use coremltools.

Dependencies
------------

*coremltools* has the following dependencies:

- numpy (1.12.1+)
- protobuf (3.1.0+)

In addition, it has the following soft dependencies that are only needed when
you are converting models of these formats:

- Keras (1.2.2, 2.0.4+) with Tensorflow (1.0.x, 1.1.x)
- Xgboost (0.6+)
- scikit-learn (0.15+)
- libSVM


Building from source
--------------------
To build the project, you need [CMake](https://cmake.org) to configure the project

```
  cmake .
```

after which you can use make to build the project

```
  make -j4
```

Building Installable Wheel
---------------------------
To make a wheel/egg that you can distribute, you can do the following

```
   make dist 
```

Running Unit Tests
-------------------
To run the unit tests, from the repo root, run the following command:

```
    make test
```

To add a new unit test, add it to the coremltools/test folder. Make sure you
name the file with a 'test' as the prefix.

Additionally, running unit-tests would require more packages (like
libsvm)

```
   pip install numpy
   pip install scikit-learn
```

To install libsvm

```
   git clone https://github.com/cjlin1/libsvm.git
   cd libsvm/
   make
   cd python/
   make
```

To make sure you can run libsvm python bindings everywhere, you need the
following command, replacing <LIBSVM_PATH> with the path to the root of
your repository.

```
   export PYTHONPATH=${PYTHONPATH}:<LIBSVM_PATH>/python
```

To install xgboost

```
   git clone --recursive https://github.com/dmlc/xgboost
   cd xgboost; cp make/minimum.mk ./config.mk; make -j4
   cd python-package; python setup.py develop --user
```

To install keras (Version >= 2.0)
```
  pip install keras tensorflow
```

If you'd like to use the old keras version, you can:
```
  pip install keras==1.2.2 tensorflow
```


Building Documentation
----------------------
First install all external dependencies.

```
   pip install Sphinx==1.5.3 sphinx-rtd-theme==0.2.4 numpydoc
   pip install -e git+git://github.com/michaeljones/sphinx-to-github.git#egg=sphinx-to-github
```
You also must have the *coremltools* package install, see the *Building* section.

Then from the root of the repository:
```
   cd docs
   make html
   open _build/html/index.html
```
