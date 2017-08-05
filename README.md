Core ML Community Tools
=======================

Core ML community tools contains all supporting tools for CoreML model
conversion and validation. This includes Scikit Learn, LIBSVM, Caffe,
Keras and XGBoost.

Dependencies
------------

We use virtualenv to install required python packages into the build
directory, be sure to install virtualenv using your system pip.

```
    pip install virtualenv
```

Building for running Unit Tests
--------------------------------
We require nosetests. You can install it with the following command.

```
    pip install nose
```

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

If you'd like to use the old keras version 1.2 APIs, 
```
  pip install keras==1.2.2 tensorflow
```

Running Unit Tests
-------------------
To run the unit tests, from the repo root, run the following command:

```
   nosetests coremltools/test/
```

To add a new unit test, add it to the coremltools/test folder. Make sure you
name the file with a 'test' as the prefix.


Building Installable Wheel
---------------------------
```
   python setup.py bdist_wheel
```

Building Documentation
----------------------
First install all external dependencies.

```
   pip install Sphinx==1.5.3 sphinx-rtd-theme==0.2.4
```
You also must have the `coremltools` package install, see the `Building` section.

Then from the root of the repository:
```
   cd docs
   make html
   open _build/html/index.html
```
