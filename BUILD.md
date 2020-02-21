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
cmake -H. -Bbuild \
  -DPYTHON_EXECUTABLE:FILEPATH=$VIRTUAL_ENV/bin/python \
  -DPYTHON_INCLUDE_DIR=$VIRTUAL_ENV/include/python3.7m \
  -DPYTHON_LIBRARY=$VIRTUAL_ENV/lib
make -C build -j3
```
(Note that if your `virtualenv` uses a different version of Python,
you will need to modify the `-DPYTHON_INCLUDE_DIR` value accordingly.)

This build directory does not have to be identical to the `build` directory
created by `./configure` as it is here.

CMake `POST_BUILD` hooks on shared libraries targets will handle installing the
dev tree into your virtualenv.

### Running Unit Tests

You can run all the unit tests not marked as slow using the following command.

```shell
pip install -e .
pytest -rfs -m '"no slow"' coremltools/test
```

Shortcut targets to rebuild and run all the tests exist as well.
This takes time, so the recommended workflow is to run only relevant tests until
you're confident in a change.

```shell
make -j3 -C build pytest_no_slow
make -j3 -C build pytest
```

See [pytest documentation](https://docs.pytest.org/en/latest/) to learn more
about how to run a single unit test.

### Building wheels
If you would like a wheel to install outside of the virtualenv (or in it), 
use `make -C build dist` and find the resulting wheels in `build/dist/*.whl`.

If you want to build a wheel for distribution or testing, there is a script
that automates all of the steps necessary for building a wheel,
`scripts/make_wheel.sh`, that can be used instead (but make sure to install the
wheel before running unit tests, if you plan to run the tests).

### Building Documentation

The API docs for this package can be build using the following:
```
./scripts/make_docs.sh --wheel-path=[PATH_TO_WHEEL]
```
The API docs are saved at `docs/_build/html`.
