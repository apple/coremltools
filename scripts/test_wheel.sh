#!/bin/bash

set -e

##=============================================================================
## Main configuration processing
COREMLTOOLS_HOME=$( cd "$( dirname "$0" )/.." && pwd )
BUILD_DIR="${COREMLTOOLS_HOME}/build"
TEST_DIR=coremltools/test/

# command flag options
PYTHON="3.7"

unknown_option() {
  echo "Unknown option $1. Exiting."
  exit 1
}

print_help() {
  echo "Test the wheel by running all unit tests"
  echo
  echo "Usage: zsh -i test_wheel.sh"
  echo
  echo "  --wheel-path=*          Specify which wheel to test."
  echo "  --test-package=*        Test package to run."
  echo "  --python=*              Python to use for configuration."
  echo "  --requirements=*        [Optional] Path to the requirements.txt file."
  echo
  exit 1
} # end of print help

# command flag options
# Parse command line configure flags ------------------------------------------
while [ $# -gt 0 ]
  do case $1 in
    --requirements=*)    REQUIREMENTS=${1##--requirements=} ;;
    --python=*)          PYTHON=${1##--python=} ;;
    --test-package=*)    TEST_PACKAGE=${1##--test-package=} ;;
    --wheel-path=*)      WHEEL_PATH=${1##--wheel-path=} ;;
    --help)              print_help ;;
    *) unknown_option $1 ;;
  esac
  shift
done

# First configure
echo ${COREMLTOOLS_HOME}
pushd ${COREMLTOOLS_HOME}
zsh -i -e scripts/env_create.sh --python=$PYTHON

# Setup the right python
source scripts/env_activate.sh --python=$PYTHON
echo
echo "Using python from $(which python)"
echo

$PIP_EXECUTABLE install $~WHEEL_PATH --upgrade

# Install dependencies if specified
if [ ! -z "${REQUIREMENTS}" ]; then
   $PIP_EXECUTABLE install -r "${REQUIREMENTS}"
fi

# Now run the tests
echo "Running tests"

$PYTEST_EXECUTABLE -ra -W "ignore::FutureWarning" -W "ignore::DeprecationWarning" -m "not slow" --durations=100 --pyargs ${TEST_PACKAGE} --junitxml=${BUILD_DIR}/py-test-report.xml
