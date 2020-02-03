#!/bin/bash

set -e

##=============================================================================
## Main configuration processing
COREMLTOOLS_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..
BUILD_DIR="${COREMLTOOLS_HOME}/build"

# command flag options
PYTHON=$(which python)

unknown_option() {
  echo "Unknown option $1. Exiting."
  exit 1
}

print_help() {
  echo "Builds the release branch and produce a wheel to the targets directory "
  echo
  echo "Usage: ./make_wheel.sh"
  echo
  echo "  --wheel-path=*          Specify which wheel to test."
  echo "  --python=*              Python to use for configuration."
  echo
  exit 1
} # end of print help

# command flag options
# Parse command line configure flags ------------------------------------------
while [ $# -gt 0 ]
  do case $1 in
    --python=*)          PYTHON=${1##--python=} ;;
    --wheel-path=*)      WHEEL_PATH=${1##--wheel-path=} ;;
    --help)              print_help ;;
    *) unknown_option $1 ;;
  esac
  shift
done

# First configure
echo ${COREMLTOOLS_HOME}
cd ${COREMLTOOLS_HOME}
bash -e configure --python=$PYTHON

# Setup the right python
source scripts/python_env.sh
echo
echo "Using python from $(which python)"
echo

$PIP_EXECUTABLE install ${WHEEL_PATH}
#  (f)ailed, (E)error, (s)skipped, (x)failed, (X)passed (w)pytest-warnings
$PYTEST_EXECUTABLE -r wxXsf -m "not slow" --durations=100 coremltools/test
