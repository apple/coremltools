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
  echo "Builds the docs associated with the code"
  echo
  echo "Usage: ./make_docs.sh"
  echo
  echo "  --wheel-path=*          Specify which wheel to use to make docs."
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
pushd ${COREMLTOOLS_HOME}
bash -e configure --python=$PYTHON --include-docs-deps

# Setup the right python
source scripts/python_env.sh
echo
echo "Using python from $(which python)"
echo

$PIP_EXECUTABLE install ${WHEEL_PATH}
pushd ${COREMLTOOLS_HOME}/docs
make html
popd

echo "Zipping docs"
TARGET_DIR=${COREMLTOOLS_HOME}/build/dist
pushd ${COREMLTOOLS_HOME}/docs/_build/
zip -r ${TARGET_DIR}/docs.zip html
popd

popd
