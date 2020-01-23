#!/bin/bash

set -e

##=============================================================================
## Main configuration processing
COREMLTOOLS_HOME=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..
BUILD_DIR="${COREMLTOOLS_HOME}/build"

# command flag options
NUM_PROCS=1
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
  echo "  --num_procs=n (default 1)       Specify the number of proceses to run in parallel."
  echo "  --python=*                      Python to use for configuration."
  echo
  exit 1
} # end of print help

# command flag options
# Parse command line configure flags ------------------------------------------
while [ $# -gt 0 ]
  do case $1 in
    --python=*)          PYTHON=${1##--python=} ;;
    --num-procs=*)       NUM_PROCS=${1##--num-procs=} ;;
    --help)              print_help ;;
    *) unknown_option $1 ;;
  esac
  shift
done

# First configure
echo
echo "Configuring using python from $PYTHON"
echo
echo ${COREMLTOOLS_HOME}
cd ${COREMLTOOLS_HOME}
bash -e configure --python=$PYTHON

# Setup the right python
source scripts/python_env.sh
echo
echo "Using python from $(which python)"
echo

# Create a directory for building the artifacts
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure CMake correctly
ADDITIONAL_CMAKE_OPTIONS=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_PROCS=$(sysctl -n hw.ncpu)
    ADDITIONAL_CMAKE_OPTIONS=" -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13"
else
    NUM_PROCS=$(nproc)
fi

# Call CMake
cmake $ADDITIONAL_CMAKE_OPTIONS -DCMAKE_BUILD_TYPE=Release\
 -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}\
 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}\
 -DPYTHON_LIBRARY=${PYTHON_LIBRARY} ..

# Make the python wheel
make -j${NUM_PROCS}
make dist
