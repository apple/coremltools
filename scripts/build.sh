#!/usr/bin/env bash

set -e

##=============================================================================
## Main configuration processing
COREMLTOOLS_HOME=$( cd "$( dirname "$0" )/.." && pwd )
BUILD_DIR="${COREMLTOOLS_HOME}/build"

# command flag options
BUILD_MODE="Release"
NUM_PROCS=1
BUILD_PROTO=0
BUILD_DIST=0
BUILD_DIST_DEV=0
PYTHON="3.8"
CHECK_ENV=1

# Defaults, which may be overridden
sdk=macosx

# Grab any settings defined in this folder
build_settings=$( cd "$( dirname "$0" )" && pwd )/build_settings.sh
test -f "${build_settings}" && source "${build_settings}"

unknown_option() {
  echo "Unknown option $1. Exiting."
  exit 1
}

print_help() {
  echo "Builds coremltools and dependent libraries."
  echo
  echo "Usage: build.sh"
  echo
  echo "  --num_procs=n (default 1)       Specify the number of proceses to run in parallel."
  echo "  --python=*                      Python to use for configuration."
  echo "  --protobuf                      Rebuild & overwrite the protocol buffers in MLModel."
  echo "  --debug                         Build without optimizations and stripping symbols."
  echo "  --dist                          Build the distribution (wheel)."
  echo "  --dist-dev                      Build the distribution (wheel) and tag it as a dev release."
  echo "  --no-check-env                  Don't check the environment to verify it's up to date."
  echo
  echo ""
  exit 0
} # end of print help

# command flag options
# Parse command line configure flags ------------------------------------------
while [ $# -gt 0 ]
  do case $1 in
    --python=*)          PYTHON=${1##--python=} ;;
    --num-procs=*)       NUM_PROCS=${1##--num-procs=} ;;
    --protobuf)          BUILD_PROTO=1 ;;
    --debug)             BUILD_MODE="Debug" ;;
    --dist)              BUILD_DIST=1 ;;
    --dist-dev)          BUILD_DIST_DEV=1 ;;
    --no-check-env)      CHECK_ENV=0 ;;
    --help)              print_help ;;
    *) unknown_option $1 ;;
  esac
  shift
done

set -x

# First configure
echo
echo "Configuring using python from $PYTHON"
echo
echo ${COREMLTOOLS_HOME}
cd ${COREMLTOOLS_HOME}
if [[ $CHECK_ENV == 1 ]]; then
    zsh -i -e scripts/env_create.sh --python=$PYTHON --exclude-test-deps
fi

# Setup the right python
source scripts/env_activate.sh --python=$PYTHON
echo
echo "Using python from $(which python)"
echo

# Uninstall any existing coremltools inside the build environment
pip uninstall -y coremltools

# Create a directory for building the artifacts
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure CMake correctly
ADDITIONAL_CMAKE_OPTIONS=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_PROCS=$(sysctl -n hw.ncpu)
    ADDITIONAL_CMAKE_OPTIONS="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15"
else
    NUM_PROCS=$(nproc)
fi

# if BUILD_TAG has not been set for dev wheel, set it with git describe
if [ $BUILD_DIST_DEV -eq 1 ] && [ -z ${BUILD_TAG+x} ]
then
  BUILD_TAG=$(${COREMLTOOLS_HOME}/scripts/build_tag.py)
fi

# Call CMake
CMAKE_COMMAND=""
if [[ $OSTYPE == darwin* ]]; then
  CMAKE_COMMAND="xcrun --sdk ${sdk} "
fi
CMAKE_COMMAND+="/opt/homebrew/opt/cmake/bin/cmake $ADDITIONAL_CMAKE_OPTIONS \
  -DCMAKE_BUILD_TYPE=$BUILD_MODE \
  -DPYTHON_EXECUTABLE:FILEPATH=$PYTHON_EXECUTABLE \
  -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
  -DPYTHON_LIBRARY=$PYTHON_LIBRARY \
  -DOVERWRITE_PB_SOURCE=$BUILD_PROTO \
  -DBUILD_TAG=$BUILD_TAG \
  ${COREMLTOOLS_HOME}"
eval ${CMAKE_COMMAND}

# Make the python wheel
make -j${NUM_PROCS}

# if [ $BUILD_DIST -eq 1 ] || [ $BUILD_DIST_DEV -eq 1 ]
# then
make dist
# fi


