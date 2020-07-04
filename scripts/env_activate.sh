#!/bin/bash

####  Usage ####
# source env_activate.sh
COREMLTOOLS_HOME=$( cd "$( dirname "$0" )/.." && pwd )
COREMLTOOLS_NAME=$(basename $COREMLTOOLS_HOME)
PYTHON=3.7
ENV_DIR="${COREMLTOOLS_HOME}/envs"
DEV=0

function print_help {
  echo "Activates the build environment for the specified python version."
  echo
  echo "Usage: source env_activate <options>"
  echo
  echo "  --dev                Init an environment setup for development."
  echo "  --python=*           Python to use for configuration."
  echo
  echo "Example: source env_activate --python=3.7"
  echo
  exit 1
} # end of print help

function unknown_option {
  echo "Unrecognized option: $1"
  echo "To get help, run source env_activate --help"
  exit 1
} # end of unknown option

###############################################################################
#
# Parse command line configure flags ------------------------------------------
#
while [ $# -gt 0 ]
  do case $1 in
    --python=*)            PYTHON=${1##--python=} ;;
    --dev)                 DEV=1 ;;
    --help)                print_help ;;
    *) unknown_option $1 ;;
  esac
  shift
done

if [[ $DEV == 1 ]]; then
    PYTHON_ENV="${ENV_DIR}/${COREMLTOOLS_NAME}-dev-py${PYTHON}"
else
    PYTHON_ENV="${ENV_DIR}/${COREMLTOOLS_NAME}-py${PYTHON}"
fi

# python executable
export PYTHON_EXECUTABLE=$PYTHON_ENV/bin/python
export PYTHON_VERSION=$($PYTHON_EXECUTABLE -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
export PYTEST_EXECUTABLE=$PYTHON_ENV/bin/pytest
export PIP_EXECUTABLE=$PYTHON_ENV/bin/pip
export PYTHON_LIBRARY=$PYTHON_ENV/lib/libpython${PYTHON}m.dylib
if [[ ${PYTHON_VERSION:0:1} == "3" ]];
then
    if [[ ${PYTHON_VERSION:2:3} -ge 8 ]]; then
        export PYTHON_INCLUDE_DIR=$PYTHON_ENV/include/python${PYTHON_VERSION}/
    else
        export PYTHON_INCLUDE_DIR=$PYTHON_ENV/include/python${PYTHON_VERSION}m/
    fi
else 
    export PYTHON_INCLUDE_DIR=$PYTHON_ENV/include/python${PYTHON_VERSION}/
fi

# Print it out
echo "Export environment variables"
echo PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE
echo PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR
echo PYTEST_EXECUTABLE=$PYTEST_EXECUTABLE
echo PIP_EXECUTABLE=$PIP_EXECUTABLE
echo PYTHON_VERSION=$PYTHON_VERSION
echo PYTHON_LIBRARY=$PYTHON_LIBRARY
echo 

echo "DEBUG: shell is $SHELL"
echo "Making sure conda hooks are enabled"
eval "$(conda shell.$SHELL hook)"
echo "Activating conda env in $PYTHON_ENV"
conda activate $PYTHON_ENV
echo
