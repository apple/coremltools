#!/bin/bash

####  Usage ####
# source python_env.sh

if [[ -z $BASH_SOURCE ]]; then
        SCRIPT_DIR=$( cd "$( dirname $0)" && pwd )
else
        SCRIPT_DIR=$( cd "$( dirname $BASH_SOURCE )" && pwd )
fi
COREMLTOOLS_HOME=$SCRIPT_DIR/..
BUILD_DIR=${COREMLTOOLS_HOME}/build
COREMLTOOLS_ENV=coremltools-dev
PYTHON_ENV=$BUILD_DIR/$COREMLTOOLS_ENV

# python executable
export PYTHON_EXECUTABLE=$PYTHON_ENV/bin/python
export PYTHON_VERSION=$($PYTHON_EXECUTABLE -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
export PYTEST_EXECUTABLE=$PYTHON_ENV/bin/pytest
export PIP_EXECUTABLE=$PYTHON_ENV/bin/pip
export PYTHON_LIBRARY=$PYTHON_ENV/lib/python${PYTHON_VERSION}/
if [[ ${PYTHON_VERSION:0:1} == "3" ]] ;
then 
    export PYTHON_INCLUDE_DIR=$PYTHON_ENV/include/python${PYTHON_VERSION}m/
else 
    export PYTHON_INCLUDE_DIR=$PYTHON_ENV/include/python${PYTHON_VERSION}/
fi

# Print it out
echo "Export environment variables"
echo BUILD_DIR=$BUILD_DIR
echo PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE
echo PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR
echo PYTEST_EXECUTABLE=$PYTEST_EXECUTABLE
echo PIP_EXECUTABLE=$PIP_EXECUTABLE
echo PYTHON_VERSION=$PYTHON_VERSION
echo PYTHON_LIBRARY=$PYTHON_LIBRARY
echo 

echo "Activating virtualenv in $PYTHON_ENV"
. $PYTHON_ENV/bin/activate
echo
