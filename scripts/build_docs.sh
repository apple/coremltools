#!/bin/bash

set -e

##=============================================================================
## Main configuration processing
COREMLTOOLS_HOME=$( cd "$( dirname "$0" )/.." && pwd )

# command flag options
PYTHON="3.7"
CHECK_ENV=1
WHEEL_PATH=""

unknown_option() {
  echo "Unknown option $1. Exiting."
  exit 1
}

print_help() {
  echo "Builds the docs associated with the code"
  echo
  echo "Usage: zsh -i make_docs.sh"
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
    --python=*)              PYTHON=${1##--python=} ;;
    --wheel-path=*)          WHEEL_PATH=${1##--wheel-path=} ;;
    --help)                  print_help ;;
    *) unknown_option $1 ;;
  esac
  shift
done

cd ${COREMLTOOLS_HOME}

if [[ $PYTHON != "None" ]]; then
  # Setup the right python
  if [[ $CHECK_ENV == 1 ]]; then
    zsh -i scripts/env_create.sh --python=$PYTHON --include-docs-deps --exclude-test-deps
  fi
  source scripts/env_activate.sh --python=$PYTHON
fi

echo
echo "Using python from $(which python)"
echo

if [[ $WHEEL_PATH != "" ]]; then
    $PIP_EXECUTABLE install ${WHEEL_PATH} --upgrade
else
    cd ..
    $PIP_EXECUTABLE install -e coremltools --upgrade
    cd ${COREMLTOOLS_HOME}
fi

cd docs
make html
cd ..

pip uninstall -y coremltools # We're using the build env for this script, so uninstall the wheel when we're done.
