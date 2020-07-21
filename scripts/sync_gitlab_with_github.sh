#!/bin/bash

set -e

##=============================================================================
## Main configuration processing
COREMLTOOLS_HOME=$( cd "$( dirname "$0" )/.." && pwd )
COREMLTOOLS_NAME=$(basename $COREMLTOOLS_HOME)

# Make sure we have an API token
if [[ -z "$COREMLTOOLS_GITHUB_API_TOKEN" ]]; then
    echo "Expected COREMLTOOLS_GITHUB_API_TOKEN environment variable to be defined. Exiting."
    exit 1
fi

# Create and set up Python env
# Copied from test.sh
cd ${COREMLTOOLS_HOME}
zsh -i -e scripts/env_create.sh --python=3.7
source scripts/env_activate.sh --python=3.7
echo
echo "Using python from $(which python)"
echo

# Install PyGithub and gitpython
$PIP_EXECUTABLE install PyGithub gitpython

# Now run sync_gitlab_with_github.py, which contains the real logic
$PYTHON_EXECUTABLE scripts/sync_gitlab_with_github.py