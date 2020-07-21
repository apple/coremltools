#!/bin/bash

# This script is designed to be run in the Docker container python:3.7.
# For instance, on the command line from the root of this repo:
#
# docker run -ti -v "$PWD:/build" -v "$HOME/.ssh:/root/.ssh" -e COREMLTOOLS_GITHUB_API_TOKEN=$COREMLTOOLS_GITHUB_API_TOKEN registry.gitlab.com/zach_nation/coremltools/sync-gitlab-with-github:1.0 "/build/scripts/sync_gitlab_with_github.sh"
#
# Note that -v "$PWD:/build" mounts your local repo at /build within the container,
#       and -v "$HOME/.ssh:/root/.ssh" mounts your .ssh directory within the container!
# This is needed to `git push` to the CI repo.

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

# Now run sync_gitlab_with_github.py, which contains the real logic
python3 $COREMLTOOLS_HOME/scripts/sync_gitlab_with_github.py