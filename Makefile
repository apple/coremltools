#
# Before invoking `make`, make sure the python environment is setup by following the instructions
# in README.md.
#
SHELL := /bin/zsh

# All the packages that we care about for linting, testing, etc.
CURR_DIR=$(notdir $(shell pwd))
SRC_PACKAGES=$(subst -,_,${CURR_DIR})
TEST_PACKAGES=${SRC_PACKAGES}
PACKAGES=${SRC_PACKAGES} ${TEST_PACKAGES} ${EXAMPLES}
ENV_DIR=envs/${CURR_DIR}-py${python}

# Docker image for CoreML
DOCKERFILE_PATH=docker/Dockerfile-coremltools-linux
DOCKERFILE_ROOT=docker/
DOCKER_TAG=docker.apple.com/turi/coremltools-linux:latest

# Define python version to use
PY_EXE ?= $(shell command -v python || command -v python)
python = 3.7

.PHONY: all build checkstyle clean clean_envs docker_build docker_push env env_force lint proto release style test test_fast test_slow wheel

all: build

build: ${ENV_DIR}/build_reqs
	zsh -i scripts/build.sh --python=${python} --debug --no-check-env

checkstyle:
	${PY_EXE} -m yapf -rdp ${PACKAGES}

clean:
	rm -rf build

clean_envs:
	rm -rf envs

docker_build:
	docker build -f ${DOCKERFILE_PATH} -t ${DOCKER_TAG} ${DOCKERFILE_ROOT}

docker_push:
	docker push ${DOCKER_TAG}

docs: ${ENV_DIR}/docs_reqs
	zsh -i scripts/build_docs.sh --python=${python} --no-check-env

env:
	zsh -i scripts/env_create.sh --python=${python} --dev

env_force:
	zsh -i scripts/env_create.sh --python=${python} --force --dev --no-check-env

lint:
	${PY_EXE} -m pylint -j 0 ${PACKAGES}

proto: ${ENV_DIR}/build_reqs
	zsh -i scripts/build.sh --python=${python} --protobuf --debug --no-check-env

release: ${ENV_DIR}/build_reqs
	zsh -i scripts/release_wheel.sh --no-check-env

style:
	${PY_EXE} -m yapf -rip --verify ${PACKAGES}

test: ${ENV_DIR}/test_reqs
	zsh -i scripts/test.sh --python=${python} --test-package="${TEST_PACKAGES}" --cov="${SRC_PACKAGES}" --no-check-env

test_fast: ${ENV_DIR}/test_reqs
	zsh -i scripts/test.sh --python=${python} --test-package="${TEST_PACKAGES}" --cov="${SRC_PACKAGES}" --fast --no-check-env

test_slow: ${ENV_DIR}/test_reqs
	zsh -i scripts/test.sh --python=${python} --test-package="${TEST_PACKAGES}" --cov="${SRC_PACKAGES}" --slow --no-check-env

wheel: ${ENV_DIR}/build_reqs
	zsh -i scripts/build.sh --python=${python} --dist --no-check-env

# For Managing Environments, so we don't need to rebuild every time we run a target.
${ENV_DIR}/build_reqs: ./reqs/build.pip
	zsh -i scripts/env_create.sh --python=${python} --exclude-test-deps
	touch ${ENV_DIR}/build_reqs

${ENV_DIR}/test_reqs: ${ENV_DIR}/build_reqs ./reqs/test.pip
	zsh -i scripts/env_create.sh --python=${python} --exclude-build-deps
	touch ${ENV_DIR}/test_reqs

${ENV_DIR}/docs_reqs: ${ENV_DIR}/build_reqs ${ENV_DIR}/test_reqs ./reqs/docs.pip
	zsh -i scripts/env_create.sh --python=${python} --include-docs-deps --exclude-build-deps --exclude-test-deps
	touch ${ENV_DIR}/docs_reqs
