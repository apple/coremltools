#
# Before invoking `make`, make sure the python environment is setup by following the instructions
# in README.md.
#
SHELL := /bin/zsh

# All the packages that we care about for linting, testing, etc.
SRC_PACKAGES=coremltools_internal
TEST_PACKAGES=coremltools
PACKAGES=${SRC_PACKAGES} ${TEST_PACKAGES} ${EXAMPLES}

.PHONY: all lint test style checkstyle run_examples dist local_tests remote_tests

PY_EXE ?= $(shell command -v python || command -v python)

python = "3.7"

all: build

checkstyle:
	${PY_EXE} -m yapf -rdp ${PACKAGES}

clean:
	rm -rf build

clean_envs:
	rm -rf envs

lint:
	${PY_EXE} -m pylint -j 0 ${PACKAGES}

build:
	zsh -i scripts/build.sh --python=${python} --debug

env:
	zsh -i scripts/env_create.sh --python=${python} --dev

env_force:
	zsh -i scripts/env_create.sh --python=${python} --force --dev

proto:
	zsh -i scripts/build.sh --python=${python} --protobuf --debug

wheel:
	zsh -i scripts/build.sh --python=${python} --dist

style:
	${PY_EXE} -m yapf -rip --verify ${PACKAGES}

test:
	${PY_EXE} -m pytest ${TEST_PACKAGES} $(addprefix --cov ,${SRC_PACKAGES})

test_fast:
	${PY_EXE} -m pytest ${TEST_PACKAGES} -m 'not slow' $(addprefix --cov ,${SRC_PACKAGES})

test_slow:
	${PY_EXE} -m pytest ${TEST_PACKAGES} -m 'slow' $(addprefix --cov ,${SRC_PACKAGES})
