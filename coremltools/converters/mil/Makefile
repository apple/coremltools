# Before invoking `make`, make sure the python environment is setup by following the instructions
# in README.md.
#
SHELL := /bin/bash

# All the packages that we care about for linting, testing, etc.
SRC_PACKAGES=.

TF_IOS13_TEST=../tensorflow/test
MIL_TEST="."
MIL_TEST_INTERNAL="../../../../coremltools-internal/coremltools_internal/converters/mil"

.PHONY: all lint test style checkstyle

PYTHON ?= $(shell command -v python || command -v python)

ifeq ($(PIP_INDEX_URL),)
	PIP_INDEX_URL := "https://pypi.apple.com/simple"
endif

all: checkstyle lint test

style:
	${PYTHON} -m yapf -rip --verify ${SRC_PACKAGES}

checkstyle:
	${PYTHON} -m yapf -rdp ${SRC_PACKAGES}

lint:
	${PYTHON} -m pylint -j 0 ${SRC_PACKAGES}

test:
	${PYTHON} -m pytest -W ignore::DeprecationWarning ${MIL_TEST} ${MIL_TEST_INTERNAL}

test_ref:
	${PYTHON} -m pytest -W ignore::DeprecationWarning ${TF_IOS13_TEST}
