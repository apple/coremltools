#!/bin/bash

set -e
set -x
mkdir build
cd build

cmake \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} \
    -DPython3_EXECUTABLE:FILEPATH=${PREFIX}/bin/python \
    ..
make -j ${CPU_COUNT}

${PYTHON} -m pip install --no-deps --ignore-installed ../
