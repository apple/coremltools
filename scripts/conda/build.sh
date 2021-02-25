#!/bin/bash

set -e
set -x
mkdir build
cd build

cmake \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} \
    -DPYTHON_EXECUTABLE:FILEPATH=${PREFIX}/bin/python \
    -DPYTHON_INCLUDE_DIR=$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["include"])') \
    -DPYTHON_LIBRARY=${PREFIX}/lib \
    ..
make -j ${CPU_COUNT}

${PYTHON} -m pip install --no-deps --ignore-installed ../
