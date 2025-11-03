cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(coremltools)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
    Source directory '${PROJECT_SOURCE_DIR}' is the same
    as binary directory '${PROJECT_BINARY_DIR}'; coremltools requires
    an out-of-source build.  Note that your directory tree will require
    you to remove CMakeCache.txt before this will work, and CMake may have
    clobbered some source files (use git reset --hard).
    See: https://gitlab.kitware.com/cmake/community/wikis/FAQ#i-run-an-out-of-source-build-but-cmake-generates-in-source-anyway-why
  ")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Globally ignore "no symbols" warnings during compilation
SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
if(APPLE)
    SET(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
endif()

find_program(HAS_CCACHE ccache)
if(HAS_CCACHE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

add_subdirectory(deps)
add_subdirectory(mlmodel)

find_package(PythonInterp)
find_package(PythonLibs)

message("Found python at ${PYTHON_EXECUTABLE}")
message("Found python version ${PYTHON_VERSION_STRING}")
message("Found python includes ${PYTHON_INCLUDE_DIRS}")

include_directories(
  .
  deps/protobuf/src
  deps/pybind11/include
  deps/nlohmann
  mlmodel/src
  ${PYTHON_INCLUDE_DIRS}
  )

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")
endif()

add_library(milstoragepython
  SHARED
  milstoragepython/MilStorage.cpp
  milstoragepython/MilStoragePython.cpp
  )

target_link_libraries(milstoragepython
  mlmodel
  )

add_library(modelpackage
  SHARED
  modelpackage/src/ModelPackage.cpp
  modelpackage/src/utils/JsonMap.cpp
  modelpackage/src/ModelPackagePython.cpp
  )

target_compile_definitions(modelpackage
  PRIVATE
  CPU_ONLY=1
  )

target_link_libraries(modelpackage
  mlmodel
  libprotobuf
  )

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  target_link_libraries(modelpackage
    stdc++fs
    )
endif()

if (APPLE)
  # Allow Python to be found at runtime instead of compile/link time
  # This is apparently the default on Linux
  set_target_properties(milstoragepython PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  set_target_properties(modelpackage PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

file(COPY ${CMAKE_SOURCE_DIR}/README.md DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/coremltools/__init__.py
  DESTINATION ${CMAKE_BINARY_DIR}/coremltools)
file(COPY ${CMAKE_SOURCE_DIR}/coremltools/version.py
  DESTINATION ${CMAKE_BINARY_DIR}/coremltools)

set(copy_dirs _deps converters models proto)
foreach(cdir IN ITEMS ${copy_dirs})
  file(COPY ${CMAKE_SOURCE_DIR}/coremltools/${cdir}
    DESTINATION ${CMAKE_BINARY_DIR}/coremltools)
endforeach()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_additional_milstoragepython_command COMMAND strip -x ${PROJECT_SOURCE_DIR}/coremltools/libmilstoragepython.so)
  set(_additional_modelpackage_command COMMAND strip -x ${PROJECT_SOURCE_DIR}/coremltools/libmodelpackage.so)
endif()

add_custom_command(
  TARGET modelpackage
  POST_BUILD
  COMMAND cp $<TARGET_FILE:modelpackage> ${PROJECT_SOURCE_DIR}/coremltools/libmodelpackage.so
  ${_additional_modelpackage_command}
)
if (NOT APPLE)
  target_link_libraries(modelpackage uuid)
endif()

add_custom_command(
  TARGET milstoragepython
  POST_BUILD
  COMMAND cp $<TARGET_FILE:milstoragepython> ${PROJECT_SOURCE_DIR}/coremltools/libmilstoragepython.so
  ${_additional_milstoragepython_command}
)

find_library(CORE_VIDEO CoreVideo)
find_library(CORE_ML CoreML)
find_library(FOUNDATION Foundation)

if (APPLE AND CORE_VIDEO AND CORE_ML AND FOUNDATION)
  execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
      RESULT_VARIABLE NUMPY_INCLUDE_STATUS
      OUTPUT_VARIABLE NUMPY_INCLUDE
  )

  if("${NUMPY_INCLUDE}" STREQUAL "" OR NOT NUMPY_INCLUDE_STATUS EQUAL 0)
      message(FATAL_ERROR "Could not find numpy include path. Exit code: ${NUMPY_INCLUDE_STATUS}")
  endif()
  message("Found numpy include path at ${NUMPY_INCLUDE}")

  include_directories(
    ${NUMPY_INCLUDE}
  )

  add_library(coremlpython
    SHARED
    coremlpython/CoreMLPython.mm
    coremlpython/CoreMLPython.h
    coremlpython/CoreMLPythonArray.mm
    coremlpython/CoreMLPythonArray.h
    coremlpython/CoreMLPythonUtils.mm
    coremlpython/CoreMLPythonUtils.h
  )
  target_link_libraries(coremlpython
    mlmodel
    ${CORE_VIDEO}
    ${CORE_ML}
    ${FOUNDATION}
  )

  set(osx_export_file ${CMAKE_SOURCE_DIR}/coremlpython/exported_symbols_osx.ver)
  set_property(TARGET coremlpython APPEND PROPERTY LINK_DEPENDS "${osx_export_file}")
  set_property(TARGET coremlpython APPEND_STRING PROPERTY LINK_FLAGS " -Wl,-exported_symbols_list,${osx_export_file} ")

  # Allow Python to be found at runtime instead of compile/link time
  # This is apparently the default on Linux
  set_property(TARGET coremlpython APPEND_STRING PROPERTY LINK_FLAGS "-undefined dynamic_lookup")

  set_property(TARGET coremlpython APPEND_STRING PROPERTY LINK_FLAGS " -Wl,-dead_strip")

  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_additional_libcoremlpython_command
      COMMAND strip -x ${PROJECT_SOURCE_DIR}/coremltools/libcoremlpython.so
    )
  endif()

  add_custom_command(
    TARGET coremlpython
    POST_BUILD
    COMMAND cp $<TARGET_FILE:coremlpython> ${PROJECT_SOURCE_DIR}/coremltools/libcoremlpython.so
    ${_additional_libcoremlpython_command}
  )

else()
  message(STATUS "CoreML.framework and dependent frameworks not found. Skipping libcoremlpython build.")
endif()


# Build kmeans-1d
set(KMEANS_DIR "${PROJECT_SOURCE_DIR}/deps/kmeans1d")
execute_process(
  COMMAND python3 setup.py build_ext --inplace
  WORKING_DIRECTORY ${KMEANS_DIR}
  RESULT_VARIABLE KMEANS1D_BUILD_STATUS
)

if(NOT KMEANS1D_BUILD_STATUS EQUAL 0)
  message(FATAL_ERROR "Could not build kmeans1d dependency")
endif()

# Somehow Python's setuptools is building this shared object file so that it tries to load the C++
# standard library using an rpath that only exist on the build machine. Change that so it gets
# loaded from the standard location.
if(APPLE)
  file(GLOB SO_FILE "${PROJECT_SOURCE_DIR}/deps/kmeans1d/kmeans1d/_core.*.so")
  execute_process(
    COMMAND install_name_tool -change @rpath/libc++.1.dylib /usr/lib/libc++.1.dylib ${SO_FILE}
  )
endif()

# Copy kmeans-1d to Python deps folder
execute_process(
  COMMAND cp -r kmeans1d ../../coremltools/_deps
  WORKING_DIRECTORY ${KMEANS_DIR}
)


set(PYTHON_TAG "cp${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
if(APPLE)
  execute_process(COMMAND uname -m OUTPUT_VARIABLE HARDWARE_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(${HARDWARE_NAME} MATCHES "x86_64")
    set(MIN_MAC_OS "10_15")
  elseif(${HARDWARE_NAME} MATCHES "arm64")
    set(MIN_MAC_OS "11_0")
  else()
    message(FATAL_ERROR "Unsupported hardware type. On macOS, x86_64 and arm64 are supported.")
  endif()
  set(PLAT_NAME "macosx_${MIN_MAC_OS}_${HARDWARE_NAME}")
elseif("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
  set(PLAT_NAME "manylinux1_x86_64")
else()
  message(FATAL_ERROR "Unsupported build platform. Supported platforms are Linux and macOS.")
endif()


if(BUILD_TAG)
  set(BUILD_TAG_OPTION "--build-number=${BUILD_TAG}")
  message(STATUS "Using ${BUILD_TAG} as build tag for wheels.")
else()
  set(BUILD_TAG_OPTION "")
endif()

# Add a target for each platform, and then a 'dist' that will build all of them.
# Parallel invocations of setup.py is not safe, so we serialize them.
set(plat_targets "")
foreach(platform IN ITEMS ${PLAT_NAME})
  add_custom_target(dist_${platform}
    COMMENT "Building dist for platform ${platform}..."
    COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/setup.py
      bdist_wheel
      --plat-name=${platform}
      --python-tag=${PYTHON_TAG}
      ${BUILD_TAG_OPTION}
      --dist-dir=${PROJECT_BINARY_DIR}/dist
    DEPENDS "milstoragepython;modelpackage;coremlpython;${plat_targets}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
  set(plat_targets "${plat_targets};dist_${platform}")
endforeach()
# Add a 'dist' target that will build wheels for all possible platforms.
add_custom_target(dist DEPENDS ${plat_targets})

add_custom_target(pip_install_dev
  COMMAND pip install -e ${PROJECT_SOURCE_DIR}
  DEPENDS "coremlpython"
)

add_custom_target(pytest
  COMMAND pytest -r fs ${PROJECT_SOURCE_DIR}/coremltools/test/ --timeout=600
  DEPENDS pip_install_dev
  USES_TERMINAL
)

add_custom_target(pytest_no_slow
  COMMAND pytest -r fs -m '"no slow"' ${PROJECT_SOURCE_DIR}/coremltools/test/
  DEPENDS pip_install_dev
  USES_TERMINAL
)
