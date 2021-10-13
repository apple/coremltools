# User options
include("${CMAKE_CURRENT_LIST_DIR}/protobuf-options.cmake")

# Depend packages
if(NOT ZLIB_FOUND)
  find_package(ZLIB)
endif()

# Imported targets
include("${CMAKE_CURRENT_LIST_DIR}/protobuf-targets.cmake")

# CMake FindProtobuf module compatible file
if(protobuf_MODULE_COMPATIBLE)
  include("${CMAKE_CURRENT_LIST_DIR}/protobuf-module.cmake")
endif()
