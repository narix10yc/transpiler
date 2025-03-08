file(GLOB ALL_SRC_FILES "*.cpp")
if (CAST_USE_CUDA)
  file(GLOB CUDA_SRC_FILES "*.cu")
  list(APPEND ALL_SRC_FILES ${CUDA_SRC_FILES})
else()
  list(REMOVE_ITEM ALL_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/StatevectorCUDA.cpp")
endif()

add_library(utils STATIC ${ALL_SRC_FILES})
