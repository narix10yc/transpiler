cmake_minimum_required(VERSION 3.20)

project(simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

message("########### Custom Setups ###########")
message(STATUS "PROJECT SOURCE DIR: ${PROJECT_SOURCE_DIR}")

# Apple has a system LLVM. We need to use a specific version
# specified by enviroment variable 'LLVM_ROOT' or 'llvm_root'
if(DEFINED ENV{llvm_root})
    set(LLVM_ROOT "$ENV{llvm_root}")
elseif(DEFINED ENV{LLVM_ROOT})
    set(LLVM_ROOT "$ENV{LLVM_ROOT}")
else()
    message(FATAL_ERROR "Missing enviroment variable 'LLVM_ROOT' or 'llvm_root'")
endif()

set(CMAKE_C_COMPILER "${LLVM_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${LLVM_ROOT}/bin/clang++")

set(LLVM_DIR "${LLVM_ROOT}/lib/cmake/llvm")

message("STATUS C Compiler: ${CMAKE_C_COMPILER}")
message("STATUS C++ Compiler: ${CMAKE_CXX_COMPILER}")

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include_directories(include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(openqasm)
add_subdirectory(quench)
add_subdirectory(timeit)
add_subdirectory(simulation)
add_subdirectory(tools)
add_subdirectory(tests)
