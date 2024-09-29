cmake_minimum_required(VERSION 3.20)

message("########### Custom Setups ###########")
# Apple has a system LLVM. We need to use a specific version
# specified by enviroment variable 'LLVM_ROOT' or 'llvm_root'
if(DEFINED ENV{llvm_root})
    set(LLVM_ROOT "$ENV{llvm_root}")
elseif(DEFINED ENV{LLVM_ROOT})
    set(LLVM_ROOT "$ENV{LLVM_ROOT}")
else()
    message(FATAL_ERROR "Missing enviroment variable 'LLVM_ROOT' or 'llvm_root'")
endif()

message(STATUS "LLVM_ROOT: ${LLVM_ROOT}")

project(simulation LANGUAGES C CXX)

set(CMAKE_C_COMPILER "${LLVM_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${LLVM_ROOT}/bin/clang++")
### Skip cmake checks for working compiler
# set(CMAKE_C_COMPILER_WORKS 1)
# set(CMAKE_CXX_COMPILER_WORKS 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
add_link_options("-fuse-ld=lld")

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

if(DEFINED ENV{SELECT_GCC_TOOLCHAIN})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-install-dir=$ENV{SELECT_GCC_TOOLCHAIN}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-install-dir=$ENV{SELECT_GCC_TOOLCHAIN}")
    message(WARNING "Selected gcc toolchain $ENV{SELECT_GCC_TOOLCHAIN}")
endif()

set(LLVM_DIR "${LLVM_ROOT}/lib/cmake/llvm")
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS} include)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(openqasm)
add_subdirectory(saot)
add_subdirectory(timeit)
add_subdirectory(simulation)
add_subdirectory(tools)
add_subdirectory(tests)
