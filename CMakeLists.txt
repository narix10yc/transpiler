cmake_minimum_required(VERSION 3.20)

project(simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

message("########### Custom Setups ###########")
message(STATUS "PROJECT SOURCE DIR: ${PROJECT_SOURCE_DIR}")

if(APPLE)
    if(NOT DEFINED ENV{LLVM_ROOT} AND NOT DEFINED ENV{llvm_root})
    endif()
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
else()

endif()

find_package(LLVM REQUIRED CONFIG)

message("STATUS C Compiler: ${CMAKE_C_COMPILER}")
message("STATUS C++ Compiler: ${CMAKE_CXX_COMPILER}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(HandleLLVMOptions)

include_directories(include)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

option(SIMULATION_USE_OPENMP "Use OpenMP" OFF)
if(SIMULATION_USE_OPENMP)
    add_compile_definitions(SIMULATION_USE_OPENMP)
endif()

add_subdirectory(openqasm)
add_subdirectory(timeit)
add_subdirectory(qch)
add_subdirectory(simulation)
add_subdirectory(tools)
