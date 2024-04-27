add_library(simulation STATIC "gen_u3.cpp" "utils.cpp")
add_executable(ir_gen "irGen.cpp")

target_link_libraries(ir_gen simulation)

## TODO: This is an awkward way of linking to LLVM library
if(APPLE)
    target_link_libraries(ir_gen "${LLVM_ROOT}/lib/libLLVM-17.dylib")
    target_link_libraries(simulation "${LLVM_ROOT}/lib/libLLVM-17.dylib")
else()
    target_link_libraries(ir_gen "/usr/lib/llvm-17/lib/libLLVM-17.so")
    target_link_libraries(simulation "/usr/lib/llvm-17/lib/libLLVM-17.so")
endif()

