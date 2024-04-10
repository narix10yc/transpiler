add_executable(ir_gen "irGen.cpp" "gen_u3.cpp")
add_library(irgen STATIC "gen_u3.cpp")

if(APPLE)
    target_link_libraries(ir_gen "${LLVM_ROOT}/lib/libLLVM-17.dylib")
    target_link_libraries(irgen "${LLVM_ROOT}/lib/libLLVM-17.dylib")
else()
    target_link_libraries(ir_gen "/usr/lib/llvm-17/lib/libLLVM-17.so")
    target_link_libraries(irgen "/usr/lib/llvm-17/lib/libLLVM-17.so")
endif()

