add_library(simulation STATIC "gen_u3.cpp" "utils.cpp")

## TODO: This is an awkward way of linking to LLVM library
if(APPLE)
    target_link_libraries(simulation "${LLVM_ROOT}/lib/libLLVM-17.dylib")
else()
    target_link_libraries(simulation "/usr/lib/llvm-17/lib/libLLVM-17.so")
endif()

