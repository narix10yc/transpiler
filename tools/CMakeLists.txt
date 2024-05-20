add_executable(qasm_to_qch "openqasm_to_qch.cpp")
target_link_libraries(qasm_to_qch qch qasm simulation)

add_executable(gen_u3 "gen_u3.cpp")
add_executable(gen_u2q "gen_u2q.cpp")

target_link_libraries(gen_u3 simulation)
target_link_libraries(gen_u2q simulation)

## TODO: This is an awkward way to link LLVM library
if(APPLE)
    target_link_libraries(gen_u3 "${LLVM_ROOT}/lib/libLLVM-17.dylib")
    target_link_libraries(gen_u2q "${LLVM_ROOT}/lib/libLLVM-17.dylib")
else()
    target_link_libraries(gen_u3 "/usr/lib/llvm-17/lib/libLLVM-17.so")
    target_link_libraries(gen_u2q "${LLVM_ROOT}/lib/libLLVM-17.dylib")
endif()


add_executable(playground "playground.cpp")
target_link_libraries(playground qch qasm simulation)

