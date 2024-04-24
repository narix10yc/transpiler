add_executable(qasm_to_qch "openqasm_to_qch.cpp")
target_link_libraries(qasm_to_qch qch qasm irgen)

add_executable(playground "playground.cpp")
target_link_libraries(playground qch qasm irgen)

# if(APPLE)
#     target_link_libraries(qasm_to_qch "${LLVM_ROOT}/lib/libLLVM-17.dylib")
# else()
#     target_link_libraries(qasm_to_qch "/usr/lib/llvm-17/lib/libLLVM-17.so")
# endif()