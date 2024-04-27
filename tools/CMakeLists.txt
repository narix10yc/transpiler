add_executable(qasm_to_qch "openqasm_to_qch.cpp")
target_link_libraries(qasm_to_qch qch qasm simulation)

add_executable(ir_gen "ir_gen.cpp")

target_link_libraries(ir_gen simulation)

## TODO: This is an awkward way of linking to LLVM library
if(APPLE)
    target_link_libraries(ir_gen "${LLVM_ROOT}/lib/libLLVM-17.dylib")
else()
    target_link_libraries(ir_gen "/usr/lib/llvm-17/lib/libLLVM-17.so")
endif()


add_executable(playground "playground.cpp")
target_link_libraries(playground qch qasm simulation)

