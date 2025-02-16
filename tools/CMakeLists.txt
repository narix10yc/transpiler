llvm_map_components_to_libnames(LLVM_LIBS support core irreader orcjit native)
target_link_libraries(simulation utils ${LLVM_LIBS})
target_link_libraries(cast utils ${LLVM_LIBS})

add_executable(unit_test "test_main.cpp")
target_link_libraries(unit_test cast simulation TestKit)

add_executable(qasm_to_cast "qasm_to_cast.cpp")
target_link_libraries(qasm_to_cast cast qasm simulation)

add_executable(qasm_to_qch "qasm_to_qch.cpp")
target_link_libraries(qasm_to_qch cast qasm utils)

add_executable(kernel "kernel.cpp")
target_link_libraries(kernel qasm cast simulation utils timeit)

add_executable(jit_playground "jit_playground.cpp")
# llvm_map_components_to_libnames(LLVM_LIBS orcjit native)
target_link_libraries(jit_playground cast simulation qasm)

add_executable(ptx_playground "ptx_playground.cpp")
llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_TARGETS_TO_BUILD})
if (CUDA_FOUND)
    target_link_libraries(ptx_playground qasm cast simulation ${LLVM_LIBS} cuda cudart)
else()
    target_link_libraries(ptx_playground qasm cast simulation ${LLVM_LIBS})
endif()

add_executable(scratch "scratch.cpp")
target_link_libraries(scratch utils cast simulation)

add_executable(fpga_playground "fpga_playground.cpp")
target_link_libraries(fpga_playground qasm cast simulation)

add_executable(fpga_vary_buffer "fpga_vary_buffer.cpp")
target_link_libraries(fpga_vary_buffer qasm cast simulation)

add_executable(cost_model "cost_model.cpp")
target_link_libraries(cost_model qasm cast simulation timeit utils)

add_executable(blank "blank.cpp")
