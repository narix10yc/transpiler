llvm_map_components_to_libnames(llvm_libs support core irreader orcjit native)
target_link_libraries(simulation ${llvm_libs})
target_link_libraries(saot ${llvm_libs})

add_executable(qasm_to_saot "qasm_to_saot.cpp")
target_link_libraries(qasm_to_saot saot qasm simulation)

add_executable(kernel "kernel.cpp")
target_link_libraries(kernel saot simulation)

add_executable(correctness_kernel "correctness_kernel.cpp")
target_link_libraries(correctness_kernel saot qasm simulation)

add_executable(jit_playground "jit_playground.cpp")
# llvm_map_components_to_libnames(llvm_libs orcjit native)
target_link_libraries(jit_playground saot simulation qasm)

add_executable(ptx_playground "ptx_playground.cpp")
llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD})
target_link_libraries(ptx_playground saot simulation ${llvm_libs} cuda)

add_executable(pg_polynomial "pg_polynomial.cpp")
target_link_libraries(pg_polynomial saot simulation)

add_executable(fpga_playground "fpga_playground.cpp")
target_link_libraries(fpga_playground qasm saot simulation)