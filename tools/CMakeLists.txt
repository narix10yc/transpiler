llvm_map_components_to_libnames(llvm_libs support core irreader orcjit native)
target_link_libraries(simulation utils ${llvm_libs})
target_link_libraries(saot utils ${llvm_libs})

add_executable(qasm_to_saot "qasm_to_saot.cpp")
target_link_libraries(qasm_to_saot saot qasm simulation)

add_executable(kernel "kernel.cpp")
target_link_libraries(kernel saot simulation utils)

add_executable(correctness_kernel "correctness_kernel.cpp")
target_link_libraries(correctness_kernel saot qasm simulation)

add_executable(jit_playground "jit_playground.cpp")
# llvm_map_components_to_libnames(llvm_libs orcjit native)
target_link_libraries(jit_playground saot simulation qasm)

add_executable(ptx_playground "ptx_playground.cpp")
llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD})
if (CUDA_FOUND)
    target_link_libraries(ptx_playground qasm saot simulation ${llvm_libs} cuda cudart)
else()
    target_link_libraries(ptx_playground qasm saot simulation ${llvm_libs})
endif()

add_executable(pg_polynomial "pg_polynomial.cpp")
target_link_libraries(pg_polynomial saot simulation)

add_executable(fpga_playground "fpga_playground.cpp")
target_link_libraries(fpga_playground qasm saot simulation)

add_executable(fpga_vary_buffer "fpga_vary_buffer.cpp")
target_link_libraries(fpga_vary_buffer qasm saot simulation)

add_executable(cost_model "cost_model.cpp")
target_link_libraries(cost_model qasm saot simulation)