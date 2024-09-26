llvm_map_components_to_libnames(llvm_libs support core irreader)
target_link_libraries(simulation ${llvm_libs})
target_link_libraries(saot ${llvm_libs})

add_executable(qasm_to_saot "qasm_to_saot.cpp")
target_link_libraries(qasm_to_saot saot qasm simulation)

add_executable(kernel "kernel.cpp")
target_link_libraries(kernel saot simulation)

add_executable(correctness_kernel "correctness_kernel.cpp")
target_link_libraries(correctness_kernel saot qasm simulation)

add_executable(playground "playground.cpp")
target_link_libraries(playground saot qasm simulation)

add_executable(pg_polynomial "pg_polynomial.cpp")
target_link_libraries(pg_polynomial saot simulation)
