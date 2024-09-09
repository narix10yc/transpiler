add_executable(qasm_to_quench "openqasm_to_quench.cpp")
target_link_libraries(qasm_to_quench quench qasm simulation)

add_executable(kernel "kernel.cpp")
target_link_libraries(kernel quench simulation)

add_executable(correctness_kernel "correctness_kernel.cpp")
target_link_libraries(correctness_kernel quench qasm simulation)

add_executable(playground "playground.cpp")
target_link_libraries(playground quench qasm simulation)

add_executable(pg_polynomial "pg_polynomial.cpp")
target_link_libraries(pg_polynomial quench simulation)
