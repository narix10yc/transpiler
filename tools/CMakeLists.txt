add_executable(qasm_to_quench "openqasm_to_quench.cpp")
target_link_libraries(qasm_to_quench quench qasm simulation)

add_executable(playground "playground.cpp")
target_link_libraries(playground quench qasm simulation)

add_executable(kernel "kernel.cpp")
target_link_libraries(kernel quench simulation)