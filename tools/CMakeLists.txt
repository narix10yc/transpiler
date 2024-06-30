add_executable(qasm_to_qch "openqasm_to_qch.cpp")
target_link_libraries(qasm_to_qch qch qasm simulation)

add_executable(gen_u3 "gen_u3.cpp")
add_executable(gen_u2q "gen_u2q.cpp")

target_link_libraries(gen_u3 simulation)
target_link_libraries(gen_u2q simulation)

add_executable(playground "playground.cpp")
target_link_libraries(playground quench qch qasm simulation)

add_executable(kernel "general_kernel.cpp")
target_link_libraries(kernel quench simulation)