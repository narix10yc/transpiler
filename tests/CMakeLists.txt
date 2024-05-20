add_executable(unit_test "unit_test.cpp" "ir.ll.o")

target_link_libraries(unit_test simulation)