add_custom_target(ll_file
    COMMAND "${LLVM_ROOT}/bin/clang -O3 -c ${CMAKE_CURRENT_SOURCE_DIR}/gen_file.ll -o ${CMAKE_CURRENT_SOURCE_DIR}/gen_file.ll.o"
    )

add_executable(perf "gen_file.cpp" "gen_file.ll.o")

add_dependencies(perf ll_file timeit)
