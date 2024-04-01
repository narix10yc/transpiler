add_library(timeit STATIC "timeit.cpp")
target_compile_options(timeit PUBLIC -O3 -march=native)
