add_library(timeit STATIC "timeit.cpp")
target_compile_options(timeit PRIVATE -O3 -march=native)
