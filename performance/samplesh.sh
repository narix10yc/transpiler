$llvm_root/bin/clang++ -Ofast ../performance/test.cpp ../performance/gen_file.ll \
-o irgen_perftest -I../include -Ltimeit -ltimeit

$llvm_root/bin/clang++ -Ofast ../performance/test_tplt.cpp \
-o tplt_perftest -I../include -Ltimeit -ltimeit

# gcc-13 -Ofast ../performance/test_tplt.cpp ../timeit/timeit.cpp \
# -o tplt_perftest_gcc -I../include
