$llvm_root/bin/clang++ -Ofast ../performance/test.cpp ../performance/gen_file.ll \
-o irgen_perftest -I../include -Ltimeit -ltimeit

# clang++ -Ofast ../performance/test.cpp ../performance/gen_file.ll \
# -o irgen_perftest -I../include -Ltimeit -ltimeit \
# --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -mavx512f -std=c++17

$llvm_root/bin/clang++ -Ofast ../performance/test_tplt.cpp \
-o tplt_perftest -I../include -Ltimeit -ltimeit

# g++ -Ofast ../performance/test_tplt.cpp ../timeit/timeit.cpp \
# -o tplt_perftest_gcc -I../include -ftree-vectorize