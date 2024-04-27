$llvm_root/bin/clang++ -O3 ../performance/test.cpp ../performance/gen_file.ll \
-o perftest -I../include -Ltimeit -ltimeit