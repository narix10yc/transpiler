$llvm_root/bin/clang -O3 -c gen_file.ll -o gen_file.ll.o 

$llvm_root/bin/clang++ -O3 gen_file.ll.o test.cpp -o perftest \
-I../include -L../build/timeit -ltimeit