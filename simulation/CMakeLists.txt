add_executable(ir_gen "irGen.cpp" "gen_u3.cpp")
# add_executable(cpu_gen "cpu.cpp")

if(APPLE)
    target_link_libraries(ir_gen "${LLVM_ROOT}/lib/libLLVM-17.dylib")
    # target_link_libraries(cpu_gen "${LLVM_ROOT}/lib/libLLVM-17.dylib")
else()
    target_link_libraries(ir_gen "/usr/lib/llvm-17/lib/libLLVM-17.so")
    # target_link_libraries(cpu_gen "/usr/lib/llvm-17/lib/libLLVM-17.so")
endif()

