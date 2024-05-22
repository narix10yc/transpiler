file(GLOB SRC_FILES "*.cpp")
add_library(simulation STATIC ${SRC_FILES})

llvm_map_components_to_libnames(llvm_libs support core irreader)
target_link_libraries(simulation ${llvm_libs})
