file(GLOB SRC_FILES "*.cpp")

add_library(TestKit STATIC ${SRC_FILES})
target_link_libraries(TestKit timeit)
target_compile_definitions(TestKit PRIVATE
        TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}") # Note the escaping attempts