add_library(testkit STATIC "TestKit.cpp")

add_executable(unit_test "test_main.cpp")
target_link_libraries(unit_test saot simulation testkit)

