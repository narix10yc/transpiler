file(GLOB SRC_FILES "*.cpp")
add_library(simulation STATIC ${SRC_FILES})

# TODO: Separate gen_ptx.cpp as it requires NVPTX installation
