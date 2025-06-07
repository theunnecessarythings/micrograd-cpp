# Add source files for the core library
set(CORE_SOURCES
    Value.cpp
    Tensor.cpp
)

# Create a library for the core components
add_library(micrograd_core ${CORE_SOURCES})

# Make header files in this directory available
target_include_directories(micrograd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
