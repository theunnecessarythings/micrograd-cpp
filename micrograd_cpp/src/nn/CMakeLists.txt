set(NN_SOURCES
    Linear.cpp
    ReLU.cpp
    Sigmoid.cpp
    Embedding.cpp
    LayerNorm.cpp
    RMSNorm.cpp
    losses.cpp
    metrics.cpp
)

# Using STATIC library for nn module as it's likely to be small and specific to this project
# SHARED could be used if it were a more general-purpose library.
add_library(micrograd_nn STATIC ${NN_SOURCES})

# Link against the core library since Linear.h includes Tensor.h etc. from core
target_link_libraries(micrograd_nn PRIVATE micrograd_core)

# Make header files in this directory (like Layer.h, Linear.h) available
# to targets that link against micrograd_nn, and also for includes within micrograd_nn itself.
target_include_directories(micrograd_nn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
