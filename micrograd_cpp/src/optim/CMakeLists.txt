set(OPTIM_SOURCES
    SGD.cpp
)

add_library(micrograd_optim STATIC ${OPTIM_SOURCES})

# Optimizers need Value from core (e.g. SGD.h includes Optimizer.h which includes Value.h)
target_link_libraries(micrograd_optim PRIVATE micrograd_core)

# Make header files in this directory (like Optimizer.h, SGD.h) available
# to targets that link against micrograd_optim, and also for includes within micrograd_optim itself.
target_include_directories(micrograd_optim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
