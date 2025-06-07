set(TRAIN_SOURCES
    Trainer.cpp
)

add_library(micrograd_train STATIC ${TRAIN_SOURCES})

# micrograd_train needs access to core, nn, and optim modules
target_link_libraries(micrograd_train PRIVATE micrograd_core micrograd_nn micrograd_optim)

target_include_directories(micrograd_train PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
