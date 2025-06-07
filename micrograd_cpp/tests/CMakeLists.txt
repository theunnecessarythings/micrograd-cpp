include(GoogleTest)

# Value tests
add_executable(value_tests test_value.cpp)
target_link_libraries(value_tests PRIVATE GTest::gtest_main micrograd_core)
gtest_discover_tests(value_tests)

# Tensor tests
add_executable(tensor_tests test_tensor.cpp)
target_link_libraries(tensor_tests PRIVATE GTest::gtest_main micrograd_core)
gtest_discover_tests(tensor_tests)

# NN Layer tests (Embedding, LayerNorm tests are added to test_nn.cpp)
add_executable(nn_layer_tests test_nn.cpp)
target_link_libraries(nn_layer_tests PRIVATE GTest::gtest_main micrograd_core micrograd_nn)
gtest_discover_tests(nn_layer_tests)

# Optimizer tests
add_executable(optimizer_tests test_optim.cpp)
target_link_libraries(optimizer_tests PRIVATE GTest::gtest_main micrograd_core micrograd_optim) # Link to micrograd_optim
gtest_discover_tests(optimizer_tests)
