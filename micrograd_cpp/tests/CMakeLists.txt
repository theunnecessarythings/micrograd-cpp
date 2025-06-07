include(GoogleTest)

# Add test executable for Value tests
add_executable(value_tests test_value.cpp)
target_link_libraries(value_tests PRIVATE GTest::gtest_main micrograd_core)
gtest_discover_tests(value_tests)

# Add test executable for Tensor tests
add_executable(tensor_tests test_tensor.cpp)
target_link_libraries(tensor_tests PRIVATE GTest::gtest_main micrograd_core)
gtest_discover_tests(tensor_tests)
