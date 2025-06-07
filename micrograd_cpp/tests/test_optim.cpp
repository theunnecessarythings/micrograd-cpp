#include "gtest/gtest.h"
#include "core/Value.h"
#include "optim/Optimizer.h" // For Optimizer base (not strictly needed but good include)
#include "optim/SGD.h"
#include <vector>
#include <memory> // For std::make_shared
#include <stdexcept> // For std::invalid_argument

TEST(OptimizerTest, SGDStep) {
    auto p1 = std::make_shared<Value>(5.0f);
    auto p2 = std::make_shared<Value>(-2.0f);
    std::vector<std::shared_ptr<Value>> params = {p1, p2};

    SGD sgd_optimizer(params, 0.1f); // learning_rate = 0.1

    // Simulate some gradients
    p1->grad = 10.0f;
    p2->grad = -5.0f;

    sgd_optimizer.step();

    // p1_new = 5.0 - 0.1 * 10.0 = 5.0 - 1.0 = 4.0
    // p2_new = -2.0 - 0.1 * (-5.0) = -2.0 + 0.5 = -1.5
    EXPECT_NEAR(p1->data, 4.0f, 1e-4);
    EXPECT_NEAR(p2->data, -1.5f, 1e-4);
}

TEST(OptimizerTest, SGDZeroGrad) {
    auto p1 = std::make_shared<Value>(1.0f);
    p1->grad = 10.0f;
    std::vector<std::shared_ptr<Value>> params = {p1};
    SGD sgd_optimizer(params, 0.1f);

    sgd_optimizer.zero_grad();
    EXPECT_EQ(p1->grad, 0.0f);
}

TEST(OptimizerTest, SGDConstructorInvalidLR) {
    auto p1 = std::make_shared<Value>(1.0f);
    std::vector<std::shared_ptr<Value>> params = {p1};
    EXPECT_THROW(SGD(params, -0.1f), std::invalid_argument);
}


// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
