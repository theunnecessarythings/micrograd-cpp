#include "gtest/gtest.h"
#include "core/Value.h"
#include "optim/Optimizer.h"
#include "optim/SGD.h"
#include "optim/Adam.h"      // Added
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

TEST(OptimizerTest, AdamStep) {
    auto p1 = std::make_shared<Value>(1.0f);
    p1->grad = 0.1f;
    auto p2 = std::make_shared<Value>(2.0f);
    p2->grad = -0.2f;
    std::vector<std::shared_ptr<Value>> params = {p1, p2};

    Adam adam_optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f);

    // Step 1
    adam_optimizer.step();
    // t=1
    // For p1 (grad=0.1):
    // m1 = (1-0.9)*0.1 = 0.01
    // v1 = (1-0.999)*0.1^2 = 0.001 * 0.01 = 0.00001
    // m_hat1 = 0.01 / (1-0.9^1) = 0.01 / 0.1 = 0.1
    // v_hat1 = 0.00001 / (1-0.999^1) = 0.00001 / 0.001 = 0.01
    // update1 = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) = 0.001 * 0.1 / (0.1 + 1e-8) = 0.001 * (0.1/0.10000001) approx 0.001 * 0.9999999 = 0.0009999999
    // p1_new = 1.0 - 0.0009999999 = 0.9990000001
    EXPECT_NEAR(p1->data, 0.999000f, 1e-6f); // Adjusted expected value slightly for precision

    // For p2 (grad=-0.2):
    // m1 = (1-0.9)*(-0.2) = -0.02
    // v1 = (1-0.999)*(-0.2)^2 = 0.001 * 0.04 = 0.00004
    // m_hat1 = -0.02 / 0.1 = -0.2
    // v_hat1 = 0.00004 / 0.001 = 0.04
    // update2 = 0.001 * (-0.2) / (sqrt(0.04) + 1e-8) = 0.001 * (-0.2) / (0.2 + 1e-8) = 0.001 * (-0.2/0.20000001) approx 0.001 * (-0.99999995) = -0.00099999995
    // p2_new = 2.0 - (-0.00099999995) = 2.00099999995
    EXPECT_NEAR(p2->data, 2.001000f, 1e-6f); // Adjusted expected value slightly for precision

    // Simulate new gradients for step 2
    p1->grad = 0.2f;
    p2->grad = 0.1f; // new grad for p2
    adam_optimizer.step(); // t=2

    // For p1 (prev_m_data=0.01, prev_v_data=0.00001, new_grad=0.2, p1_current_data = 0.999000f):
    // m2 = 0.9*0.01 + (1-0.9)*0.2 = 0.009 + 0.02 = 0.029
    // v2 = 0.999*0.00001 + (1-0.999)*0.2^2 = 0.00000999 + 0.001*0.04 = 0.00000999 + 0.00004 = 0.00004999
    // beta1_t2 = 0.9^2 = 0.81. beta2_t2 = 0.999^2 = 0.998001
    // m_hat2 = 0.029 / (1-0.81) = 0.029 / 0.19 = 0.1526315789
    // v_hat2 = 0.00004999 / (1-0.998001) = 0.00004999 / 0.001999 = 0.02500750375
    // update1_s2 = 0.001 * 0.1526315789 / (sqrt(0.02500750375) + 1e-8)
    //            = 0.001 * 0.1526315789 / (0.158137609 + 1e-8)
    //            = 0.001 * 0.1526315789 / 0.158137619
    //            = 0.001 * 0.9651857...
    //            = 0.0009651857
    // p1_s2_new = 0.999000 - 0.0009651857 = 0.9980348143
    EXPECT_NEAR(p1->data, 0.9980348f, 1e-6f);
}

TEST(OptimizerTest, AdamZeroGrad) {
    auto p1 = std::make_shared<Value>(1.0f);
    p1->grad = 10.0f;
    std::vector<std::shared_ptr<Value>> params = {p1};
    Adam adam_optimizer(params); // Default LR etc.

    adam_optimizer.zero_grad();
    EXPECT_EQ(p1->grad, 0.0f);
}

TEST(OptimizerTest, AdamConstructorValidation) {
    auto p1 = std::make_shared<Value>(1.0f);
    std::vector<std::shared_ptr<Value>> params = {p1};
    EXPECT_THROW(Adam(params, -0.001f), std::invalid_argument); // Negative LR
    EXPECT_THROW(Adam(params, 0.001f, -0.1f), std::invalid_argument); // Negative beta1
    EXPECT_THROW(Adam(params, 0.001f, 1.1f), std::invalid_argument);  // beta1 >= 1
    EXPECT_THROW(Adam(params, 0.001f, 0.9f, -0.1f), std::invalid_argument); // Negative beta2
    EXPECT_THROW(Adam(params, 0.001f, 0.9f, 1.1f), std::invalid_argument);  // beta2 >= 1
    EXPECT_THROW(Adam(params, 0.001f, 0.9f, 0.999f, -1e-9f), std::invalid_argument); // Negative epsilon
}


// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
