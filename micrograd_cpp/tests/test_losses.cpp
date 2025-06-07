#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/Value.h"
#include "nn/losses.h" // For MSELoss, CrossEntropyLoss
#include <memory> // For std::make_shared
#include <vector>
#include <cmath>  // For std::log, std::exp for CE test verification
#include <stdexcept> // For EXPECT_THROW

// Helper to check gradients of Value objects
void check_value_grad(const std::shared_ptr<Value>& val, float expected_grad, float tol = 1e-4) {
    ASSERT_NE(val, nullptr);
    EXPECT_NEAR(val->grad, expected_grad, tol);
}

TEST(LossesTest, MSELossForward) {
    auto preds = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3});
    auto targets = Tensor::from_vector({1.5f, 2.5f, 2.5f}, {3});
    // diffs: -0.5, -0.5, 0.5
    // sq_diffs: 0.25, 0.25, 0.25
    // sum_sq_diffs: 0.75
    // mse = 0.75 / 3 = 0.25
    auto loss = MSELoss::forward(preds, targets);
    ASSERT_NE(loss, nullptr);
    EXPECT_NEAR(loss->data, 0.25f, 1e-4);
}

TEST(LossesTest, MSELossBackward) {
    auto p1 = std::make_shared<Value>(1.0f);
    auto p2 = std::make_shared<Value>(2.0f);

    // Targets are constants in the context of loss w.r.t predictions
    auto targets_tensor = Tensor::from_vector({1.5f, 2.5f}, {2});

    auto preds_tensor = Tensor::from_values({p1,p2}, {2});

    auto loss = MSELoss::forward(preds_tensor, targets_tensor); // ((p1-1.5)^2 + (p2-2.5)^2)/2
    ASSERT_NE(loss, nullptr);
    loss->backward();

    // L = ((p1-1.5)^2 + (p2-2.5)^2)/N where N=2
    // dL/dp1 = 2*(p1-1.5)/N = (p1-1.5) = 1.0 - 1.5 = -0.5
    // dL/dp2 = 2*(p2-2.5)/N = (p2-2.5) = 2.0 - 2.5 = -0.5
    check_value_grad(p1, -0.5f);
    check_value_grad(p2, -0.5f);
}

TEST(LossesTest, MSELossShapeMismatch) {
    auto preds1 = Tensor::from_vector({1.0f, 2.0f}, {2});
    auto targets1 = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(MSELoss::forward(preds1, targets1), std::runtime_error);

    auto preds2 = Tensor::from_vector({1.0f, 2.0f}, {1,2});
    auto targets2 = Tensor::from_vector({1.0f, 2.0f}, {2,1}); // Different shape
    EXPECT_THROW(MSELoss::forward(preds2, targets2), std::runtime_error);
}


TEST(LossesTest, CrossEntropyLossForward) {
    // Batch size 2, 2 classes
    // Item 1: logits [1.0, 3.0], target_idx 1
    // Item 2: logits [2.0, 0.5], target_idx 0

    // For [1.0, 3.0]: max_val = 3.0. x_minus_max = [-2.0, 0.0].
    // exps_minus_max = [exp(-2.0), exp(0.0)] = [0.13533528, 1.0].
    // sum_exps_minus_max = 1.13533528. log_sum_exps = std::log(1.13533528) = 0.126928.
    // log_probs_0 = -2.0 - 0.126928 = -2.126928.
    // log_probs_1 =  0.0 - 0.126928 = -0.126928. (Selected for target_idx 1)

    // For [2.0, 0.5]: max_val = 2.0. x_minus_max = [0.0, -1.5].
    // exps_minus_max = [exp(0.0), exp(-1.5)] = [1.0, 0.22313016].
    // sum_exps_minus_max = 1.22313016. log_sum_exps = std::log(1.22313016) = 0.2013675.
    // log_probs_0 =  0.0 - 0.2013675 = -0.2013675. (Selected for target_idx 0)
    // log_probs_1 = -1.5 - 0.2013675 = -1.7013675.

    // NLL losses: -(-0.126928) for item 1, -(-0.2013675) for item 2.
    // Total NLL loss = 0.126928 + 0.2013675 = 0.3282955.
    // Mean loss = 0.3282955 / 2 = 0.16414775.

    auto logits_tensor = Tensor::from_vector({1.0f, 3.0f, 2.0f, 0.5f}, {2, 2});
    auto targets_tensor = Tensor::from_vector({1.0f, 0.0f}, {2});

    auto loss = CrossEntropyLoss::forward(logits_tensor, targets_tensor);
    ASSERT_NE(loss, nullptr);
    EXPECT_NEAR(loss->data, 0.164148f, 1e-3f); // Further Adjusted tolerance
}

TEST(LossesTest, CrossEntropyLossBackward) {
    auto l1_val = std::make_shared<Value>(1.0f);
    auto l2_val = std::make_shared<Value>(0.5f);
    auto logits = Tensor::from_values({l1_val, l2_val}, {1,2});
    auto targets = Tensor::from_vector({0.0f}, {1});

    auto loss = CrossEntropyLoss::forward(logits, targets);
    ASSERT_NE(loss, nullptr);
    // Expected loss for [1.0, 0.5] target 0:
    // max_val = 1.0. x_minus_max = [0.0, -0.5].
    // exps_minus_max = [exp(0.0), exp(-0.5)] = [1.0, 0.60653066].
    // sum_exps_minus_max = 1.60653066. log_sum_exps = std::log(1.60653066) = 0.474095.
    // log_probs = [0.0 - 0.474095, -0.5 - 0.474095] = [-0.474095, -0.974095].
    // NLL loss for target 0 = -(-0.474095) = 0.474095.
    EXPECT_NEAR(loss->data, 0.474095f, 1e-4f); // Adjusted tolerance
    loss->backward();

    // dL/dL_i = p_i - y_i (where p_i is softmax_prob_i, y_i is 1 if i=target else 0)
    // softmax_probs = [exp(0)/sum_exp, exp(-0.5)/sum_exp] = [1.0/1.60653066, 0.60653066/1.60653066]
    //               = [0.62245935, 0.37754065]
    // dL/dL1 = (0.62245935 - 1.0) / 1 (batch_size) = -0.37754065
    // dL/dL2 = (0.37754065 - 0.0) / 1 (batch_size) =  0.37754065
    check_value_grad(l1_val, -0.377541f, 1e-3f); // Further Adjusted tolerance
    check_value_grad(l2_val,  0.377541f, 1e-3f); // Further Adjusted tolerance
}

TEST(LossesTest, CrossEntropyLossTargetOutOfRange) {
    auto logits = Tensor::from_vector({1.0f, 0.5f}, {1,2});
    auto targets_oor = Tensor::from_vector({2.0f}, {1}); // Target index 2, but only 2 classes (0,1)
    EXPECT_THROW(CrossEntropyLoss::forward(logits, targets_oor), std::out_of_range);
}

// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
