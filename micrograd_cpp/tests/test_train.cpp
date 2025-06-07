#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/Value.h"
#include "nn/Layer.h"
#include "nn/Linear.h"
#include "nn/losses.h"
#include "nn/metrics.h"
#include "optim/Optimizer.h"
#include "optim/SGD.h"
#include "train/Trainer.h"
#include <memory>
#include <vector>
#include <iostream> // For printing loss in FitRunsAndLossDecreases if needed
#include <stdexcept> // For EXPECT_THROW
#include <cmath>     // For std::isnan

// A very simple model for testing: 1 linear layer
class SimpleModel : public Layer {
public:
    std::shared_ptr<Linear> linear_layer;
    SimpleModel(int in_features, int out_features) {
        linear_layer = std::make_shared<Linear>(in_features, out_features);
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        if (!linear_layer) throw std::runtime_error("SimpleModel: linear_layer is null.");
        return linear_layer->forward(input);
    }
    std::vector<std::shared_ptr<Value>> parameters() const override {
        if (!linear_layer) return {};
        return linear_layer->parameters();
    }
     std::string name() const override { return "SimpleModel"; }
};

TEST(TrainTest, TrainStepRuns) {
    auto model = std::make_shared<SimpleModel>(2, 1);
    ASSERT_FALSE(model->parameters().empty());
    auto optimizer = std::make_shared<SGD>(model->parameters(), 0.01f);

    auto input_batch = Tensor::from_vector({0.5f, -0.5f}, {1,2});
    auto target_batch = Tensor::from_vector({1.0f}, {1,1});

    auto loss_fn = [](const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& t) {
        return MSELoss::forward(p, t);
    };

    ASSERT_NO_THROW({
        auto loss = train_step(model, input_batch, target_batch, loss_fn, optimizer);
        ASSERT_NE(loss, nullptr);
    });
}

TEST(TrainTest, FitRunsAndLossDecreases) {
    auto model = std::make_shared<SimpleModel>(2, 1);
    ASSERT_FALSE(model->parameters().empty());

    auto optimizer = std::make_shared<SGD>(model->parameters(), 0.01f);
    auto loss_fn = MSELoss::forward;

    std::vector<std::shared_ptr<Tensor>> X_train;
    std::vector<std::shared_ptr<Tensor>> Y_train;
    X_train.push_back(Tensor::from_vector({1.0f, 1.0f}, {1,2}));
    Y_train.push_back(Tensor::from_vector({0.0f}, {1,1}));
    X_train.push_back(Tensor::from_vector({2.0f, 1.0f}, {1,2}));
    Y_train.push_back(Tensor::from_vector({2.0f}, {1,1}));
    X_train.push_back(Tensor::from_vector({1.0f, 2.0f}, {1,2}));
    Y_train.push_back(Tensor::from_vector({-3.0f}, {1,1}));

    float initial_total_loss = 0;
    for(size_t i=0; i<X_train.size(); ++i) {
        auto pred = model->forward(X_train[i]);
        initial_total_loss += loss_fn(pred, Y_train[i])->data;
    }
    float initial_avg_loss = X_train.empty() ? 0 : initial_total_loss / X_train.size();

    ASSERT_NO_THROW({
        fit(model, X_train, Y_train, loss_fn, optimizer, 50, 1); // 50 epochs
    });

    float final_total_loss = 0;
    for(size_t i=0; i<X_train.size(); ++i) {
        auto pred = model->forward(X_train[i]);
        final_total_loss += loss_fn(pred, Y_train[i])->data;
    }
    float final_avg_loss = X_train.empty() ? 0 : final_total_loss / X_train.size();

    std::cout << "FitRunsAndLossDecreases - Initial avg loss: " << initial_avg_loss
              << ", Final avg loss: " << final_avg_loss << std::endl;
    if (initial_avg_loss > 1e-5 && !X_train.empty()) {
        EXPECT_LT(final_avg_loss, initial_avg_loss);
    } else if (!X_train.empty()) {
        EXPECT_NEAR(final_avg_loss, initial_avg_loss, 1e-5);
    }
}

TEST(TrainTest, AccuracyScore) { // Renamed from AccuracyScoreTest to avoid main() conflict if ever combined
    auto preds = Tensor::from_vector({0.1f,0.9f, 0.8f,0.2f, 0.4f,0.6f, 0.7f,0.3f}, {4,2});
    auto targets = Tensor::from_vector({1.0f, 1.0f, 0.0f, 0.0f}, {4});
    float acc = accuracy_score(preds, targets);
    EXPECT_NEAR(acc, 0.5f, 1e-4);

    auto pred_single = Tensor::from_vector({2.0f, 5.0f}, {2});
    // For scalar target, Tensor::from_vector expects a non-empty shape if data is present.
    // The previous fix was to Tensor::numel() to return 1 for empty shape, and Tensor::from_vector
    // to allow empty shape if data size is 1.
    auto target_single_scalar = Tensor::from_vector({1.0f}, {});
    EXPECT_NEAR(accuracy_score(pred_single, target_single_scalar), 1.0f, 1e-4);

    auto target_single_1d = Tensor::from_vector({0.0f}, {1});
    EXPECT_NEAR(accuracy_score(pred_single, target_single_1d), 0.0f, 1e-4);
}

TEST(TrainTest, EvaluateAverageLossRuns) {
    auto model = std::make_shared<SimpleModel>(1,1);
    ASSERT_FALSE(model->parameters().empty());
    auto loss_fn = MSELoss::forward;
    std::vector<std::shared_ptr<Tensor>> X_test = { Tensor::from_vector({1.0f}, {1,1}) };
    std::vector<std::shared_ptr<Tensor>> Y_test = { Tensor::from_vector({1.0f}, {1,1}) };

    ASSERT_NO_THROW({
        float avg_loss = evaluate_average_loss(model, X_test, Y_test, loss_fn);
        EXPECT_FALSE(std::isnan(avg_loss));
    });
}

// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
