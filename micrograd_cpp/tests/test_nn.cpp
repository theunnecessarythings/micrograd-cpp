#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/Value.h"
#include "nn/Layer.h"
#include "nn/Linear.h"
#include "nn/ReLU.h"
#include "nn/Sigmoid.h"
#include "nn/Embedding.h"
#include "nn/LayerNorm.h"
#include "nn/RMSNorm.h"
#include "nn/Conv2D.h"
#include "nn/Conv1D.h"
#include "nn/attention.h"
#include <memory> // For std::make_shared
#include <vector>
#include <cmath> // For std::exp, std::fabs

// Helper function to check gradients of Value objects
void check_value_grad(const std::shared_ptr<Value>& val, float expected_grad, float tol = 1e-4) {
    ASSERT_NE(val, nullptr);
    EXPECT_NEAR(val->grad, expected_grad, tol);
}

// Helper function to check gradients of all Value objects in a Tensor
void check_tensor_grads(const std::shared_ptr<Tensor>& tensor, const std::vector<float>& expected_grads, float tol = 1e-4) {
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(tensor->data.size(), expected_grads.size()) << "Tensor data size vs expected_grads size mismatch.";
    for (size_t i = 0; i < tensor->data.size(); ++i) {
        ASSERT_NE(tensor->data[i], nullptr) << "Null value pointer at index " << i;
        EXPECT_NEAR(tensor->data[i]->grad, expected_grads[i], tol) << "Gradient mismatch at index " << i;
    }
}

// Helper to check tensor data
void check_tensor_data(const std::shared_ptr<Tensor>& tensor, const std::vector<float>& expected_data, float tol = 1e-4) {
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(tensor->data.size(), expected_data.size()) << "Tensor data size vs expected_data size mismatch.";
    for (size_t i = 0; i < tensor->data.size(); ++i) {
        ASSERT_NE(tensor->data[i], nullptr) << "Null value pointer at index " << i;
        EXPECT_NEAR(tensor->data[i]->data, expected_data[i], tol) << "Data mismatch at index " << i;
    }
}

// --- Linear Layer Tests ---
TEST(NNLayerTest, LinearForwardShapeAndValues) {
    auto linear_layer = std::make_shared<Linear>(2, 3);
    linear_layer->weights = Tensor::from_vector({1,2,3,4,5,6}, {3,2});
    linear_layer->bias = Tensor::from_vector({0.1, 0.2, 0.3}, {3});
    auto input = Tensor::from_vector({10, 20}, {1, 2});
    auto output = linear_layer->forward(input);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->ndim(), 2);
    EXPECT_EQ(output->shape[0], 1);
    EXPECT_EQ(output->shape[1], 3);
    check_tensor_data(output, {50.1f, 110.2f, 170.3f});
}

TEST(NNLayerTest, LinearForwardVectorInput) {
    auto linear_layer = std::make_shared<Linear>(2, 3);
    linear_layer->weights = Tensor::from_vector({1,2,3,4,5,6}, {3,2});
    linear_layer->bias = Tensor::from_vector({0.1f, 0.2f, 0.3f}, {3});
    auto input_vector = Tensor::from_vector({10,20}, {2});
    auto output_vector = linear_layer->forward(input_vector);
    ASSERT_NE(output_vector, nullptr);
    EXPECT_EQ(output_vector->ndim(), 1);
    EXPECT_EQ(output_vector->shape[0], 3);
    check_tensor_data(output_vector, {50.1f, 110.2f, 170.3f});
}

TEST(NNLayerTest, LinearForwardNoBias) {
    auto linear_layer = std::make_shared<Linear>(2, 1, false);
    linear_layer->weights = Tensor::from_vector({2, 3}, {1,2});
    auto input = Tensor::from_vector({5, 10}, {1,2});
    auto output = linear_layer->forward(input);
    check_tensor_data(output, {40.0f});
}

TEST(NNLayerTest, LinearBackward) {
    auto linear_layer = std::make_shared<Linear>(1, 1);
    linear_layer->weights = Tensor::from_vector({2.0}, {1,1});
    linear_layer->bias = Tensor::from_vector({0.5}, {1});
    auto input_val = std::make_shared<Value>(3.0);
    auto input_tensor = Tensor::from_values({input_val}, {1});
    auto output_tensor = linear_layer->forward(input_tensor);
    ASSERT_EQ(output_tensor->data.size(), 1);
    output_tensor->data[0]->backward();
    EXPECT_NEAR(input_val->grad, 2.0f, 1e-4);
    EXPECT_NEAR(linear_layer->weights->data[0]->grad, 3.0f, 1e-4);
    EXPECT_NEAR(linear_layer->bias->data[0]->grad, 1.0f, 1e-4);
}

TEST(NNLayerTest, LinearParameters) {
    auto layer_with_bias = std::make_shared<Linear>(2,3,true);
    EXPECT_EQ(layer_with_bias->parameters().size(), 2*3 + 3);
    auto layer_no_bias = std::make_shared<Linear>(2,3,false);
    EXPECT_EQ(layer_no_bias->parameters().size(), 2*3);
}

// --- ReLU Layer Tests ---
// ... (ReLU tests from Subtask 15, Turn 1 file content) ...
TEST(NNLayerTest, ReLUForward) {
    auto relu_layer = std::make_shared<ReLU>();
    auto input = Tensor::from_vector({-1.0f, 0.0f, 2.0f, -3.0f}, {1, 4});
    auto output = relu_layer->forward(input);
    check_tensor_data(output, {0.0f, 0.0f, 2.0f, 0.0f});
}

TEST(NNLayerTest, ReLUBackward) {
    auto relu_layer = std::make_shared<ReLU>();
    auto v1 = std::make_shared<Value>(-2.0f);
    auto v2 = std::make_shared<Value>(3.0f);
    auto input_tensor = Tensor::from_values({v1,v2}, {1,2});
    auto output_tensor = relu_layer->forward(input_tensor);
    auto loss = output_tensor->data[0] + output_tensor->data[1];
    loss->backward();
    EXPECT_NEAR(v1->grad, 0.0f, 1e-4);
    EXPECT_NEAR(v2->grad, 1.0f, 1e-4);
}

TEST(NNLayerTest, ReLUParameters) {
    auto relu_layer = std::make_shared<ReLU>();
    EXPECT_EQ(relu_layer->parameters().size(), 0);
}

// --- Sigmoid Layer Tests ---
// ... (Sigmoid tests from Subtask 15, Turn 1 file content) ...
TEST(NNLayerTest, SigmoidForward) {
    auto sigmoid_layer = std::make_shared<Sigmoid>();
    auto v_zero = std::make_shared<Value>(0.0f);
    auto v_pos = std::make_shared<Value>(2.19722f);
    auto v_neg = std::make_shared<Value>(-2.19722f);
    auto input1 = Tensor::from_values({v_zero}, {1,1});
    auto output1 = sigmoid_layer->forward(input1);
    EXPECT_NEAR(output1->data[0]->data, 0.5f, 1e-4);
    auto input2 = Tensor::from_values({v_pos, v_neg}, {1,2});
    auto output2 = sigmoid_layer->forward(input2);
    EXPECT_NEAR(output2->data[0]->data, 0.9f, 1e-4);
    EXPECT_NEAR(output2->data[1]->data, 0.1f, 1e-4);
}

TEST(NNLayerTest, SigmoidBackward) {
    auto sigmoid_layer = std::make_shared<Sigmoid>();
    auto v = std::make_shared<Value>(0.5f);
    auto input_tensor = Tensor::from_values({v}, {1,1});
    auto output_tensor = sigmoid_layer->forward(input_tensor);
    output_tensor->data[0]->backward();
    float s_val = 1.0f / (1.0f + std::exp(-0.5f));
    float expected_grad = s_val * (1.0f - s_val);
    EXPECT_NEAR(v->grad, expected_grad, 1e-4);
}

TEST(NNLayerTest, SigmoidParameters) {
    auto sigmoid_layer = std::make_shared<Sigmoid>();
    EXPECT_EQ(sigmoid_layer->parameters().size(), 0);
}

// --- Embedding Layer Tests ---
// ... (Embedding tests from Subtask 15, Turn 1 file content) ...
TEST(NNLayerTest, EmbeddingForwardShapeAndValues) {
    auto embedding_layer = std::make_shared<Embedding>(10, 3);
    embedding_layer->weights->set({2,0}, std::make_shared<Value>(0.1f));
    embedding_layer->weights->set({2,1}, std::make_shared<Value>(0.2f));
    embedding_layer->weights->set({2,2}, std::make_shared<Value>(0.3f));
    embedding_layer->weights->set({5,0}, std::make_shared<Value>(0.4f));
    embedding_layer->weights->set({5,1}, std::make_shared<Value>(0.5f));
    embedding_layer->weights->set({5,2}, std::make_shared<Value>(0.6f));
    auto input_indices = Tensor::from_vector({2.0f, 5.0f}, {1, 2});
    auto output = embedding_layer->forward(input_indices);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->ndim(), 3);
    EXPECT_EQ(output->shape[0], 1);
    EXPECT_EQ(output->shape[1], 2);
    EXPECT_EQ(output->shape[2], 3);
    std::vector<float> expected_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    check_tensor_data(output, expected_data);
}

TEST(NNLayerTest, EmbeddingForwardSingleIndex) {
    auto embedding_layer = std::make_shared<Embedding>(5, 2);
    embedding_layer->weights->set({3,0}, std::make_shared<Value>(1.0f));
    embedding_layer->weights->set({3,1}, std::make_shared<Value>(2.0f));
    auto output = embedding_layer->forward(3);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->ndim(), 1);
    EXPECT_EQ(output->shape[0], 2);
    check_tensor_data(output, {1.0f, 2.0f});
}

TEST(NNLayerTest, EmbeddingBackward) {
    auto embedding_layer = std::make_shared<Embedding>(5, 1);
    auto w2 = embedding_layer->weights->get({2,0});
    ASSERT_NE(w2, nullptr); w2->data = 0.5f;
    auto w4 = embedding_layer->weights->get({4,0});
    ASSERT_NE(w4, nullptr); w4->data = 0.8f;
    auto idx_tensor = Tensor::from_vector({2.0f, 4.0f}, {1,2});
    auto output_tensor = embedding_layer->forward(idx_tensor);
    auto loss = output_tensor->get({0,0,0}) + output_tensor->get({0,1,0});
    loss->backward();
    for(int i=0; i<5; ++i) {
        auto weight_val = embedding_layer->weights->get({i,0});
        ASSERT_NE(weight_val, nullptr);
        if (i==2 || i==4) {
            EXPECT_NEAR(weight_val->grad, 1.0f, 1e-4) << "Grad mismatch for W_idx" << i;
        } else {
            EXPECT_NEAR(weight_val->grad, 0.0f, 1e-4) << "Grad mismatch for W_idx" << i;
        }
    }
}

TEST(NNLayerTest, EmbeddingParameters) {
    auto embedding_layer = std::make_shared<Embedding>(100,10);
    EXPECT_EQ(embedding_layer->parameters().size(), 100*10);
}

// --- LayerNorm Layer Tests ---
// ... (LayerNorm tests from Subtask 15, Turn 1 file content) ...
TEST(NNLayerTest, LayerNormForwardValues) {
    int feature_dim = 3;
    auto ln_layer = std::make_shared<LayerNorm>(feature_dim, 1e-5f);
    ln_layer->gamma = Tensor::from_vector({1.0f, 1.5f, 2.0f}, {feature_dim});
    ln_layer->beta = Tensor::from_vector({0.1f, 0.2f, 0.3f}, {feature_dim});
    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, feature_dim});
    auto output = ln_layer->forward(input);
    std::vector<float> expected_data = {-1.12473f, 0.2f, 2.74946f};
    check_tensor_data(output, expected_data, 1e-4f);
}

TEST(NNLayerTest, LayerNormBackward) {
    int feature_dim = 2;
    auto ln_layer = std::make_shared<LayerNorm>(feature_dim, 1e-5f);
    ln_layer->gamma = Tensor::from_vector({1.0f, 1.0f}, {feature_dim});
    ln_layer->beta  = Tensor::from_vector({0.0f, 0.0f}, {feature_dim});
    auto x1 = std::make_shared<Value>(1.0f);
    auto x2 = std::make_shared<Value>(3.0f);
    auto input_tensor = Tensor::from_values({x1, x2}, {1, feature_dim});
    auto output_tensor = ln_layer->forward(input_tensor);
    auto loss = output_tensor->get({0,0}) + output_tensor->get({0,1});
    loss->backward();
    EXPECT_NEAR(x1->grad, 0.0f, 1e-4f);
    EXPECT_NEAR(x2->grad, 0.0f, 1e-4f);
    EXPECT_NEAR(ln_layer->gamma->get({0})->grad, -0.999995f, 1e-4f);
    EXPECT_NEAR(ln_layer->gamma->get({1})->grad, 0.999995f, 1e-4f);
    EXPECT_NEAR(ln_layer->beta->get({0})->grad, 1.0f, 1e-4f);
    EXPECT_NEAR(ln_layer->beta->get({1})->grad, 1.0f, 1e-4f);
}

TEST(NNLayerTest, LayerNormParameters) {
    auto ln_layer = std::make_shared<LayerNorm>(10);
    EXPECT_EQ(ln_layer->parameters().size(), 10 + 10);
}

// --- RMSNorm Layer Tests ---
// ... (RMSNorm tests from Subtask 16, Turn 1 file content) ...
TEST(NNLayerTest, RMSNormForwardValues) {
    int feature_dim = 3;
    auto rms_layer = std::make_shared<RMSNorm>(feature_dim, 1e-5f);
    rms_layer->gamma = Tensor::from_vector({1.0f, 1.5f, 0.5f}, {feature_dim});
    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, feature_dim});
    auto output = rms_layer->forward(input);
    std::vector<float> expected_data = {0.462910f, 1.388730f, 0.694365f};
    check_tensor_data(output, expected_data, 1e-5f);
}

TEST(NNLayerTest, RMSNormBackward) {
    int feature_dim = 2;
    auto rms_layer = std::make_shared<RMSNorm>(feature_dim, 1e-5f);
    rms_layer->gamma = Tensor::from_vector({1.0f, 1.0f}, {feature_dim});
    auto x1_val = std::make_shared<Value>(1.0f);
    auto x2_val = std::make_shared<Value>(3.0f);
    auto input_tensor = Tensor::from_values({x1_val, x2_val}, {1, feature_dim});
    auto output_tensor = rms_layer->forward(input_tensor);
    auto loss = output_tensor->get({0,0}) + output_tensor->get({0,1});
    loss->backward();
    EXPECT_NE(rms_layer->gamma->get({0})->grad, 0.0f);
    EXPECT_NE(rms_layer->gamma->get({1})->grad, 0.0f);
    EXPECT_NE(x1_val->grad, 0.0f);
    EXPECT_NE(x2_val->grad, 0.0f);
}

TEST(NNLayerTest, RMSNormParameters) {
    auto rms_layer = std::make_shared<RMSNorm>(20);
    EXPECT_EQ(rms_layer->parameters().size(), 20);
}

// --- Conv2D Layer Tests ---
// ... (Conv2D tests from Subtask 17, Turn 1 file content) ...
TEST(NNLayerTest, Conv2DForwardSimple) {
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 1, 0, false);
    conv_layer->weights = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2});
    auto input = Tensor::from_vector({1,1,1, 1,1,1, 1,1,1}, {1, 1, 3, 3});
    auto output = conv_layer->forward(input);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1);
    EXPECT_EQ(output->shape[1], 1);
    EXPECT_EQ(output->shape[2], 2);
    EXPECT_EQ(output->shape[3], 2);
    check_tensor_data(output, {10.0f, 10.0f, 10.0f, 10.0f});
}

TEST(NNLayerTest, Conv2DForwardWithBias) {
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 1, 0, true);
    conv_layer->weights = Tensor::from_vector({1,0,0,1}, {1,1,2,2});
    conv_layer->biases = Tensor::from_vector({0.5f}, {1});
    auto input = Tensor::from_vector({1,2,3, 1,1,1, 2,2,2}, {1,1,3,3});
    auto output = conv_layer->forward(input);
    check_tensor_data(output, {2.5f, 3.5f, 3.5f, 3.5f});
}

TEST(NNLayerTest, Conv2DForwardWithStride) {
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 2, 0, false);
    conv_layer->weights = Tensor::from_vector({1,2,3,4}, {1,1,2,2});
    auto input = Tensor::from_vector({1,1,1,1,1,1,1,1,1}, {1,1,3,3});
    auto output = conv_layer->forward(input);
    ASSERT_EQ(output->shape[2], 1);
    ASSERT_EQ(output->shape[3], 1);
    check_tensor_data(output, {10.0f});
}

TEST(NNLayerTest, Conv2DForwardWithPadding) {
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 1, 1, false);
    conv_layer->weights = Tensor::from_vector({1,1,1,1}, {1,1,2,2});
    auto input = Tensor::from_vector({1,2,3,4}, {1,1,2,2});
    auto output = conv_layer->forward(input);
    ASSERT_EQ(output->shape[2], 3);
    ASSERT_EQ(output->shape[3], 3);
    std::vector<float> expected = { 1.0f,  3.0f,  2.0f, 4.0f, 10.0f,  6.0f, 3.0f,  7.0f,  4.0f };
    check_tensor_data(output, expected);
}

TEST(NNLayerTest, Conv2DBackwardSmall) {
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 1, 1, 0, true);
    conv_layer->weights = Tensor::from_vector({2.0f}, {1,1,1,1});
    conv_layer->biases = Tensor::from_vector({0.5f}, {1});
    auto x1 = std::make_shared<Value>(1.0f); auto x2 = std::make_shared<Value>(2.0f);
    auto x3 = std::make_shared<Value>(3.0f); auto x4 = std::make_shared<Value>(4.0f);
    auto input_tensor = Tensor::from_values({x1,x2,x3,x4}, {1,1,2,2});
    auto output_tensor = conv_layer->forward(input_tensor);
    auto loss = std::make_shared<Value>(0.0f);
    for(const auto& val_ptr : output_tensor->data) {
        loss = loss + val_ptr;
    }
    loss->backward();
    check_value_grad(conv_layer->weights->data[0], 10.0f);
    check_value_grad(conv_layer->biases->data[0], 4.0f);
    check_value_grad(x1, 2.0f);
    check_value_grad(x2, 2.0f);
    check_value_grad(x3, 2.0f);
    check_value_grad(x4, 2.0f);
}

TEST(NNLayerTest, Conv2DParameters) {
    auto conv_with_bias = std::make_shared<Conv2D>(3,16,3,1,1,true);
    EXPECT_EQ(conv_with_bias->parameters().size(), 16*3*3*3 + 16);
    auto conv_no_bias = std::make_shared<Conv2D>(3,16,3,1,1,false);
    EXPECT_EQ(conv_no_bias->parameters().size(), 16*3*3*3);
}

// --- Conv1D Layer Tests ---
// ... (Conv1D tests from Subtask 18, Turn 1 file content) ...
TEST(NNLayerTest, Conv1DForwardSimple) {
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 3, 1, 0, false);
    conv_layer->weights = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, 1, 3});
    auto input = Tensor::from_vector({1.0f,1.0f,1.0f,1.0f,1.0f}, {1, 1, 5});
    auto output = conv_layer->forward(input);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1);
    EXPECT_EQ(output->shape[1], 1);
    EXPECT_EQ(output->shape[2], 3);
    check_tensor_data(output, {6.0f, 6.0f, 6.0f});
}

TEST(NNLayerTest, Conv1DForwardWithBias) {
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 2, 1, 0, true);
    conv_layer->weights = Tensor::from_vector({1.0f, 0.5f}, {1,1,2});
    conv_layer->biases = Tensor::from_vector({0.1f}, {1});
    auto input = Tensor::from_vector({1.0f,2.0f,3.0f}, {1,1,3});
    auto output = conv_layer->forward(input);
    check_tensor_data(output, {2.1f, 3.6f});
}

TEST(NNLayerTest, Conv1DForwardWithStride) {
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 3, 2, 0, false);
    conv_layer->weights = Tensor::from_vector({1,0,1}, {1,1,3});
    auto input = Tensor::from_vector({1,2,3,4,5}, {1,1,5});
    auto output = conv_layer->forward(input);
    ASSERT_EQ(output->shape[2], 2);
    check_tensor_data(output, {4.0f, 8.0f});
}

TEST(NNLayerTest, Conv1DForwardWithPadding) {
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 3, 1, 1, false);
    conv_layer->weights = Tensor::from_vector({1,1,1}, {1,1,3});
    auto input = Tensor::from_vector({1,2,3}, {1,1,3});
    auto output = conv_layer->forward(input);
    ASSERT_EQ(output->shape[2], 3);
    check_tensor_data(output, {3.0f, 6.0f, 5.0f});
}

TEST(NNLayerTest, Conv1DBackwardSmall) {
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 2, 1, 0, true);
    conv_layer->weights = Tensor::from_vector({2.0f, 0.5f}, {1,1,2});
    conv_layer->biases = Tensor::from_vector({0.1f}, {1});
    auto x1_v = std::make_shared<Value>(1.0f);
    auto x2_v = std::make_shared<Value>(2.0f);
    auto x3_v = std::make_shared<Value>(3.0f);
    auto input_tensor = Tensor::from_values({x1_v,x2_v,x3_v}, {1,1,3});
    auto output_tensor = conv_layer->forward(input_tensor);
    auto loss = output_tensor->get({0,0,0}) + output_tensor->get({0,0,1});
    loss->backward();
    check_value_grad(conv_layer->weights->get({0,0,0}), 3.0f);
    check_value_grad(conv_layer->weights->get({0,0,1}), 5.0f);
    check_value_grad(conv_layer->biases->get({0}), 2.0f);
    check_value_grad(x1_v, 2.0f);
    check_value_grad(x2_v, 2.5f);
    check_value_grad(x3_v, 0.5f);
}

TEST(NNLayerTest, Conv1DParameters) {
    auto conv_with_bias = std::make_shared<Conv1D>(3,16,3,1,1,true);
    EXPECT_EQ(conv_with_bias->parameters().size(), 16*3*3 + 16);
    auto conv_no_bias = std::make_shared<Conv1D>(3,16,3,1,1,false);
    EXPECT_EQ(conv_no_bias->parameters().size(), 16*3*3);
}

// --- Scaled Dot-Product Attention Tests ---
// ... (ScaledDotProductAttention tests from Subtask 19, Turn 1 file content) ...
TEST(NNLayerTest, ScaledDotProductAttentionForwardSimple) {
    auto q_tensor = Tensor::from_vector({1.0f, 2.0f}, {1, 1, 2});
    auto k_tensor = Tensor::from_vector({3.0f, 4.0f}, {1, 1, 2});
    auto v_tensor = Tensor::from_vector({5.0f, 6.0f}, {1, 1, 2});
    auto output = micrograd_nn::scaled_dot_product_attention(q_tensor, k_tensor, v_tensor);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1);
    EXPECT_EQ(output->shape[1], 1);
    EXPECT_EQ(output->shape[2], 2);
    check_tensor_data(output, {5.0f, 6.0f}, 1e-4f);
}

TEST(NNLayerTest, ScaledDotProductAttentionForwardMultipleKeys) {
    auto q = Tensor::from_vector({1.0f, 0.0f}, {1,1,2});
    auto k = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {1,2,2});
    auto v = Tensor::from_vector({0.1f,0.2f, 0.3f,0.4f}, {1,2,2});
    auto output = micrograd_nn::scaled_dot_product_attention(q, k, v);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1);
    EXPECT_EQ(output->shape[1], 1);
    EXPECT_EQ(output->shape[2], 2);
    check_tensor_data(output, {0.166046f, 0.266046f}, 1e-5f);
}

TEST(NNLayerTest, ScaledDotProductAttentionWithMask) {
    auto q = Tensor::from_vector({1.0f, 0.0f}, {1,1,2});
    auto k = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f, 1.0f, 1.0f}, {1,3,2});
    auto v = Tensor::from_vector({0.1f,0.2f,  0.3f,0.4f,  0.5f,0.6f}, {1,3,2});
    auto mask = Tensor::from_vector({0.0f, 1.0f, 0.0f}, {1,1,3});
    auto output = micrograd_nn::scaled_dot_product_attention(q, k, v, mask);
    check_tensor_data(output, {0.3f, 0.4f}, 1e-5f);
}

TEST(NNLayerTest, ScaledDotProductAttentionBackward) {
    auto q_val = std::make_shared<Value>(1.0f);
    auto k_val = std::make_shared<Value>(1.0f);
    auto v_val = std::make_shared<Value>(1.0f);
    auto q_tensor = Tensor::from_values({q_val}, {1,1,1});
    auto k_tensor = Tensor::from_values({k_val}, {1,1,1});
    auto v_tensor = Tensor::from_values({v_val}, {1,1,1});
    auto output_tensor = micrograd_nn::scaled_dot_product_attention(q_tensor, k_tensor, v_tensor);
    auto loss = output_tensor->get({0,0,0});
    ASSERT_NE(loss, nullptr);
    loss->backward();
    EXPECT_NEAR(q_val->grad, 0.0f, 1e-4f);
    EXPECT_NEAR(k_val->grad, 0.0f, 1e-4f);
    EXPECT_NE(v_val->grad, 0.0f);
    EXPECT_NEAR(v_val->grad, 1.0f, 1e-4f);
}


// --- MultiHeadAttention Layer Tests ---
#include "nn/MultiHeadAttention.h" // Add include for MultiHeadAttention

TEST(NNLayerTest, MultiHeadAttentionForwardShape) {
    int embed_dim = 6;
    int num_heads = 2;
    int batch_size = 1;
    int seq_len = 3;

    auto mha_layer = std::make_shared<MultiHeadAttention>(embed_dim, num_heads);

    auto query = Tensor::randn({batch_size, seq_len, embed_dim});
    auto key   = Tensor::randn({batch_size, seq_len, embed_dim}); // Self-attention for shape test
    auto value = Tensor::randn({batch_size, seq_len, embed_dim});

    auto output = mha_layer->forward(query, key, value);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->ndim(), 3);
    EXPECT_EQ(output->shape[0], batch_size);
    EXPECT_EQ(output->shape[1], seq_len);    // seq_len_q
    EXPECT_EQ(output->shape[2], embed_dim);
}

TEST(NNLayerTest, MultiHeadAttentionForwardValuesSimple) {
    // Test with very simple weights and inputs to trace values
    // embed_dim = 2, num_heads = 1 (effectively single head attention but using MHA machinery)
    // head_dim = 2
    int embed_dim = 2;
    int num_heads = 1;
    int batch_size = 1;
    int seq_len_q = 1;
    int seq_len_kv = 1;

    auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, true); // bias=true

    // Make projection weights identity and biases zero for simplicity
    // Wq, Wk, Wv, Wo will be identity matrices of size 2x2
    // Linear layer weights are [out_features, in_features]
    mha->wq->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2}); // Identity
    mha->wq->bias    = Tensor::from_vector({0.0f,0.0f}, {2});          // Zero bias
    mha->wk->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2});
    mha->wk->bias    = Tensor::from_vector({0.0f,0.0f}, {2});
    mha->wv->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2});
    mha->wv->bias    = Tensor::from_vector({0.0f,0.0f}, {2});
    mha->wo->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2});
    mha->wo->bias    = Tensor::from_vector({0.0f,0.0f}, {2});

    // Q = [[0.5, 1.0]], K = [[0.8, 0.2]], V = [[1.0, 2.0]]
    // All shapes [1,1,2] (batch, seq_len, embed_dim)
    auto q_tensor = Tensor::from_vector({0.5f, 1.0f}, {batch_size, seq_len_q, embed_dim});
    auto k_tensor = Tensor::from_vector({0.8f, 0.2f}, {batch_size, seq_len_kv, embed_dim});
    auto v_tensor = Tensor::from_vector({1.0f, 2.0f}, {batch_size, seq_len_kv, embed_dim});

    // Projections will be same as Q,K,V due to identity weights and zero bias
    // q_proj = [0.5, 1.0], k_proj = [0.8, 0.2], v_proj = [1.0, 2.0]
    // For num_heads=1, head_dim=2:
    // q_final = q_proj, k_final = k_proj, v_final = v_proj (after reshape/permute, still same values)
    // SDPA input shapes: Q_in,K_in,V_in are all [1*1, 1, 2] = [1,1,2]
    // d_k = head_dim = 2. sqrt(d_k) = 1.41421
    // Q_in @ K_in.T = [0.5, 1.0] @ [[0.8],[0.2]] = 0.5*0.8 + 1.0*0.2 = 0.4 + 0.2 = 0.6
    // Scaled score = 0.6 / 1.41421 = 0.42426
    // Softmax(scalar) = 1.0. Attention weight = [[1.0]] (shape [1,1,1] for SDPA)
    // SDPA output = [[1.0]] @ [[1.0, 2.0]] = [[1.0, 2.0]] (shape [1,1,2])
    // This output is reshaped/permuted back, then passed to Wo.
    // Since Wo is identity, final output should be [1.0, 2.0]

    auto output = mha->forward(q_tensor, k_tensor, v_tensor);
    ASSERT_NE(output, nullptr);
    check_tensor_data(output, {1.0f, 2.0f}, 1e-4f);
}

TEST(NNLayerTest, MultiHeadAttentionSelfAttention) {
    int embed_dim = 4;
    int num_heads = 2;
    auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads);
    auto x = Tensor::randn({1, 3, embed_dim}); // B, S, E

    ASSERT_NO_THROW({
        auto output = mha->forward(x); // Self-attention
        ASSERT_NE(output, nullptr);
        EXPECT_EQ(output->ndim(), 3);
        EXPECT_EQ(output->shape[0], 1);
        EXPECT_EQ(output->shape[1], 3);
        EXPECT_EQ(output->shape[2], embed_dim);
    });
}

TEST(NNLayerTest, MultiHeadAttentionWithMask) {
    int embed_dim = 2;
    int num_heads = 1; // Keep it simple
    int batch_size = 1;
    int seq_len_q = 1;
    int seq_len_kv = 2; // Two keys/values

    auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, false); // no bias

    // Make Wq, Wk, Wv, Wo identity
    mha->wq->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2});
    mha->wk->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2});
    mha->wv->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2});
    mha->wo->weights = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {2,2});

    auto q_tensor = Tensor::from_vector({1.0f, 0.0f}, {batch_size, seq_len_q, embed_dim}); // Q=[[1,0]]
    // K = [[[1,0], [0,1]]] (K0, K1)
    auto k_tensor = Tensor::from_vector({1.0f,0.0f,  0.0f,1.0f}, {batch_size, seq_len_kv, embed_dim});
    // V = [[[0.1,0.2], [0.3,0.4]]] (V0, V1)
    auto v_tensor = Tensor::from_vector({0.1f,0.2f,  0.3f,0.4f}, {batch_size, seq_len_kv, embed_dim});

    // Mask out the second key/value. Mask shape [B, Sq, Sk] = [1,1,2]
    // Mask value of 1.0 means mask it.
    auto mask = Tensor::from_vector({0.0f, 1.0f}, {batch_size, seq_len_q, seq_len_kv});

    auto output = mha->forward(q_tensor, k_tensor, v_tensor, mask);
    ASSERT_NE(output, nullptr);
    check_tensor_data(output, {0.1f, 0.2f}, 1e-5f);
}


TEST(NNLayerTest, MultiHeadAttentionParameters) {
    auto mha_layer = std::make_shared<MultiHeadAttention>(6, 2, true); // embed_dim=6, num_heads=2, bias=true
    EXPECT_EQ(mha_layer->parameters().size(), 4 * (6*6 + 6));

    auto mha_no_bias = std::make_shared<MultiHeadAttention>(6, 2, false); // bias=false
    EXPECT_EQ(mha_no_bias->parameters().size(), 4 * (6*6));
}

TEST(NNLayerTest, MultiHeadAttentionBackwardRuns) {
    int embed_dim = 2;
    int num_heads = 1;
    auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, true);

    auto q_val = std::make_shared<Value>(0.5f);
    auto k_val = std::make_shared<Value>(0.8f);
    auto v_val = std::make_shared<Value>(1.0f);
    auto q_other_dim_val = std::make_shared<Value>(1.0f);
    auto k_other_dim_val = std::make_shared<Value>(0.2f);
    auto v_other_dim_val = std::make_shared<Value>(2.0f);


    auto q_tensor = Tensor::from_values({q_val, q_other_dim_val}, {1,1,2});
    auto k_tensor = Tensor::from_values({k_val, k_other_dim_val}, {1,1,2});
    auto v_tensor = Tensor::from_values({v_val, v_other_dim_val}, {1,1,2});

    auto output_tensor = mha->forward(q_tensor, k_tensor, v_tensor);

    auto loss = std::make_shared<Value>(0.0f);
    for(const auto& val_ptr : output_tensor->data) {
        loss = loss + val_ptr;
    }

    ASSERT_NO_THROW({
        loss->backward();
    });

    // Gradients for q_val and k_val can be zero in specific simple single-token attention cases
    // depending on the softmax output and further operations.
    // The most important is that v_val and weights receive gradients.
    EXPECT_NE(v_val->grad, 0.0f); // dL/dV should generally be non-zero if V contributes

    bool has_non_zero_weight_grad = false;
    for(const auto& p : mha->parameters()){
        if (p->grad != 0.0f) {
            has_non_zero_weight_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero_weight_grad);
}


// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
