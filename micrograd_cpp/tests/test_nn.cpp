#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/Value.h"
#include "nn/Layer.h"
#include "nn/Linear.h"
#include "nn/ReLU.h"
#include "nn/Sigmoid.h"
#include "nn/Embedding.h"
#include "nn/LayerNorm.h"
#include "nn/RMSNorm.h"   // Added
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
    // Input(1, 2), Weights(3, 2), Bias(3) -> Output(1, 3)
    auto linear_layer = std::make_shared<Linear>(2, 3); // in_features=2, out_features=3

    // Manually set weights and biases for predictable output
    // Weights: [[w11, w12], [w21, w22], [w31, w32]] = [[1,2],[3,4],[5,6]] (shape 3x2)
    linear_layer->weights = Tensor::from_vector({1,2,3,4,5,6}, {3,2});
    // Bias: [b1,b2,b3] = [0.1, 0.2, 0.3] (shape 3)
    linear_layer->bias = Tensor::from_vector({0.1, 0.2, 0.3}, {3});

    auto input = Tensor::from_vector({10, 20}, {1, 2}); // shape 1x2 (batch_size=1, in_features=2)
    auto output = linear_layer->forward(input);

    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->ndim(), 2); // Output of Linear layer is always 2D [batch, out_features]
                                  // even if input batch is 1 and was passed as 1D
    EXPECT_EQ(output->shape[0], 1); // batch_size
    EXPECT_EQ(output->shape[1], 3); // out_features

    // Expected: input @ W.T + bias
    // W.T = [[1,3,5], [2,4,6]]
    // output_row0_col0 = 10*1 + 20*2 + 0.1 = 10 + 40 + 0.1 = 50.1
    // output_row0_col1 = 10*3 + 20*4 + 0.2 = 30 + 80 + 0.2 = 110.2
    // output_row0_col2 = 10*5 + 20*6 + 0.3 = 50 + 120 + 0.3 = 170.3
    check_tensor_data(output, {50.1f, 110.2f, 170.3f});
}

TEST(NNLayerTest, LinearForwardVectorInput) {
    // Input(2), Weights(3,2), Bias(3) -> Output(3)
    auto linear_layer = std::make_shared<Linear>(2, 3);
    linear_layer->weights = Tensor::from_vector({1,2,3,4,5,6}, {3,2});
    linear_layer->bias = Tensor::from_vector({0.1f, 0.2f, 0.3f}, {3});

    auto input_vector = Tensor::from_vector({10,20}, {2}); // 1D input
    auto output_vector = linear_layer->forward(input_vector);

    ASSERT_NE(output_vector, nullptr);
    EXPECT_EQ(output_vector->ndim(), 1);
    EXPECT_EQ(output_vector->shape[0], 3);
    check_tensor_data(output_vector, {50.1f, 110.2f, 170.3f});
}


TEST(NNLayerTest, LinearForwardNoBias) {
    auto linear_layer = std::make_shared<Linear>(2, 1, false); // in=2, out=1, no bias
    linear_layer->weights = Tensor::from_vector({2, 3}, {1,2}); // W = [[2,3]]
    auto input = Tensor::from_vector({5, 10}, {1,2}); // X = [[5,10]]
    auto output = linear_layer->forward(input);
    // Expected: X @ W.T = [[5,10]] @ [[2],[3]] = [[5*2 + 10*3]] = [[10+30]] = [[40]]
    check_tensor_data(output, {40.0f});
}

TEST(NNLayerTest, LinearBackward) {
    auto linear_layer = std::make_shared<Linear>(1, 1); // 1 input feature, 1 output feature
    // W = [[2.0]], B = [0.5]
    linear_layer->weights = Tensor::from_vector({2.0}, {1,1});
    linear_layer->bias = Tensor::from_vector({0.5}, {1});

    auto input_val = std::make_shared<Value>(3.0);
    // Pass input as a 1D tensor (vector)
    auto input_tensor = Tensor::from_values({input_val}, {1});

    auto output_tensor = linear_layer->forward(input_tensor); // Output = 3.0 * 2.0 + 0.5 = 6.5
                                                              // Output will be 1D tensor {6.5}
    ASSERT_EQ(output_tensor->data.size(), 1);
    output_tensor->data[0]->backward(); // Scalar output, call backward on its Value

    // Check gradients
    // Output L = x*w + b
    // dL/dx = w = 2.0
    // dL/dw = x = 3.0
    // dL/db = 1.0
    EXPECT_NEAR(input_val->grad, 2.0f, 1e-4);
    EXPECT_NEAR(linear_layer->weights->data[0]->grad, 3.0f, 1e-4);
    EXPECT_NEAR(linear_layer->bias->data[0]->grad, 1.0f, 1e-4);
}

TEST(NNLayerTest, LinearParameters) {
    auto layer_with_bias = std::make_shared<Linear>(2,3,true);
    EXPECT_EQ(layer_with_bias->parameters().size(), 2*3 + 3); // 6 weights + 3 biases

    auto layer_no_bias = std::make_shared<Linear>(2,3,false);
    EXPECT_EQ(layer_no_bias->parameters().size(), 2*3); // 6 weights
}


// --- ReLU Layer Tests ---
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

    auto output_tensor = relu_layer->forward(input_tensor); // Output: [[0.0, 3.0]]
    // Sum output for backward pass: L = sum(output_tensor_elements)
    auto loss = output_tensor->data[0] + output_tensor->data[1]; // 0.0 + 3.0 = 3.0
    loss->backward();

    // dL/dv1 = dL/d(relu(v1)) * d(relu(v1))/dv1 = 1 * 0 = 0 (since v1 < 0)
    // dL/dv2 = dL/d(relu(v2)) * d(relu(v2))/dv2 = 1 * 1 = 1 (since v2 > 0)
    EXPECT_NEAR(v1->grad, 0.0f, 1e-4);
    EXPECT_NEAR(v2->grad, 1.0f, 1e-4);
}

TEST(NNLayerTest, ReLUParameters) {
    auto relu_layer = std::make_shared<ReLU>();
    EXPECT_EQ(relu_layer->parameters().size(), 0);
}

// --- Sigmoid Layer Tests ---
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
TEST(NNLayerTest, EmbeddingForwardShapeAndValues) {
    auto embedding_layer = std::make_shared<Embedding>(10, 3); // 10 words, 3 dim embedding
    // Manually set some weights for predictability
    // Weight for index 2: [0.1, 0.2, 0.3]
    // Weight for index 5: [0.4, 0.5, 0.6]
    embedding_layer->weights->set({2,0}, std::make_shared<Value>(0.1f));
    embedding_layer->weights->set({2,1}, std::make_shared<Value>(0.2f));
    embedding_layer->weights->set({2,2}, std::make_shared<Value>(0.3f));
    embedding_layer->weights->set({5,0}, std::make_shared<Value>(0.4f));
    embedding_layer->weights->set({5,1}, std::make_shared<Value>(0.5f));
    embedding_layer->weights->set({5,2}, std::make_shared<Value>(0.6f));

    // Input: tensor of indices [[2, 5]]
    auto input_indices = Tensor::from_vector({2.0f, 5.0f}, {1, 2}); // batch_size=1, seq_len=2
    auto output = embedding_layer->forward(input_indices);

    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->ndim(), 3); // batch_size, seq_len, embedding_dim
    EXPECT_EQ(output->shape[0], 1);
    EXPECT_EQ(output->shape[1], 2);
    EXPECT_EQ(output->shape[2], 3); // embedding_dim

    // Expected output data: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    std::vector<float> expected_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    check_tensor_data(output, expected_data);
}

TEST(NNLayerTest, EmbeddingForwardSingleIndex) {
    auto embedding_layer = std::make_shared<Embedding>(5, 2); // 5 words, 2 dim
    embedding_layer->weights->set({3,0}, std::make_shared<Value>(1.0f));
    embedding_layer->weights->set({3,1}, std::make_shared<Value>(2.0f));

    auto output = embedding_layer->forward(3); // Single index
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->ndim(), 1);
    EXPECT_EQ(output->shape[0], 2);
    check_tensor_data(output, {1.0f, 2.0f});
}


TEST(NNLayerTest, EmbeddingBackward) {
    auto embedding_layer = std::make_shared<Embedding>(5, 1); // 5 words, 1 dim embedding
    // W_idx2 = [0.5], W_idx4 = [0.8]
    auto w2 = embedding_layer->weights->get({2,0});
    ASSERT_NE(w2, nullptr); w2->data = 0.5f;
    auto w4 = embedding_layer->weights->get({4,0});
    ASSERT_NE(w4, nullptr); w4->data = 0.8f;

    // Input indices: [[2, 4]] (as floats in Tensor)
    auto idx_tensor = Tensor::from_vector({2.0f, 4.0f}, {1,2});
    auto output_tensor = embedding_layer->forward(idx_tensor); // Output: [[[0.5], [0.8]]]

    // Sum output for backward pass: L = output_0_0_0 + output_0_1_0 = W_idx2 + W_idx4
    auto loss = output_tensor->get({0,0,0}) + output_tensor->get({0,1,0}); // 0.5 + 0.8 = 1.3
    loss->backward();

    // dL/dW_idx2 = 1
    // dL/dW_idx4 = 1
    // Other weights should have grad 0
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
TEST(NNLayerTest, LayerNormForwardValues) {
    int feature_dim = 3;
    auto ln_layer = std::make_shared<LayerNorm>(feature_dim, 1e-5f); // eps=1e-5
    // Manually set gamma and beta
    ln_layer->gamma = Tensor::from_vector({1.0f, 1.5f, 2.0f}, {feature_dim});
    ln_layer->beta = Tensor::from_vector({0.1f, 0.2f, 0.3f}, {feature_dim});

    // Input: [[1, 2, 3]]
    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, feature_dim});
    auto output = ln_layer->forward(input);

    // Calculations for input [1,2,3]:
    // mean = (1+2+3)/3 = 2
    // var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3 = 0.666666...
    // std_inv = 1 / sqrt(var + eps) = 1 / sqrt(0.666666 + 1e-5) = 1 / sqrt(0.666676) approx 1 / 0.816504 = 1.22473
    // Normalized:
    // x_norm_0 = (1-2)*1.22473 = -1.22473
    // x_norm_1 = (2-2)*1.22473 = 0
    // x_norm_2 = (3-2)*1.22473 = 1.22473
    // Output: gamma * x_norm + beta
    // y0 = 1.0 * -1.22473 + 0.1 = -1.12473
    // y1 = 1.5 * 0      + 0.2 = 0.2
    // y2 = 2.0 * 1.22473  + 0.3 = 2.44946 + 0.3 = 2.74946
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
    auto input_tensor = Tensor::from_values({x1, x2}, {1, feature_dim}); // [[1.0, 3.0]]

    auto output_tensor = ln_layer->forward(input_tensor);
    // mean = (1+3)/2 = 2
    // var = ((1-2)^2 + (3-2)^2)/2 = (1+1)/2 = 1 (using N as denominator)
    // std_inv = 1/sqrt(1+1e-5) approx 1.0 (more precisely 1/sqrt(1.00001) = 0.999995)
    // x_norm1 = (1-2)*std_inv approx -1.0
    // x_norm2 = (3-2)*std_inv approx  1.0
    // y1 = 1*x_norm1 + 0 approx -1.0
    // y2 = 1*x_norm2 + 0 approx  1.0

    auto loss = output_tensor->get({0,0}) + output_tensor->get({0,1}); // Sum of outputs
    loss->backward();

    // With L = y1+y2, and gamma=1, beta=0, dL/dx1 and dL/dx2 should be 0 due to mean subtraction.
    EXPECT_NEAR(x1->grad, 0.0f, 1e-4f);
    EXPECT_NEAR(x2->grad, 0.0f, 1e-4f);

    // dL/dgamma_i = sum_over_batch( (x_i - mean_i)/std_i ) * dL/dy_i
    // Here, dL/dy_i = 1. So dL/dgamma_1 = x_norm1 approx -1.0. dL/dgamma_2 = x_norm2 approx 1.0
    EXPECT_NEAR(ln_layer->gamma->get({0})->grad, -0.999995f, 1e-4f);
    EXPECT_NEAR(ln_layer->gamma->get({1})->grad, 0.999995f, 1e-4f);

    // dL/dbeta_i = dL/dy_i = 1
    EXPECT_NEAR(ln_layer->beta->get({0})->grad, 1.0f, 1e-4f);
    EXPECT_NEAR(ln_layer->beta->get({1})->grad, 1.0f, 1e-4f);
}

TEST(NNLayerTest, LayerNormParameters) {
    auto ln_layer = std::make_shared<LayerNorm>(10); // 10 features
    EXPECT_EQ(ln_layer->parameters().size(), 10 + 10); // 10 gamma + 10 beta
}

// --- RMSNorm Layer Tests ---
TEST(NNLayerTest, RMSNormForwardValues) {
    int feature_dim = 3;
    auto rms_layer = std::make_shared<RMSNorm>(feature_dim, 1e-5f); // eps=1e-5
    // Manually set gamma
    rms_layer->gamma = Tensor::from_vector({1.0f, 1.5f, 0.5f}, {feature_dim});

    // Input: [[1, 2, 3]]
    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, feature_dim});
    auto output = rms_layer->forward(input);

    // Calculations for input [1,2,3]:
    // x_sq = [1, 4, 9]
    // mean(x_sq) = (1+4+9)/3 = 14/3 = 4.666666...
    // rsqrt_val = 1 / sqrt(mean(x_sq) + eps) = 1 / sqrt(4.666666 + 1e-5) = 1 / sqrt(4.666676)
    //           = 1 / 2.160249 = 0.462910
    // Normalized_x (before gamma):
    // x_norm_0 = 1 * 0.462910 = 0.462910
    // x_norm_1 = 2 * 0.462910 = 0.925820
    // x_norm_2 = 3 * 0.462910 = 1.388730
    // Output: gamma * normalized_x
    // y0 = 1.0 * 0.462910 = 0.462910
    // y1 = 1.5 * 0.925820 = 1.388730
    // y2 = 0.5 * 1.388730 = 0.694365
    std::vector<float> expected_data = {0.462910f, 1.388730f, 0.694365f};
    check_tensor_data(output, expected_data, 1e-5f);
}

TEST(NNLayerTest, RMSNormBackward) {
    int feature_dim = 2;
    auto rms_layer = std::make_shared<RMSNorm>(feature_dim, 1e-5f);
    // gamma = [g1,g2] = [1.0, 1.0] for simplicity in checking input grads
    rms_layer->gamma = Tensor::from_vector({1.0f, 1.0f}, {feature_dim});

    auto x1_val = std::make_shared<Value>(1.0f);
    auto x2_val = std::make_shared<Value>(3.0f); // Input: [[1.0, 3.0]]
    auto input_tensor = Tensor::from_values({x1_val, x2_val}, {1, feature_dim});

    auto output_tensor = rms_layer->forward(input_tensor);
    // x_sq = [1, 9], mean(x_sq) = (1+9)/2 = 5
    // rsqrt_val = 1/sqrt(5+1e-5) approx 1/sqrt(5.00001) = 1/2.236071 = 0.447213
    // norm_x1 = 1 * 0.447213 = 0.447213
    // norm_x2 = 3 * 0.447213 = 1.341639
    // y1 = 1.0 * 0.447213 = 0.447213
    // y2 = 1.0 * 1.341639 = 1.341639

    // Loss = y1 + y2 = 0.447213 + 1.341639 = 1.788852
    auto loss = output_tensor->get({0,0}) + output_tensor->get({0,1});
    loss->backward();

    // Gradients are complex. Check they are non-zero.
    // dL/dg1, dL/dg2
    EXPECT_NE(rms_layer->gamma->get({0})->grad, 0.0f);
    EXPECT_NE(rms_layer->gamma->get({1})->grad, 0.0f);
    // dL/dx1, dL/dx2
    EXPECT_NE(x1_val->grad, 0.0f);
    EXPECT_NE(x2_val->grad, 0.0f);
}

TEST(NNLayerTest, RMSNormParameters) {
    auto rms_layer = std::make_shared<RMSNorm>(20); // 20 features
    EXPECT_EQ(rms_layer->parameters().size(), 20); // Only gamma parameters
}

// --- Conv2D Layer Tests ---
#include "nn/Conv2D.h" // Add include for Conv2D

TEST(NNLayerTest, Conv2DForwardSimple) {
    // Input: 1x1x3x3, Kernel: 1x1x2x2, Output: 1x1x2x2
    // Stride=1, Padding=0
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 1, 0, false); // in_c=1, out_c=1, kernel=2, stride=1, pad=0, no_bias

    // Weights: filter0_channel0 = [[1, 2], [3, 4]]
    conv_layer->weights = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2});

    // Input: batch1_channel0 = [[1,1,1], [1,1,1], [1,1,1]]
    auto input = Tensor::from_vector({1,1,1, 1,1,1, 1,1,1}, {1, 1, 3, 3});
    auto output = conv_layer->forward(input);

    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1); // N
    EXPECT_EQ(output->shape[1], 1); // C_out
    EXPECT_EQ(output->shape[2], 2); // H_out
    EXPECT_EQ(output->shape[3], 2); // W_out

    // Expected output:
    // O[0,0] = 1*1 + 1*2 + 1*3 + 1*4 = 1+2+3+4 = 10
    // O[0,1] = 1*1 + 1*2 + 1*3 + 1*4 = 10
    // O[1,0] = 1*1 + 1*2 + 1*3 + 1*4 = 10
    // O[1,1] = 1*1 + 1*2 + 1*3 + 1*4 = 10
    check_tensor_data(output, {10.0f, 10.0f, 10.0f, 10.0f});
}

TEST(NNLayerTest, Conv2DForwardWithBias) {
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 1, 0, true); // Use bias
    conv_layer->weights = Tensor::from_vector({1,0,0,1}, {1,1,2,2}); // [[1,0],[0,1]]
    conv_layer->biases = Tensor::from_vector({0.5f}, {1}); // bias for out_channel_0

    auto input = Tensor::from_vector({1,2,3, 1,1,1, 2,2,2}, {1,1,3,3}); // [[1,2,3],[1,1,1],[2,2,2]]
    auto output = conv_layer->forward(input);

    // O[0,0] = (1*1 + 2*0 + 1*0 + 1*1) + 0.5 = (1+0+0+1) + 0.5 = 2 + 0.5 = 2.5
    // O[0,1] = (2*1 + 3*0 + 1*0 + 1*1) + 0.5 = (2+0+0+1) + 0.5 = 3 + 0.5 = 3.5
    // O[1,0] = (1*1 + 1*0 + 2*0 + 2*1) + 0.5 = (1+0+0+2) + 0.5 = 3 + 0.5 = 3.5
    // O[1,1] = (1*1 + 1*0 + 2*0 + 2*1) + 0.5 = (1+0+0+2) + 0.5 = 3 + 0.5 = 3.5
    check_tensor_data(output, {2.5f, 3.5f, 3.5f, 3.5f});
}

TEST(NNLayerTest, Conv2DForwardWithStride) {
    // Input: 1x1x3x3, Kernel: 1x1x2x2, Stride=2, Padding=0, Output: 1x1x1x1
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 2, 0, false); // stride=2
    conv_layer->weights = Tensor::from_vector({1,2,3,4}, {1,1,2,2});
    auto input = Tensor::from_vector({1,1,1,1,1,1,1,1,1}, {1,1,3,3});
    auto output = conv_layer->forward(input);

    ASSERT_EQ(output->shape[2], 1); // H_out = floor((3-2+0)/2)+1 = floor(1/2)+1 = 0+1 = 1
    ASSERT_EQ(output->shape[3], 1); // W_out
    // O[0,0] = 1*1+1*2+1*3+1*4 = 10
    check_tensor_data(output, {10.0f});
}

TEST(NNLayerTest, Conv2DForwardWithPadding) {
    // Input: 1x1x2x2, Kernel: 1x1x2x2, Stride=1, Padding=1, Output: 1x1x3x3
    auto conv_layer = std::make_shared<Conv2D>(1, 1, 2, 1, 1, false); // padding=1
    conv_layer->weights = Tensor::from_vector({1,1,1,1}, {1,1,2,2}); // W = [[1,1],[1,1]]
    auto input = Tensor::from_vector({1,2,3,4}, {1,1,2,2}); // data: 1 (0,0), 2 (0,1), 3 (1,0), 4 (1,1)
    auto output = conv_layer->forward(input);

    ASSERT_EQ(output->shape[2], 3); // H_out = floor((2-2+2*1)/1)+1 = 2+1=3
    ASSERT_EQ(output->shape[3], 3); // W_out

    // Expected values based on manual calculation with padding:
    // Padded input conceptually:
    // 0 0 0 0
    // 0 1 2 0
    // 0 3 4 0
    // 0 0 0 0
    // Kernel [[1,1],[1,1]]
    // O[0,0] = 0*1+0*1+0*1+1*1 = 1
    // O[0,1] = 0*1+0*1+1*1+2*1 = 3
    // O[0,2] = 0*1+0*1+2*1+0*1 = 2
    // O[1,0] = 0*1+1*1+0*1+3*1 = 4
    // O[1,1] = 1*1+2*1+3*1+4*1 = 10
    // O[1,2] = 2*1+0*1+4*1+0*1 = 6
    // O[2,0] = 0*1+3*1+0*1+0*1 = 3
    // O[2,1] = 3*1+4*1+0*1+0*1 = 7
    // O[2,2] = 4*1+0*1+0*1+0*1 = 4
    std::vector<float> expected = {
        1.0f,  3.0f,  2.0f,
        4.0f, 10.0f,  6.0f,
        3.0f,  7.0f,  4.0f
    };
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
#include "nn/Conv1D.h" // Add include for Conv1D

TEST(NNLayerTest, Conv1DForwardSimple) {
    // Input: 1x1x5 (N, C_in, L_in), Kernel: 1x1x3 (C_out, C_in, K_L), Output: 1x1x3 (N, C_out, L_out)
    // Stride=1, Padding=0
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 3, 1, 0, false); // in_c=1, out_c=1, kernel=3, stride=1, pad=0, no_bias

    // Weights: filter0_channel0 = [1, 2, 3]
    conv_layer->weights = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, 1, 3});

    // Input: batch1_channel0 = [1,1,1,1,1]
    auto input = Tensor::from_vector({1.0f,1.0f,1.0f,1.0f,1.0f}, {1, 1, 5});
    auto output = conv_layer->forward(input);

    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1); // N
    EXPECT_EQ(output->shape[1], 1); // C_out
    EXPECT_EQ(output->shape[2], 3); // L_out = (5-3+0)/1 + 1 = 3

    // Expected output:
    // O[0] = 1*1 + 1*2 + 1*3 = 1+2+3 = 6
    // O[1] = 1*1 + 1*2 + 1*3 = 6
    // O[2] = 1*1 + 1*2 + 1*3 = 6
    check_tensor_data(output, {6.0f, 6.0f, 6.0f});
}

TEST(NNLayerTest, Conv1DForwardWithBias) {
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 2, 1, 0, true); // Use bias, kernel_size=2
    conv_layer->weights = Tensor::from_vector({1.0f, 0.5f}, {1,1,2}); // W = [1, 0.5]
    conv_layer->biases = Tensor::from_vector({0.1f}, {1}); // bias for out_channel_0

    auto input = Tensor::from_vector({1.0f,2.0f,3.0f}, {1,1,3}); // Input: [1,2,3]
    auto output = conv_layer->forward(input); // L_out = (3-2)/1 + 1 = 2

    // O[0] = (1*1 + 2*0.5) + 0.1 = (1+1) + 0.1 = 2.1
    // O[1] = (2*1 + 3*0.5) + 0.1 = (2+1.5) + 0.1 = 3.5 + 0.1 = 3.6
    check_tensor_data(output, {2.1f, 3.6f});
}

TEST(NNLayerTest, Conv1DForwardWithStride) {
    // Input: 1x1x5, Kernel: 1x1x3, Stride=2, Padding=0, Output: 1x1x2
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 3, 2, 0, false); // stride=2
    conv_layer->weights = Tensor::from_vector({1,0,1}, {1,1,3}); // W = [1,0,1]
    auto input = Tensor::from_vector({1,2,3,4,5}, {1,1,5}); // Input: [1,2,3,4,5]
    auto output = conv_layer->forward(input);

    ASSERT_EQ(output->shape[2], 2); // L_out = floor((5-3+0)/2)+1 = floor(2/2)+1 = 1+1 = 2
    // O[0] = (1*1 + 2*0 + 3*1) = 1+0+3 = 4
    // O[1] (starts at input index 2*1=2) = (3*1 + 4*0 + 5*1) = 3+0+5 = 8
    check_tensor_data(output, {4.0f, 8.0f});
}

TEST(NNLayerTest, Conv1DForwardWithPadding) {
    // Input: 1x1x3, Kernel: 1x1x3, Stride=1, Padding=1, Output: 1x1x3
    // Padded input becomes effectively length 5: [0, 1, 2, 3, 0]
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 3, 1, 1, false); // padding=1
    conv_layer->weights = Tensor::from_vector({1,1,1}, {1,1,3}); // W = [1,1,1]
    auto input = Tensor::from_vector({1,2,3}, {1,1,3}); // Input: [1,2,3]
    auto output = conv_layer->forward(input);

    ASSERT_EQ(output->shape[2], 3); // L_out = floor((3-3+2*1)/1)+1 = 2+1=3

    // O[0] (kernel on [0,1,2] of padded input) = 0*1+1*1+2*1 = 3
    // O[1] (kernel on [1,2,3] of padded input) = 1*1+2*1+3*1 = 6
    // O[2] (kernel on [2,3,0] of padded input) = 2*1+3*1+0*1 = 5
    check_tensor_data(output, {3.0f, 6.0f, 5.0f});
}

TEST(NNLayerTest, Conv1DBackwardSmall) {
    // Input: 1x1x3, Kernel: 1x1x2 (2 weights), Output: 1x1x2
    auto conv_layer = std::make_shared<Conv1D>(1, 1, 2, 1, 0, true); // bias=true
    // Weights w = [w1, w2], Bias b = [b0]
    conv_layer->weights = Tensor::from_vector({2.0f, 0.5f}, {1,1,2}); // w1=2, w2=0.5
    conv_layer->biases = Tensor::from_vector({0.1f}, {1}); // b0=0.1

    // Input values x = [x1, x2, x3]
    auto x1_v = std::make_shared<Value>(1.0f);
    auto x2_v = std::make_shared<Value>(2.0f);
    auto x3_v = std::make_shared<Value>(3.0f);
    auto input_tensor = Tensor::from_values({x1_v,x2_v,x3_v}, {1,1,3});

    auto output_tensor = conv_layer->forward(input_tensor);
    // Output y = [y1, y2]
    // y1 = x1*w1 + x2*w2 + b0 = 1*2 + 2*0.5 + 0.1 = 2+1+0.1 = 3.1
    // y2 = x2*w1 + x3*w2 + b0 = 2*2 + 3*0.5 + 0.1 = 4+1.5+0.1 = 5.6

    // Sum all output elements for a scalar loss L
    auto loss = output_tensor->get({0,0,0}) + output_tensor->get({0,0,1}); // L = y1+y2 = 3.1+5.6 = 8.7
    loss->backward();

    // dL/dw1 = x1 + x2 = 1+2 = 3
    // dL/dw2 = x2 + x3 = 2+3 = 5
    check_value_grad(conv_layer->weights->get({0,0,0}), 3.0f); // grad for w1
    check_value_grad(conv_layer->weights->get({0,0,1}), 5.0f); // grad for w2
    // dL/db0 = N_outputs_summed = 1+1 = 2 (since y1 and y2 both use b0 and are summed)
    check_value_grad(conv_layer->biases->get({0}), 2.0f);
    // dL/dx1 = w1 = 2.0
    // dL/dx2 = w1 (from y1) + w2 (from y2) = 2.0 + 0.5 = 2.5
    // dL/dx3 = w2 = 0.5
    check_value_grad(x1_v, 2.0f);
    check_value_grad(x2_v, 2.5f);
    check_value_grad(x3_v, 0.5f);
}

TEST(NNLayerTest, Conv1DParameters) {
    auto conv_with_bias = std::make_shared<Conv1D>(3,16,3,1,1,true); // in=3, out=16, k=3, bias=true
    // Weights: 16 * 3 * 3 = 144. Biases: 16. Total = 160
    EXPECT_EQ(conv_with_bias->parameters().size(), 16*3*3 + 16);

    auto conv_no_bias = std::make_shared<Conv1D>(3,16,3,1,1,false); // bias=false
    EXPECT_EQ(conv_no_bias->parameters().size(), 16*3*3);
}

// --- Scaled Dot-Product Attention Tests ---
#include "nn/attention.h" // Add include for attention functions

TEST(NNLayerTest, ScaledDotProductAttentionForwardSimple) {
    // Q, K, V: [batch_size=1, seq_len=1, d_k=2]
    // Q = [[1, 2]], K = [[3, 4]], V = [[5, 6]]
    // d_k = 2, sqrt(d_k) = 1.41421356
    auto q_tensor = Tensor::from_vector({1.0f, 2.0f}, {1, 1, 2});
    auto k_tensor = Tensor::from_vector({3.0f, 4.0f}, {1, 1, 2});
    auto v_tensor = Tensor::from_vector({5.0f, 6.0f}, {1, 1, 2}); // d_v = 2

    // K.T (manual for single K vector in batch): shape [1,2,1]
    // Q @ K.T = [1,2] @ [[3],[4]] = 1*3 + 2*4 = 3 + 8 = 11. (Scores shape [1,1,1])
    // Scaled score = 11 / 1.41421356 = 7.77817
    // Softmax of a single score is 1.0. (Attention weights shape [1,1,1], value is 1.0)
    // Output = AttentionWeights @ V.
    // Matmul: [B, Sq, Sk] @ [B, Sv, Dv] where Sk == Sv. Here Sk=1, Sv=1.
    // Output = 1.0 * V = [[5,6]] (shape [1,1,2])

    auto output = micrograd_nn::scaled_dot_product_attention(q_tensor, k_tensor, v_tensor);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1); // N
    EXPECT_EQ(output->shape[1], 1); // seq_len_q
    EXPECT_EQ(output->shape[2], 2); // d_v
    check_tensor_data(output, {5.0f, 6.0f}, 1e-4f);
}

TEST(NNLayerTest, ScaledDotProductAttentionForwardMultipleKeys) {
    // Q: [1,1,2] = [[1,0]]
    // K: [1,2,2] = [[[1,0], [0,1]]] (Key0, Key1)
    // V: [1,2,2] = [[[0.1,0.2], [0.3,0.4]]] (Value0, Value1)
    // d_k = 2, scale = 1/sqrt(2) = 0.70710678
    auto q = Tensor::from_vector({1.0f, 0.0f}, {1,1,2});
    auto k = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f}, {1,2,2});
    auto v = Tensor::from_vector({0.1f,0.2f, 0.3f,0.4f}, {1,2,2});

    // Q @ K.T:
    // K.T (batch item 0): [[1,0],[0,1]] (shape [2,2] conceptually for slice)
    // scores_slice = [1,0] @ [[1,0],[0,1]] = [1*1+0*0, 1*0+0*1] = [1, 0]
    // Scaled_scores_slice = [1*0.70710678, 0*0.70710678] = [0.70710678, 0]
    // Softmax([0.70710678, 0]):
    // exp_scores = [exp(0.70710678), exp(0)] = [2.02808, 1.0]
    // sum_exp = 3.02808
    // attn_w_slice = [2.02808/3.02808, 1.0/3.02808] = [0.66977, 0.33023] (shape [1,1,2] for this item)
    // Output_slice = attn_w_slice @ V_slice = [0.66977, 0.33023] @ [[0.1,0.2], [0.3,0.4]]
    //        = 0.66977 * [0.1,0.2] + 0.33023 * [0.3,0.4]
    //        = [0.066977, 0.133954] + [0.099069, 0.132092]
    //        = [0.166046, 0.266046]
    auto output = micrograd_nn::scaled_dot_product_attention(q, k, v);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape[0], 1); // N
    EXPECT_EQ(output->shape[1], 1); // seq_len_q
    EXPECT_EQ(output->shape[2], 2); // d_v
    check_tensor_data(output, {0.166046f, 0.266046f}, 1e-5f);
}


TEST(NNLayerTest, ScaledDotProductAttentionWithMask) {
    auto q = Tensor::from_vector({1.0f, 0.0f}, {1,1,2});
    auto k = Tensor::from_vector({1.0f,0.0f, 0.0f,1.0f, 1.0f, 1.0f}, {1,3,2}); // 3 keys
    auto v = Tensor::from_vector({0.1f,0.2f,  0.3f,0.4f,  0.5f,0.6f}, {1,3,2}); // 3 values
    // Mask out the second key/value pair. Mask shape [1,1,3] (N, Sq, Sk)
    // Mask value of 1.0 means mask it (set to large negative before softmax)
    auto mask = Tensor::from_vector({0.0f, 1.0f, 0.0f}, {1,1,3});

    // Q = [1,0], scale = 0.70710678
    // K0=[1,0], K1=[0,1], K2=[1,1]
    // Scores before scaling: Q@K0.T=1, Q@K1.T=0, Q@K2.T=1
    // Scaled scores: [0.70710678, 0, 0.70710678]
    // Masked scores (add -1e9 to where mask is 1.0): [0.70710678, -1e9, 0.70710678]
    // Softmax of this will give very low weight to middle element.
    // exp_scores approx [exp(0.70710678), 0, exp(0.70710678)] = [2.02808, 0, 2.02808]
    // sum_exp approx 4.05616
    // attn_w approx [0.5, 0, 0.5] (actually 2.02808/4.05616 = 0.49999)
    // Output = 0.5*V0 + 0*V1 + 0.5*V2
    //        = 0.5*[0.1,0.2] + 0.5*[0.5,0.6]
    //        = [0.05, 0.1] + [0.25, 0.3]
    //        = [0.3, 0.4]
    auto output = micrograd_nn::scaled_dot_product_attention(q, k, v, mask);
    check_tensor_data(output, {0.3f, 0.4f}, 1e-5f); // Using 1e-5 due to softmax precision
}

TEST(NNLayerTest, ScaledDotProductAttentionBackward) {
    // Q, K, V: [1,1,1], all Value(1.0)
    auto q_val = std::make_shared<Value>(1.0f);
    auto k_val = std::make_shared<Value>(1.0f);
    auto v_val = std::make_shared<Value>(1.0f); // d_v = 1

    auto q_tensor = Tensor::from_values({q_val}, {1,1,1}); // d_k = 1
    auto k_tensor = Tensor::from_values({k_val}, {1,1,1});
    auto v_tensor = Tensor::from_values({v_val}, {1,1,1});

    // Q@K.T = 1*1 = 1. sqrt(d_k)=1. Scaled_score=1. Softmax(1)=1. Output = 1*V = 1.
    auto output_tensor = micrograd_nn::scaled_dot_product_attention(q_tensor, k_tensor, v_tensor);
    auto loss = output_tensor->get({0,0,0}); // Loss is the single output value
    ASSERT_NE(loss, nullptr);
    loss->backward();

    // Check if grads are non-zero (exact values are complex for full SDPA)
    // For this specific case (Q,K,V are all 1.0, d_k=1), dL/dQ and dL/dK are 0.
    EXPECT_NEAR(q_val->grad, 0.0f, 1e-4f);
    EXPECT_NEAR(k_val->grad, 0.0f, 1e-4f);
    EXPECT_NE(v_val->grad, 0.0f); // Should still be non-zero

    // Simple check for V's gradient: dL/dV = AttentionWeight. Here, AttentionWeight is 1. So dL/dV = 1.
    EXPECT_NEAR(v_val->grad, 1.0f, 1e-4f);
}


// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
