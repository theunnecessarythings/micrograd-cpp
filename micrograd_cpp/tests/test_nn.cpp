#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/Value.h"
#include "nn/Layer.h"
#include "nn/Linear.h"
#include "nn/ReLU.h"
#include "nn/Sigmoid.h"
#include "nn/Embedding.h" // Added
#include "nn/LayerNorm.h" // Added
#include <memory> // For std::make_shared
#include <vector>
#include <cmath> // For std::exp, std::fabs

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


// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
