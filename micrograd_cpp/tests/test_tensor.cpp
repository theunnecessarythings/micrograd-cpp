#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/Value.h" // For check_grad and accessing Value members
#include <memory> // For std::make_shared
#include <vector>
#include <numeric> // For std::iota

// Helper function to check gradients of all Value objects in a Tensor
void check_tensor_grads(const std::shared_ptr<Tensor>& tensor, const std::vector<float>& expected_grads, float tol = 1e-4) {
    ASSERT_EQ(tensor->data.size(), expected_grads.size());
    for (size_t i = 0; i < tensor->data.size(); ++i) {
        ASSERT_NE(tensor->data[i], nullptr) << "Null Value pointer in tensor at index " << i;
        EXPECT_NEAR(tensor->data[i]->grad, expected_grads[i], tol) << "Gradient mismatch at index " << i;
    }
}

// Helper to get a specific Value's grad for checking
float get_grad(const std::shared_ptr<Tensor>& t, const std::vector<int>& indices) {
    auto val = t->get(indices);
    EXPECT_NE(val, nullptr);
    return val->grad;
}

// Helper to get a specific Value's data for checking
float get_data(const std::shared_ptr<Tensor>& t, const std::vector<int>& indices) {
    auto val = t->get(indices);
    EXPECT_NE(val, nullptr);
    return val->data;
}


TEST(TensorTest, Creation) {
    auto t_zeros = Tensor::zeros({2, 2});
    ASSERT_NE(t_zeros, nullptr);
    EXPECT_EQ(t_zeros->shape[0], 2);
    EXPECT_EQ(t_zeros->shape[1], 2);
    EXPECT_EQ(get_data(t_zeros, {0,0}), 0.0f);

    auto t_ones = Tensor::ones({2, 3});
    ASSERT_NE(t_ones, nullptr);
    EXPECT_EQ(t_ones->shape[0], 2);
    EXPECT_EQ(t_ones->shape[1], 3);
    EXPECT_EQ(get_data(t_ones, {1,2}), 1.0f);

    std::vector<float> data_vec = {1, 2, 3, 4};
    auto t_from_vec = Tensor::from_vector(data_vec, {2, 2});
    ASSERT_NE(t_from_vec, nullptr);
    EXPECT_EQ(get_data(t_from_vec, {0,0}), 1.0f);
    EXPECT_EQ(get_data(t_from_vec, {0,1}), 2.0f);
    EXPECT_EQ(get_data(t_from_vec, {1,0}), 3.0f);
    EXPECT_EQ(get_data(t_from_vec, {1,1}), 4.0f);

    auto t_randn = Tensor::randn({5,5});
    ASSERT_NE(t_randn, nullptr);
    EXPECT_EQ(t_randn->numel(), 25);
}

TEST(TensorTest, ElementWiseAddition) {
    auto a_data = std::vector<float>{1, 2, 3, 4};
    auto b_data = std::vector<float>{5, 6, 7, 8};
    auto a = Tensor::from_vector(a_data, {2, 2});
    auto b = Tensor::from_vector(b_data, {2, 2});
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    auto c = *a + b; // Using operator+ overload for (Tensor, shared_ptr<Tensor>)
    ASSERT_NE(c, nullptr);

    // Sum all elements of c to get a scalar for backward pass
    auto sum_c = std::make_shared<Value>(0.0f, "sum_c_add");
    ASSERT_NE(sum_c, nullptr);
    for(const auto& val_ptr : c->data){
        ASSERT_NE(val_ptr, nullptr);
        sum_c = sum_c + val_ptr; // Uses Value's operator+
    }
    sum_c->backward();

    EXPECT_EQ(get_data(c, {0,0}), 6.0f); // 1+5
    EXPECT_EQ(get_data(c, {1,1}), 12.0f); // 4+8
    check_tensor_grads(a, {1.0f, 1.0f, 1.0f, 1.0f}); // d(sum_c)/da_ij = 1
    check_tensor_grads(b, {1.0f, 1.0f, 1.0f, 1.0f}); // d(sum_c)/db_ij = 1
}

TEST(TensorTest, ElementWiseMultiplication) {
    auto a_data = std::vector<float>{1, 2, 3, 4}; // a_00=1, a_01=2, a_10=3, a_11=4
    auto b_data = std::vector<float>{5, 6, 7, 8}; // b_00=5, b_01=6, b_10=7, b_11=8
    auto a = Tensor::from_vector(a_data, {2, 2});
    auto b = Tensor::from_vector(b_data, {2, 2});
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    auto c = *a * b; // c_ij = a_ij * b_ij
    ASSERT_NE(c, nullptr);
    // c_00 = 1*5=5, c_01=2*6=12, c_10=3*7=21, c_11=4*8=32
    auto sum_c = std::make_shared<Value>(0.0f, "sum_c_mul");
    ASSERT_NE(sum_c, nullptr);
    for(const auto& val_ptr : c->data){
        ASSERT_NE(val_ptr, nullptr);
        sum_c = sum_c + val_ptr;
    }
    sum_c->backward(); // L = sum(c_ij) = sum(a_ij * b_ij)

    EXPECT_EQ(get_data(c, {0,0}), 5.0f);
    EXPECT_EQ(get_data(c, {1,1}), 32.0f);

    // dL/da_ij = b_ij
    // dL/db_ij = a_ij
    check_tensor_grads(a, {5.0f, 6.0f, 7.0f, 8.0f}); // Grads of a are values of b
    check_tensor_grads(b, {1.0f, 2.0f, 3.0f, 4.0f}); // Grads of b are values of a
}

TEST(TensorTest, ScalarAddition) {
    auto a_data = std::vector<float>{1, 2, 3, 4};
    auto a = Tensor::from_vector(a_data, {2, 2});
    ASSERT_NE(a, nullptr);

    auto c = *a + 2.0f; // c_ij = a_ij + 2.0
    ASSERT_NE(c, nullptr);
    auto sum_c = std::make_shared<Value>(0.0f, "sum_c_sadd");
    ASSERT_NE(sum_c, nullptr);
    for(const auto& val_ptr : c->data){
        ASSERT_NE(val_ptr, nullptr);
        sum_c = sum_c + val_ptr;
    }
    sum_c->backward(); // L = sum(a_ij + 2.0)

    EXPECT_EQ(get_data(c, {0,0}), 3.0f);
    EXPECT_EQ(get_data(c, {1,1}), 6.0f);
    check_tensor_grads(a, {1.0f, 1.0f, 1.0f, 1.0f}); // dL/da_ij = 1
}

TEST(TensorTest, ScalarMultiplication) {
    auto a_data = std::vector<float>{1, 2, 3, 4};
    auto a = Tensor::from_vector(a_data, {2, 2});
    ASSERT_NE(a, nullptr);

    auto c = *a * 3.0f; // c_ij = a_ij * 3.0
    ASSERT_NE(c, nullptr);
    auto sum_c = std::make_shared<Value>(0.0f, "sum_c_smul");
    ASSERT_NE(sum_c, nullptr);
    for(const auto& val_ptr : c->data){
        ASSERT_NE(val_ptr, nullptr);
        sum_c = sum_c + val_ptr;
    }
    sum_c->backward(); // L = sum(a_ij * 3.0)

    EXPECT_EQ(get_data(c, {0,0}), 3.0f);
    EXPECT_EQ(get_data(c, {1,1}), 12.0f);
    check_tensor_grads(a, {3.0f, 3.0f, 3.0f, 3.0f}); // dL/da_ij = 3.0
}


TEST(TensorTest, ReLUActivation) {
    auto a_data = std::vector<float>{-1, 2, -3, 4};
    auto a = Tensor::from_vector(a_data, {2, 2});
    ASSERT_NE(a, nullptr);

    auto b = a->relu(); // b_ij = relu(a_ij)
    ASSERT_NE(b, nullptr);
    // b_data should be {0, 2, 0, 4}
    auto sum_b = std::make_shared<Value>(0.0f, "sum_b_relu");
    ASSERT_NE(sum_b, nullptr);
    for(const auto& val_ptr : b->data){
        ASSERT_NE(val_ptr, nullptr);
        sum_b = sum_b + val_ptr;
    }
    sum_b->backward(); // L = sum(relu(a_ij))

    EXPECT_EQ(get_data(b, {0,0}), 0.0f);
    EXPECT_EQ(get_data(b, {0,1}), 2.0f);
    EXPECT_EQ(get_data(b, {1,0}), 0.0f);
    EXPECT_EQ(get_data(b, {1,1}), 4.0f);

    // dL/da_ij = 1 if a_ij > 0, else 0
    check_tensor_grads(a, {0.0f, 1.0f, 0.0f, 1.0f});
}

TEST(TensorTest, MatMul_2x2_2x2) {
    // A = [[1, 2],  B = [[5, 6],  C = A @ B = [[1*5+2*7, 1*6+2*8],  -> [[19, 22],
    //      [3, 4]]       [7, 8]]               [3*5+4*7, 3*6+4*8]]      [43, 50]]
    auto a_vals = Tensor::from_vector({1,2,3,4}, {2,2});
    auto b_vals = Tensor::from_vector({5,6,7,8}, {2,2});
    ASSERT_NE(a_vals, nullptr);
    ASSERT_NE(b_vals, nullptr);

    auto c_vals = a_vals->matmul(b_vals);
    ASSERT_NE(c_vals, nullptr);

    EXPECT_EQ(get_data(c_vals, {0,0}), 1*5+2*7); // 19
    EXPECT_EQ(get_data(c_vals, {0,1}), 1*6+2*8); // 22
    EXPECT_EQ(get_data(c_vals, {1,0}), 3*5+4*7); // 43
    EXPECT_EQ(get_data(c_vals, {1,1}), 3*6+4*8); // 50

    // To test autograd, sum all elements of C and backpropagate
    auto sum_c = std::make_shared<Value>(0.0f, "sum_c_matmul");
    ASSERT_NE(sum_c, nullptr);
    for(const auto& val_ptr : c_vals->data){
        ASSERT_NE(val_ptr, nullptr);
        sum_c = sum_c + val_ptr;
    }
    sum_c->backward(); // L = sum(C_ij)

    // dL/dC_ij = 1 for all i,j
    // Gradients for A: (dL/dC) @ B.T
    // Gradients for B: A.T @ (dL/dC)
    // Since dL/dC is a matrix of ones:
    // dA = [[1,1],[1,1]] @ B.T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[5+6, 7+8],[5+6, 7+8]] = [[11,15],[11,15]]
    // dB = A.T @ [[1,1],[1,1]] = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[1+3, 1+3],[2+4, 2+4]] = [[4,4],[6,6]]

    check_tensor_grads(a_vals, {5.0f+6.0f, 7.0f+8.0f, 5.0f+6.0f, 7.0f+8.0f}); // Expected: {11, 15, 11, 15}
    check_tensor_grads(b_vals, {1.0f+3.0f, 1.0f+3.0f, 2.0f+4.0f, 2.0f+4.0f}); // Expected: {4, 4, 6, 6}
}

TEST(TensorTest, MatMul_1x3_3x1) {
    // A = [[1, 2, 3]] (shape 1x3)
    // B = [[4], [5], [6]] (shape 3x1)
    // C = A @ B = [[1*4 + 2*5 + 3*6]] = [[4 + 10 + 18]] = [[32]] (shape 1x1)
    auto a_vals = Tensor::from_vector({1,2,3}, {1,3});
    auto b_vals = Tensor::from_vector({4,5,6}, {3,1});
    ASSERT_NE(a_vals, nullptr);
    ASSERT_NE(b_vals, nullptr);

    auto c_vals = a_vals->matmul(b_vals);
    ASSERT_NE(c_vals, nullptr);
    EXPECT_EQ(c_vals->shape[0], 1);
    EXPECT_EQ(c_vals->shape[1], 1);
    EXPECT_EQ(get_data(c_vals, {0,0}), 32.0f);

    c_vals->backward(); // Calls backward on the single Value in c_vals

    // L = C_00 = A_00*B_00 + A_01*B_10 + A_02*B_20
    // dL/dA_00 = B_00 = 4
    // dL/dA_01 = B_10 = 5
    // dL/dA_02 = B_20 = 6
    check_tensor_grads(a_vals, {4.0f, 5.0f, 6.0f});

    // dL/dB_00 = A_00 = 1
    // dL/dB_10 = A_01 = 2
    // dL/dB_20 = A_02 = 3
    check_tensor_grads(b_vals, {1.0f, 2.0f, 3.0f});
}

TEST(TensorTest, ZeroGrad) {
    auto a_data = std::vector<float>{1, 2, 3, 4};
    auto a = Tensor::from_vector(a_data, {2, 2});
    ASSERT_NE(a, nullptr);
    auto b = *a * 2.0f;
    ASSERT_NE(b, nullptr);

    auto sum_b = std::make_shared<Value>(0.0f, "sum_b_zerograd");
    ASSERT_NE(sum_b, nullptr);
    for(const auto& val_ptr : b->data){
        ASSERT_NE(val_ptr, nullptr);
        sum_b = sum_b + val_ptr;
    }
    sum_b->backward();
    check_tensor_grads(a, {2.0f, 2.0f, 2.0f, 2.0f});

    a->zero_grad();
    check_tensor_grads(a, {0.0f, 0.0f, 0.0f, 0.0f});
}


// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
