#include "gtest/gtest.h"
#include "core/Value.h"
#include <memory> // For std::make_shared

// Helper function to check gradients
void check_grad(const std::shared_ptr<Value>& val, float expected_grad, float tol = 1e-4) {
    EXPECT_NEAR(val->grad, expected_grad, tol);
}

TEST(ValueTest, Addition) {
    auto a = std::make_shared<Value>(2.0f);
    auto b = std::make_shared<Value>(3.0f);
    auto c = a + b; // c = 2 + 3 = 5
    c->backward();

    EXPECT_EQ(c->data, 5.0f);
    check_grad(a, 1.0f); // dc/da = 1
    check_grad(b, 1.0f); // dc/db = 1
}

TEST(ValueTest, Subtraction) {
    auto a = std::make_shared<Value>(5.0f);
    auto b = std::make_shared<Value>(2.0f);
    auto c = a - b; // c = 5 - 2 = 3
    c->backward();

    EXPECT_EQ(c->data, 3.0f);
    check_grad(a, 1.0f);  // dc/da = 1
    check_grad(b, -1.0f); // dc/db = -1
}

TEST(ValueTest, Multiplication) {
    auto a = std::make_shared<Value>(2.0f);
    auto b = std::make_shared<Value>(3.0f);
    auto c = a * b; // c = 2 * 3 = 6
    c->backward();

    EXPECT_EQ(c->data, 6.0f);
    check_grad(a, 3.0f); // dc/da = b = 3
    check_grad(b, 2.0f); // dc/db = a = 2
}

TEST(ValueTest, Division) {
    auto a = std::make_shared<Value>(6.0f);
    auto b = std::make_shared<Value>(2.0f);
    auto c = a / b; // c = 6 / 2 = 3
    c->backward();

    EXPECT_EQ(c->data, 3.0f);
    check_grad(a, 1.0f / 2.0f); // dc/da = 1/b = 0.5
    check_grad(b, -6.0f / (2.0f * 2.0f)); // dc/db = -a/b^2 = -6/4 = -1.5
}

TEST(ValueTest, Power) {
    auto a = std::make_shared<Value>(2.0f);
    auto c = a->pow(3.0f); // c = 2^3 = 8
    c->backward();

    EXPECT_EQ(c->data, 8.0f);
    check_grad(a, 3.0f * 2.0f * 2.0f); // dc/da = 3*a^2 = 3*4 = 12
}

TEST(ValueTest, ReLU) {
    auto a = std::make_shared<Value>(2.0f);
    auto b = a->relu(); // b = relu(2) = 2
    b->backward();
    EXPECT_EQ(b->data, 2.0f);
    check_grad(a, 1.0f);

    auto a2 = std::make_shared<Value>(-2.0f);
    auto c = a2->relu(); // c = relu(-2) = 0
    // c->backward(); // This was the bug, backward() must be called on the *final* result.
    // For this simple case, 'c' is the final result for a2's gradient path.
    c->backward();
    EXPECT_EQ(c->data, 0.0f);
    check_grad(a2, 0.0f);
}

TEST(ValueTest, ComplexExpression1) {
    // L = d*a + e*b
    // d = c + a = (a*b) + a
    // e = c + b = (a*b) + b
    auto a = std::make_shared<Value>(2.0f, "a");
    auto b = std::make_shared<Value>(3.0f, "b");
    auto c = a * b; c->op = "c"; // c = 6
    auto d = c + a; d->op = "d"; // d = 6 + 2 = 8
    auto e = c + b; e->op = "e"; // e = 6 + 3 = 9
    auto L = d * a + e * b; L->op = "L"; // L = 8*2 + 9*3 = 16 + 27 = 43
    L->backward();

    EXPECT_EQ(L->data, 43.0f);
    // Manual gradient calculation:
    // L = (a*b + a)*a + (a*b + b)*b
    // L = a^2*b + a^2 + a*b^2 + b^2
    // dL/da = 2ab + 2a + b^2 = 2*2*3 + 2*2 + 3^2 = 12 + 4 + 9 = 25
    // dL/db = a^2 + 2ab + 2b = 2^2 + 2*2*3 + 2*3 = 4 + 12 + 6 = 22
    check_grad(a, 25.0f);
    check_grad(b, 22.0f);
}

TEST(ValueTest, ComplexExpression2_FromMicrogradPython) {
    // Based on the micrograd example:
    // a = Value(-4.0)
    // b = Value(2.0)
    // c = a + b -> -2.0
    // d = a * b + b**3 -> -4*2 + 2^3 = -8 + 8 = 0
    // c = c + c + 1 -> -2 + -2 + 1 = -3
    // c = c + 1 + c + (-a) -> -3 + 1 + (-3) + 4 = -1
    // d = d + d * 2 + (b + a).relu() -> 0 + 0*2 + (2-4).relu() = 0 + 0 + (-2).relu() = 0
    // d = d + 3 * d + (b - a).relu() -> 0 + 3*0 + (2 - (-4)).relu() = (6).relu() = 6
    // e = c - d -> -1 - 6 = -7
    // f = e**2 -> (-7)^2 = 49
    // g = f / 2.0 -> 49/2 = 24.5
    // g = g + 10.0 / f -> 24.5 + 10/49 = 24.5 + 0.20408... = 24.70408...
    // g.backward()

    auto a = std::make_shared<Value>(-4.0f);
    auto b = std::make_shared<Value>(2.0f);
    auto one = std::make_shared<Value>(1.0f); // Constant 1
    auto two = std::make_shared<Value>(2.0f); // Constant 2
    auto three = std::make_shared<Value>(3.0f); // Constant 3
    auto ten = std::make_shared<Value>(10.0f); // Constant 10


    auto c_val = a + b; // Renamed to c_val to avoid conflict with 'c' in ReLU test
    auto d_val = a * b + b->pow(3.0f); // Renamed to d_val
    c_val = c_val + c_val + one;
    // For -a, create a temporary Value for -1.0f
    auto minus_one = std::make_shared<Value>(-1.0f);
    auto neg_a = a * minus_one;
    c_val = c_val + one + c_val + neg_a;

    d_val = d_val + d_val * two + (b + a)->relu();
    d_val = d_val + three * d_val + (b - a)->relu();
    auto e = c_val - d_val;
    auto f = e->pow(2.0f);
    auto g = f / two;
    g = g + ten / f;
    g->backward();

    // Expected gradients from Python micrograd
    // print(f'{a.grad=}, {b.grad=}')
    // a.grad=138.8338165283203, b.grad=645.5772705078125
    // For this C++ version, using shared_ptr for constants means they also get grads.
    // We only care about a and b here.
    check_grad(a, 138.8338f);
    check_grad(b, 645.5773f);
}

// Test case from Karpathy's video for a single neuron
TEST(ValueTest, SingleNeuron) {
    // inputs x1, x2
    auto x1 = std::make_shared<Value>(2.0f);
    auto x2 = std::make_shared<Value>(0.0f);
    // weights w1, w2
    auto w1 = std::make_shared<Value>(-3.0f);
    auto w2 = std::make_shared<Value>(1.0f);
    // bias of the neuron
    auto bias = std::make_shared<Value>(6.8813735870195432f); // So that n.data is exactly 7 for tanh test

    // x1*w1 + x2*w2 + bias
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto n = x1w1x2w2 + bias; // Net input to neuron

    // auto o = n->tanh(); // Tanh can be added later
    // Using a simple expression for now: o = n * 2 (Activation: linear with slope 2)
    // Let's use ReLU for now as tanh is not implemented
    auto o = n->relu();

    o->backward();

    // Expected:
    // o = relu(x1*w1 + x2*w2 + bias)
    // n = 2*(-3) + 0*1 + 6.88137... = -6 + 0 + 6.88137... = 0.88137...
    // o = relu(0.88137...) = 0.88137...
    // do/dn = 1 (since n > 0 for relu)
    //
    // do/dx1 = do/dn * dn/dx1 = 1 * w1 = -3
    // do/dx2 = do/dn * dn/dx2 = 1 * w2 = 1
    // do/dw1 = do/dn * dn/dw1 = 1 * x1 = 2
    // do/dw2 = do/dn * dn/dw2 = 1 * x2 = 0
    // do/dbias = do/dn * dn/dbias = 1 * 1 = 1
    EXPECT_NEAR(o->data, 0.881373587f, 1e-5);

    check_grad(x1, -3.0f);
    check_grad(x2, 1.0f);
    check_grad(w1, 2.0f);
    check_grad(w2, 0.0f);
    check_grad(bias, 1.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
