#include "optim/Adam.h"
#include <cmath>     // For std::sqrt, std::pow
#include <stdexcept> // For std::invalid_argument
#include <string>    // For std::to_string in error messages

Adam::Adam(const std::vector<std::shared_ptr<Value>>& params,
           float lr, float b1, float b2, float eps)
    : Optimizer(params), learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {

    if (lr < 0.0f) {
        throw std::invalid_argument("Learning rate must be non-negative. Got: " + std::to_string(lr));
    }
    if (b1 < 0.0f || b1 >= 1.0f) {
        throw std::invalid_argument("Beta1 must be in [0, 1). Got: " + std::to_string(b1));
    }
    if (b2 < 0.0f || b2 >= 1.0f) {
        throw std::invalid_argument("Beta2 must be in [0, 1). Got: " + std::to_string(b2));
    }
    if (eps < 0.0f) {
        throw std::invalid_argument("Epsilon must be non-negative. Got: " + std::to_string(eps));
    }

    for (const auto& p : params) {
        if (p) {
            // Initialize m and v with zeros.
            m_data[p.get()] = 0.0f;
            v_data[p.get()] = 0.0f;
        }
    }
}

void Adam::step() {
    t++; // Increment timestep

    for (const auto& p : params) {
        if (!p) continue; // Skip if parameter is null

        float grad = p->grad; // Current gradient for parameter p
        const Value* p_ptr = p.get(); // Get raw pointer for map key

        // Update biased first moment estimate
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        m_data[p_ptr] = beta1 * m_data[p_ptr] + (1.0f - beta1) * grad;

        // Update biased second raw moment estimate
        // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        v_data[p_ptr] = beta2 * v_data[p_ptr] + (1.0f - beta2) * (grad * grad);

        // Compute bias-corrected first moment estimate
        // m_hat_t = m_t / (1 - beta1^t)
        // Avoid division by zero if t is very large and beta1^t is close to 1, though practically beta1^t -> 0 for large t.
        // More critical is 1 - beta1^t becoming 0 if t is small and somehow beta1 is 1 (guarded by constructor).
        float m_hat = m_data[p_ptr] / (1.0f - std::pow(beta1, static_cast<float>(t))); // Cast t for pow

        // Compute bias-corrected second raw moment estimate
        // v_hat_t = v_t / (1 - beta2^t)
        float v_hat = v_data[p_ptr] / (1.0f - std::pow(beta2, static_cast<float>(t))); // Cast t for pow

        // Update parameters
        // theta_t = theta_{t-1} - learning_rate * m_hat_t / (sqrt(v_hat_t) + epsilon)
        p->data -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}
