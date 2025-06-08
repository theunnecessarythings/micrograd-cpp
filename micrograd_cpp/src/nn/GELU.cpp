#include "nn/GELU.h"
#include "core/Value.h"
#include <cmath>     // For M_PI, std::sqrt, std::tanh (though Value::tanh is used)
#include <stdexcept> // For std::runtime_error, std::invalid_argument
#include <vector>    // For std::vector
#include <string>    // For std::to_string

// Define M_PI if not defined (e.g. on MSVC or other compilers not defining it by default)
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * x^3)))
std::shared_ptr<Tensor> GELU::forward(const std::shared_ptr<Tensor>& input) {
    if (!input) {
        throw std::runtime_error("GELU::forward: Input tensor cannot be null.");
    }

    std::vector<std::shared_ptr<Value>> output_values;
    output_values.reserve(input->numel());

    // Constants as Value objects to integrate into the graph if they were parameters
    // or to interact with other Value objects.
    // Since these are fixed constants for the GELU formula, creating them once might be an optimization,
    // but creating them per-element ensures they don't carry state if Value objects were stateful beyond their data/grad.
    // For this implementation, it's fine to create them repeatedly or once.
    // Let's make them once outside the loop for minor efficiency, though for Value objects,
    // the overhead is primarily graph construction, not float creation.

    // These constants don't need to be Value objects themselves if they only scale/add to other Values.
    // Using raw floats for constants that multiply with Value objects will rely on operator overloads
    // like Value::operator*(float) if available, or explicit Value creation.
    // Value class does not have Value*float operator, but Tensor has Tensor*float.
    // To ensure graph operations, it's safer to make them Value objects or ensure Value has scalar ops.
    // Current Value ops take shared_ptr<Value>, so these constants need to be Value objects.

    auto const_0_5 = std::make_shared<Value>(0.5f, "c_0.5");
    auto const_1 = std::make_shared<Value>(1.0f, "c_1");
    auto const_sqrt_2_div_pi = std::make_shared<Value>(static_cast<float>(std::sqrt(2.0 / M_PI)), "c_sqrt_2_div_pi");
    auto const_0_044715 = std::make_shared<Value>(0.044715f, "c_0.044715");


    for (const auto& x_val_ptr : input->data) {
        if (!x_val_ptr) {
            throw std::runtime_error("Null Value pointer in GELU input tensor.");
        }

        // x_cubed = x^3
        auto x_cubed = x_val_ptr->pow(3.0f);
        x_cubed->op = "gelu_xcubed";

        // 0.044715 * x^3
        auto term_coeff_mul_xcubed = const_0_044715 * x_cubed;
        term_coeff_mul_xcubed->op = "gelu_coeff_mul_xcubed";

        // x + (0.044715 * x^3)
        auto inner_sum = x_val_ptr + term_coeff_mul_xcubed;
        inner_sum->op = "gelu_inner_sum";

        // sqrt(2/pi) * inner_sum
        auto tanh_arg = const_sqrt_2_div_pi * inner_sum;
        tanh_arg->op = "gelu_tanh_arg";

        // tanh(...)
        auto tanh_out = tanh_arg->tanh();
        // tanh_out->op is already "tanh" from Value::tanh()

        // 1 + tanh(...)
        auto factor_sum = const_1 + tanh_out;
        factor_sum->op = "gelu_factor_sum";

        // 0.5 * x
        auto half_x = const_0_5 * x_val_ptr;
        half_x->op = "gelu_half_x";

        // result = (0.5 * x) * (1 + tanh(...))
        auto gelu_result = half_x * factor_sum;
        gelu_result->op = "gelu_final";

        output_values.push_back(gelu_result);
    }

    return Tensor::from_values(output_values, input->shape);
}

std::vector<std::shared_ptr<Value>> GELU::parameters() const {
    return {}; // GELU has no learnable parameters
}
