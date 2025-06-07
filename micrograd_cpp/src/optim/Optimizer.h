#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "core/Value.h" // For std::shared_ptr<Value>
#include <vector>
#include <memory> // For std::shared_ptr

class Optimizer {
public:
    std::vector<std::shared_ptr<Value>> params;

    // Constructor: takes a vector of shared_ptr<Value> which are the parameters to optimize
    Optimizer(const std::vector<std::shared_ptr<Value>>& params) : params(params) {}

    // Virtual destructor for proper cleanup if Optimizer is subclassed
    virtual ~Optimizer() = default;

    // Pure virtual method for performing a single optimization step
    virtual void step() = 0;

    // Method to zero out the gradients of all parameters managed by this optimizer
    // This can be implemented in the base class as it's common.
    virtual void zero_grad() {
        for (const auto& p : params) {
            if (p) { // Ensure the parameter pointer is valid
                p->grad = 0.0f;
            }
        }
    }
};

#endif // OPTIMIZER_H
