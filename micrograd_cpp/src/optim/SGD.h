#ifndef SGD_H
#define SGD_H

#include "optim/Optimizer.h" // Base class
#include "core/Value.h"      // For std::shared_ptr<Value>
#include <vector>
#include <memory> // For std::shared_ptr

class SGD : public Optimizer {
public:
    float learning_rate;

    // Constructor
    SGD(const std::vector<std::shared_ptr<Value>>& params, float lr);

    // Perform a single optimization step
    void step() override;

    // zero_grad() is inherited from Optimizer base class
};

#endif // SGD_H
