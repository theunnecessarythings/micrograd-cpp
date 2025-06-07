#ifndef ADAM_H
#define ADAM_H

#include "optim/Optimizer.h"
#include "core/Value.h"
#include <vector>
#include <memory> // For std::shared_ptr
#include <map>    // For storing m and v states per parameter

class Adam : public Optimizer {
public:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t; // Timestep

    // Using a map to store moments for each parameter.
    // The key is Value* as Value objects are assumed to persist and their addresses are stable
    // for the lifetime of parameters they represent.
    std::map<const Value*, float> m_data; // First moment vector (stores float data)
    std::map<const Value*, float> v_data; // Second moment vector (stores float data)

    // Constructor
    Adam(const std::vector<std::shared_ptr<Value>>& params,
         float lr = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f);

    void step() override;

    // zero_grad() is inherited
};

#endif // ADAM_H
