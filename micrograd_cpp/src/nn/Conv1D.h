#ifndef CONV1D_H
#define CONV1D_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory> // For std::shared_ptr
#include <string> // For std::string

class Conv1D : public Layer {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    bool use_bias;

    std::shared_ptr<Tensor> weights; // Shape: [out_channels, in_channels, kernel_size]
    std::shared_ptr<Tensor> biases;  // Shape: [out_channels] or nullptr

    // Constructor
    Conv1D(int in_channels, int out_channels,
           int kernel_size_val,
           int stride_val = 1,
           int padding_val = 0,
           bool use_bias_flag = true);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::vector<std::shared_ptr<Value>> parameters() const override;
    std::string name() const override { return "Conv1D"; }

private:
    void initialize_parameters();
};

#endif // CONV1D_H
