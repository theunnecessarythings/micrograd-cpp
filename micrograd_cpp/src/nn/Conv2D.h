#ifndef CONV2D_H
#define CONV2D_H

#include "nn/Layer.h"
#include "core/Tensor.h"
#include <vector>
#include <memory> // For std::shared_ptr
#include <utility> // For std::pair for kernel_size, stride, padding
#include <string>  // For std::string

class Conv2D : public Layer {
public:
    int in_channels;
    int out_channels;
    std::pair<int, int> kernel_size; // height, width
    std::pair<int, int> stride;      // height, width
    std::pair<int, int> padding;     // height, width
    bool use_bias;

    std::shared_ptr<Tensor> weights; // Shape: [out_channels, in_channels, kernel_height, kernel_width]
    std::shared_ptr<Tensor> biases;  // Shape: [out_channels] or nullptr

    // Constructor
    Conv2D(int in_channels, int out_channels,
           std::pair<int, int> kernel_size,
           std::pair<int, int> stride = {1, 1},
           std::pair<int, int> padding = {0, 0},
           bool use_bias = true);

    // Overload for single int kernel_size, stride, padding
    Conv2D(int in_channels, int out_channels,
           int kernel_size_val,
           int stride_val = 1,
           int padding_val = 0,
           bool use_bias = true);


    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::vector<std::shared_ptr<Value>> parameters() const override;
    std::string name() const override { return "Conv2D"; }

private:
    void initialize_parameters();
};

#endif // CONV2D_H
