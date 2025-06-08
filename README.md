# MicroGrad C++

A C++ implementation of a small automatic differentiation library (autograd engine) and a neural network library, inspired by Andrej Karpathy's micrograd. This project aims to build out core components necessary for defining, training, and evaluating neural networks.

## Current Features

### 1. Core Autograd Engine (`src/core`)
*   **`Value` Class:** Represents a scalar value with support for automatic differentiation.
    *   Tracks operations to build a computation graph.
    *   Computes gradients using backpropagation (`backward()` method).
    *   Overloaded operators: `+`, `-`, `*`, `/`.
    *   Mathematical functions: `pow`, `exp`, `log`, `relu`, `sigmoid`.
*   **`Tensor` Class:** A multi-dimensional array holding `Value` objects.
    *   Supports basic tensor creation (zeros, ones, random, from_vector).
    *   Element-wise arithmetic operations and scalar operations.
    *   Matrix multiplication (`matmul`).
    *   Transpose (`transpose()`).
    *   Element-wise activation functions (`relu`, `sigmoid`, `exp_elem`, `log_elem`, `sqrt_elem`, `rsqrt_elem`).
    *   Statistical methods: `sum_all`, `mean_all`, `var_all`.
    *   `log_softmax()` for classification outputs.

### 2. Neural Network Layers (`src/nn`)
*   **Base `Layer` Class:** Abstract interface for all layers (`forward`, `parameters`, `zero_grad`).
*   **Implemented Layers:**
    *   `Linear`: Fully connected layer with optional bias.
    *   `ReLU`: Rectified Linear Unit activation.
    *   `Sigmoid`: Sigmoid activation.
    *   `Embedding`: Maps integer indices to dense vector representations.
    *   `LayerNorm`: Layer Normalization with learnable scale (gamma) and shift (beta).
    *   `RMSNorm`: Root Mean Square Normalization with a learnable scale (gamma).
*   **Loss Functions (`nn/losses.h`)**
    *   `MSELoss`: Mean Squared Error loss.
    *   `CrossEntropyLoss`: Cross-entropy loss (LogSoftmax + NLL).
*   **Metrics (`nn/metrics.h`)**
    *   `accuracy_score`: Computes classification accuracy.

### 3. Optimizers (`src/optim`)
*   **Base `Optimizer` Class:** Abstract interface for optimizers (`step`, `zero_grad`).
*   **Implemented Optimizers:**
    *   `SGD`: Stochastic Gradient Descent.
    *   `Adam`: Adaptive Moment Estimation optimizer.

### 4. Training Infrastructure (`src/train`)
*   **`train_step` Function:** Performs a single training step (forward pass, loss calculation, backward pass, optimizer step).
*   **`fit` Function:** A basic loop to train a model for a specified number of epochs.
*   **Evaluation Functions:**
    *   `evaluate_step`: Computes loss on an evaluation batch.
    *   `evaluate_average_loss`: Computes average loss over an evaluation dataset.

### 5. Testing (`tests`)
*   Comprehensive unit tests using Google Test.
*   Currently **55 tests** covering `Value`, `Tensor`, all implemented layers, loss functions, optimizers, and training utilities.

## Build Instructions

This project uses CMake for building.

1.  **Prerequisites:**
    *   A C++ compiler supporting C++17 (e.g., GCC, Clang, MSVC).
    *   CMake (version 3.10 or higher).
    *   Git (for cloning).

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/micrograd_cpp.git # Replace with actual URL if available
    cd micrograd_cpp
    ```

3.  **Configure CMake and Build:**
    ```bash
    cmake -B build -S .
    cmake --build build
    ```
    This will compile the libraries (`micrograd_core`, `micrograd_nn`, `micrograd_optim`, `micrograd_train`) and test executables into the `build` directory.

## Running Tests

Tests are built into the `build/tests` directory. You can run them using CTest from the `build` directory:

```bash
cd build
ctest
```

Or run individual test executables directly:
```bash
./tests/value_tests
./tests/tensor_tests
./tests/nn_layer_tests
./tests/losses_tests
./tests/optimizer_tests
./tests/trainer_tests
```

## Basic Usage Example (Conceptual)

```cpp
// main.cpp - (Illustrative - actual example would need to be part of the build)
#include "core/Tensor.h"
#include "nn/Layer.h" // For base Layer
#include "nn/Linear.h"
#include "nn/ReLU.h"
#include "nn/losses.h"
#include "optim/Adam.h"
#include "train/Trainer.h" // For fit()
#include <iostream>
#include <vector>
#include <memory> // For std::make_shared

// Define a simple model (e.g., a custom class inheriting from Layer)
class MyModel : public Layer {
public:
    std::shared_ptr<Linear> fc1;
    std::shared_ptr<ReLU> relu1;
    std::shared_ptr<Linear> fc2;

    MyModel(int in, int h, int out) {
        fc1 = std::make_shared<Linear>(in, h);
        relu1 = std::make_shared<ReLU>();
        fc2 = std::make_shared<Linear>(h, out);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        auto x1 = fc1->forward(x);
        auto x2 = relu1->forward(x1);
        return fc2->forward(x2);
    }

    std::vector<std::shared_ptr<Value>> parameters() const override {
        std::vector<std::shared_ptr<Value>> params;
        auto p1 = fc1->parameters();
        auto p2 = fc2->parameters();
        params.insert(params.end(), p1.begin(), p1.end());
        params.insert(params.end(), p2.begin(), p2.end());
        return params;
    }
    std::string name() const override { return "MyModel"; }
};


int main() {
    // 1. Model
    auto model = std::make_shared<MyModel>(10, 20, 1); // 10 input, 20 hidden, 1 output

    // 2. Data (dummy)
    std::vector<std::shared_ptr<Tensor>> X_train;
    std::vector<std::shared_ptr<Tensor>> Y_train;
    // Create a single batch for simplicity
    X_train.push_back(Tensor::randn({1, 10})); // Batch of 1, 10 features
    Y_train.push_back(Tensor::from_vector({0.5f}, {1,1}));

    // 3. Optimizer
    auto model_params = model->parameters();
    if (model_params.empty()) {
        std::cerr << "Error: Model has no parameters!" << std::endl;
        return 1;
    }
    auto optimizer = std::make_shared<Adam>(model_params, 0.001f);

    // 4. Loss function
    auto loss_fn = MSELoss::forward;

    // 5. Training
    std::cout << "Starting training..." << std::endl;
    try {
        fit(model, X_train, Y_train, loss_fn, optimizer, 5, 1); // 5 epochs, batch_size=1 (conceptual)
        std::cout << "Training finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        return 1;
    }

    // 6. Evaluation (conceptual - needs test data)
    // std::vector<std::shared_ptr<Tensor>> X_test;
    // std::vector<std::shared_ptr<Tensor>> Y_test;
    // X_test.push_back(Tensor::randn({1, 10}));
    // Y_test.push_back(Tensor::from_vector({0.8f}, {1,1}));
    // if (!X_test.empty()){
    //    auto test_loss = evaluate_average_loss(model, X_test, Y_test, loss_fn);
    //    std::cout << "Test Loss: " << test_loss << std::endl;
    // }

    return 0;
}
```
*(Note: The example `main.cpp` would need to be compiled and linked against the libraries. This is illustrative of how the components might be used. A placeholder git URL is used and should be updated.)*

## Future Work & Roadmap

The library is under active development. Planned features and areas for improvement include:

*   **More Neural Network Layers:**
    *   Convolutional Layers (Conv1D, Conv2D, Conv3D)
    *   Multi-Head Attention
    *   (Potentially others based on model requirements like GPT-2)
*   **Advanced Optimizers & Schedulers:**
    *   Learning Rate Schedulers
*   **Sample Architectures:**
    *   Implement a GPT-2 style model.
*   **Enhanced Data Handling:**
    *   `DataLoader` for more efficient batching, shuffling, and parallel data loading.
*   **Serialization:**
    *   Saving and loading model weights.
*   **Performance Optimizations:**
    *   Investigate potential bottlenecks and optimize critical operations (e.g., using BLAS libraries for matrix ops if `Tensor` is refactored to use raw float arrays).
*   **Broader Tensor Functionality:**
    *   More operations (reductions along axes, broadcasting enhancements, advanced indexing/slicing).

Contributions and feedback are welcome!
