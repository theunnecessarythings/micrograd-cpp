#include "core/Value.h" // Use quotes for local includes
#include <cmath>       // For std::pow, std::exp
#include <algorithm>   // For std::reverse
#include <stdexcept>   // For std::runtime_error

// Constructor
Value::Value(float data, std::string op, std::set<std::shared_ptr<Value>> _prev)
    : data(data), grad(0.0f), op(op), _prev(_prev), _backward([]() {}) {}

// Copy constructor
Value::Value(const Value& other)
    : data(other.data), grad(other.grad), op(other.op), _prev(other._prev), _backward(other._backward) {}

// Copy assignment operator
Value& Value::operator=(const Value& other) {
    if (this != &other) {
        data = other.data;
        grad = other.grad;
        op = other.op;
        _prev = other._prev;
        _backward = other._backward;
    }
    return *this;
}

// Operator overloading
std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(self->data + other->data, "+", std::set<std::shared_ptr<Value>>{self, other});
    out->_backward = [self, other, out]() {
        self->grad += out->grad;
        other->grad += out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::exp() {
    auto self = shared_from_this();
    float e_x = std::exp(self->data);
    auto out = std::make_shared<Value>(e_x, "exp", std::set<std::shared_ptr<Value>>{self});

    out->_backward = [self, out, e_x]() {
        // Gradient of exp(x) is exp(x)
        self->grad += e_x * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::log() {
    auto self = shared_from_this();
    if (self->data <= 0) {
        // Log is undefined for non-positive values.
        throw std::runtime_error("Logarithm undefined for non-positive value: " + std::to_string(self->data));
    }
    float log_x = std::log(self->data);
    auto out = std::make_shared<Value>(log_x, "log", std::set<std::shared_ptr<Value>>{self});

    out->_backward = [self, out]() {
        // Gradient of log(x) is 1/x
        float grad_val = (self->data == 0.0f) ? 0.0f : (1.0f / self->data); // Avoid div by zero strictly
        self->grad += grad_val * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(self->data - other->data, "-", std::set<std::shared_ptr<Value>>{self, other});
    out->_backward = [self, other, out]() {
        self->grad += out->grad;
        other->grad -= out->grad; // Corrected gradient for subtraction
    };
    return out;
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(self->data * other->data, "*", std::set<std::shared_ptr<Value>>{self, other});
    out->_backward = [self, other, out]() {
        self->grad += other->data * out->grad;
        other->grad += self->data * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
    auto self = shared_from_this();
    if (other->data == 0) {
        throw std::runtime_error("Division by zero");
    }
    auto out = std::make_shared<Value>(self->data / other->data, "/", std::set<std::shared_ptr<Value>>{self, other});
    out->_backward = [self, other, out]() {
        self->grad += (1.0f / other->data) * out->grad;
        other->grad += (-self->data / (other->data * other->data)) * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::pow(float exponent) {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(std::pow(self->data, exponent), "pow", std::set<std::shared_ptr<Value>>{self});
    out->_backward = [self, exponent, out]() {
        self->grad += (exponent * std::pow(self->data, exponent - 1)) * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::relu() {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(std::max(0.0f, self->data), "ReLU", std::set<std::shared_ptr<Value>>{self});
    out->_backward = [self, out]() {
        if (self->data > 0) {
            self->grad += out->grad;
        }
    };
    return out;
}

// Backward pass implementation
void Value::_perform_topological_sort(std::vector<std::shared_ptr<Value>>& sorted_nodes, std::set<std::shared_ptr<Value>>& visited_nodes) {
    visited_nodes.insert(shared_from_this());
    for (const auto& prev_node : _prev) {
        if (visited_nodes.find(prev_node) == visited_nodes.end()) {
            prev_node->_perform_topological_sort(sorted_nodes, visited_nodes);
        }
    }
    sorted_nodes.push_back(shared_from_this());
}

void Value::backward() {
    std::vector<std::shared_ptr<Value>> sorted_nodes;
    std::set<std::shared_ptr<Value>> visited_nodes;

    // Perform topological sort
    _perform_topological_sort(sorted_nodes, visited_nodes);
    std::reverse(sorted_nodes.begin(), sorted_nodes.end());

    // Initialize gradient of the output node to 1
    this->grad = 1.0f;

    // Call _backward for each node in reverse topological order
    for (const auto& node : sorted_nodes) {
        node->_backward();
    }
}

// Implementations for free function operators
// (exp and log should be before this block)
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return lhs->operator+(rhs);
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return lhs->operator-(rhs);
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return lhs->operator*(rhs);
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return lhs->operator/(rhs);
}

std::shared_ptr<Value> Value::sigmoid() {
    auto self = shared_from_this();
    float x = self->data;
    // Compute s = 1.0f / (1.0f + std::exp(-x)) robustly
    // If x is very large, exp(-x) -> 0, s -> 1
    // If x is very small (large negative), exp(-x) -> infinity, s -> 0
    // If x is near zero, s -> 0.5
    float s;
    if (x >= 0) { // More numerically stable for large positive x
        s = 1.0f / (1.0f + std::exp(-x));
    } else { // More numerically stable for large negative x (exp(x) is small)
        float exp_x = std::exp(x);
        s = exp_x / (1.0f + exp_x);
    }

    auto out = std::make_shared<Value>(s, "sigmoid", std::set<std::shared_ptr<Value>>{self});

    out->_backward = [self, out, s]() {
        // Gradient of sigmoid(x) is s * (1 - s)
        self->grad += (s * (1.0f - s)) * out->grad;
    };
    return out;
}
