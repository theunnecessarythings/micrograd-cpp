#ifndef VALUE_H
#define VALUE_H

#include <string>
#include <vector>
#include <functional>
#include <set>
#include <memory> // For std::shared_ptr

class Value; // Forward declaration

// Define a type for a function that computes backward pass
// It takes no arguments and returns void.
// This will be stored in each Value object to know how to compute its gradient.
using GradFunc = std::function<void()>;

class Value : public std::enable_shared_from_this<Value> {
public:
    float data;
    float grad;
    std::string op; // Operation that created this Value
    std::set<std::shared_ptr<Value>> _prev; // Operands
    GradFunc _backward; // Function to compute gradients for operands

    // Constructors
    Value(float data, std::string op = "", std::set<std::shared_ptr<Value>> _prev = {});
    Value(const Value& other); // Copy constructor
    Value& operator=(const Value& other); // Copy assignment

    // Overload operators
    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);
    // Power operator - useful for some activation functions or losses
    std::shared_ptr<Value> pow(float exponent);


    // Activation functions
    std::shared_ptr<Value> relu();
    std::shared_ptr<Value> sigmoid();
    std::shared_ptr<Value> exp();  // Exponential function
    std::shared_ptr<Value> log();  // Natural logarithm
    // Sigmoid and tanh can be added later if needed

    // Backward pass
    void backward();

private:
    void _perform_topological_sort(std::vector<std::shared_ptr<Value>>& sorted_nodes, std::set<std::shared_ptr<Value>>& visited_nodes);
};

// Free function operators for std::shared_ptr<Value>
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
// It might also be useful to have overloads for operations with floats, e.g.
// std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, float rhs);
// std::shared_ptr<Value> operator+(float lhs, const std::shared_ptr<Value>& rhs);
// etc. for other operators. For now, only Value-Value operations.

#endif // VALUE_H
