import numpy as np
# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    return x * (1 - x)
# Input and output
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
# Initialize weights and bias
np.random.seed(1)
w1 = np.random.rand(2, 2)   # input to hidden
b1 = np.random.rand(1, 2)
w2 = np.random.rand(2, 1)   # hidden to output
b2 = np.random.rand(1, 1)
# Train for 10000 epochs
for epoch in range(10000):
    # Forward
    h_input = np.dot(X, w1) + b1
    h_output = sigmoid(h_input)
    o_input = np.dot(h_output, w2) + b2
    output = sigmoid(o_input)
    # Backward
    error = y - output
    d_output = error * sigmoid_deriv(output)
    error_hidden = d_output.dot(w2.T)
    d_hidden = error_hidden * sigmoid_deriv(h_output)
    # Update weights and biases
    w2 += h_output.T.dot(d_output)
    b2 += np.sum(d_output, axis=0, keepdims=True)
    w1 += X.T.dot(d_hidden)
    b1 += np.sum(d_hidden, axis=0, keepdims=True)
# Final output
print("Final predictions:")
print(output.round(3))
