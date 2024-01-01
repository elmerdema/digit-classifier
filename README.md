# Perceptron Implementations

Author: Elmer Dema
## Overview

This Python script provides implementations of three types of perceptrons: Sigmoidal Perceptron, Threshold Perceptron, and Linear Perceptron. Each perceptron is designed for binary classification and uses different activation functions.

# Linear Perceptron Implementation

In this Python script (`linear_perceptron.py`), a Linear Perceptron is implemented using Stochastic Gradient Descent (SGD) for binary classification.

## Initialization

1. **Weights Initialization:**
   - The perceptron is initialized with random weights. It uses 21 weights, including a bias term.

## Activation Function

2. **Activation Function (Linear):**
   - The activation function is linear. The perceptron computes the scalar product of the input vector and the weights.

3. **Prediction (Binary Output):**
   - Predictions are made by applying a step function (`numpy.heaviside`) to the scalar product. If positive, the perceptron predicts 1; otherwise, it predicts 0.

## Training (Stochastic Gradient Descent)

4. **Training Loop:**
   - The perceptron undergoes a training loop with a maximum number of iterations (`MAX_ITERATIONS`). The loop continues until convergence or the maximum iterations are reached.

5. **Shuffling Data:**
   - Training data is shuffled in each iteration to introduce randomness.

6. **Batch Update:**
   - The perceptron updates its weights after processing a batch of training examples (`BATCH_SIZE`).

7. **Weight Update:**
   - For each example in the batch, the perceptron updates its weights using the stochastic gradient descent update rule.

8. **Convergence Check:**
   - The training loop checks for convergence by comparing the new weights with the previous weights.

## Testing

9. **Testing Accuracy:**
   - After training, the perceptron is tested on a separate test set, and accuracy is calculated by comparing predicted and expected labels.

## Custom Test Data

10. **Custom Test Set:**
    - The perceptron is tested with a custom dataset not seen during training. The custom dataset includes five examples with corresponding ground truth labels.

11. **Accuracy Calculation:**
    - Accuracy on the custom test set is calculated by comparing predicted and expected labels.

## Output

12. **Print Accuracy:**
    - The script prints the accuracy of the perceptron on both the standard test set and the custom test set.

## Conclusion

This Linear Perceptron aims to learn a decision boundary for binary classification through iterative updates of weights using Stochastic Gradient Descent. The training process adjusts the perceptron's weights to minimize the difference between predicted and true labels for a given set of input features.


## Perceptron Types

### Sigmoidal Perceptron (`Sigmoidal_perceptron.py`):

- Implements a perceptron with a sigmoid activation function.
- Useful for problems where the output needs to be in the range (0, 1).

### Threshold Perceptron (`Threshold_perceptron.py`):

- Implements a perceptron with a step (threshold) activation function.
- Outputs binary values (0 or 1) based on a predefined threshold.



## Usage

### Sigmoidal Perceptron:

- Refer to `Sigmoidal_perceptron.py` for implementation details.
- Run the script to train and test the sigmoidal perceptron.

### Threshold Perceptron:

- Refer to `Threshold_perceptron.py` for implementation details.
- Run the script to train and test the threshold perceptron.



## Dependencies

- Each script relies on the `numpy` library for numerical operations.
- `ValuesGenerator` class is used to generate training and test datasets.

## Running the Code

1. Ensure you have Python installed along with the required dependencies.
2. Run the script for the desired perceptron type (`Sigmoidal_perceptron.py`, `Threshold_perceptron.py`, or `linear_perceptron.py`).
3. The script will output the accuracy of the perceptron on the test set.

## Custom Test Data

- Custom test data is included in each script to assess the perceptron's performance on unseen examples.
- The output includes predicted and expected labels for each custom test example.

