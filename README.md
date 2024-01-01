# Perceptron Implementations

Author: Elmer Dema
## Overview

This Python script provides implementations of three types of perceptrons: Sigmoidal Perceptron, Threshold Perceptron, and Linear Perceptron. Each perceptron is designed for binary classification and uses different activation functions.

## Perceptron Types

### Sigmoidal Perceptron (`Sigmoidal_perceptron.py`):

- Implements a perceptron with a sigmoid activation function.
- Useful for problems where the output needs to be in the range (0, 1).

### Threshold Perceptron (`Threshold_perceptron.py`):

- Implements a perceptron with a step (threshold) activation function.
- Outputs binary values (0 or 1) based on a predefined threshold.

### Linear Perceptron (`linear_perceptron.py`):

- Implements a perceptron using Stochastic Gradient Descent (SGD) with a linear activation function.
- Suitable for problems where the decision boundary is a hyperplane.

## Usage

### Sigmoidal Perceptron:

- Refer to `Sigmoidal_perceptron.py` for implementation details.
- Run the script to train and test the sigmoidal perceptron.

### Threshold Perceptron:

- Refer to `Threshold_perceptron.py` for implementation details.
- Run the script to train and test the threshold perceptron.

### Linear Perceptron:

- Refer to `linear_perceptron.py` for implementation details.
- Run the script to train and test the linear perceptron.

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

