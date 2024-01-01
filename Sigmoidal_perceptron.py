# Sigmoidal Perceptron (Stochastic gradient descent)

# Author: Elmer Dema

# I have assumed bias to be 0.5

import numpy
from random import shuffle

ideal = {
    0: [[0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]],

    1: [[0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0]]
}

class SigmoidalPerceptron:

    MAX_ITERATIONS = 10
    TRAINING_CONSTANT = 0.05
    BATCH_SIZE = 5

    def __init__(self):
        self.weights = numpy.random.rand(21) # Random weights instead

    def R(self, vector_x, weights):
        scalar_product = numpy.dot(vector_x, weights)
        return 1 / (1 + numpy.exp(-scalar_product))

    def train(self, training):
        i = 0
        while i < self.MAX_ITERATIONS:
            delta = numpy.array([0.] * 21)
            old_weights = self.weights
            shuffle(training)
            counter = 0
            for j in range(len(training)):
                input_vec = numpy.insert(numpy.array(training[j][0]), 0, 1)
                r_value = self.R(input_vec, self.weights)
                delta += self.TRAINING_CONSTANT * (training[j][1] - r_value) * r_value * (1 - r_value) * input_vec
                counter += 1
                # This updates weights after a batch size of 5
                if counter % self.BATCH_SIZE == 0:
                    self.weights += delta
                    delta = numpy.array([0.] * 21) # Reset delta
            if numpy.array_equal(self.weights, old_weights):
                break
            i += 1

class ValuesGenerator:
    def __init__(self, ideal_dict):
        self.ideal_dict = ideal_dict
        self.training_set = []

    def flatten_matrix(self, matrix):
        return numpy.array([item for sublist in matrix for item in sublist])

    def normalize(self, array):
        return (array - numpy.mean(array)) / numpy.std(array)

    def generate_training_set(self, num_samples):
        training_set = []
        for i in self.ideal_dict.keys():
            for j in range(num_samples):
                new_digit = self.ideal_dict[i] + numpy.random.normal(0., 0.5, (5, 4))
                flattened_digit = self.flatten_matrix(new_digit)
                normalized_digit = self.normalize(flattened_digit)
                training_set.append((normalized_digit, i))
        self.training_set = training_set


if __name__ == "__main__":
    values_generator = ValuesGenerator(ideal)
    values_generator.generate_training_set(num_samples=100)

    perceptron = SigmoidalPerceptron()
    perceptron.train(values_generator.training_set)


    # The code henceforth is only for testing purposes

    values_generator_test = ValuesGenerator(ideal)
    values_generator_test.generate_training_set(num_samples=50)

    correct_predictions = 0
    total_predictions = 0

    for test_example in values_generator_test.training_set:
        input_vector = numpy.insert(numpy.array(test_example[0]), 0, 1)
        expected_output = test_example[1]

        prediction = perceptron.R(input_vector, perceptron.weights)

        predicted_output = round(prediction)

        if predicted_output == expected_output:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions

    print(f'The accuracy of the perceptron on the test set is {accuracy * 100}%')


    # Test with custom data
    print('\nTesting with custom data:')
    test_data = [
        [[1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]], # 0
        [[0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0]], # 1
        [[1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 1]], # 0
        [[1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1]], # 0
        [[0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]] # 1
    ]

    truth_data = [0, 1, 0, 0, 1]

    test_data = [values_generator.normalize(values_generator.flatten_matrix(test_example)) for test_example in test_data
                    if test_example not in values_generator.training_set]

    correct_predictions = 0
    total_predictions = 0

    for test_example, expected_output in zip(test_data, truth_data):
        input_vector = numpy.insert(numpy.array(test_example), 0, 1)
        prediction = perceptron.R(input_vector, perceptron.weights)
        
        predicted_output = round(prediction)

        print(f'Predicted output: {predicted_output}, Expected output: {expected_output}')

        if predicted_output == expected_output:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions

    print(f'The accuracy of the perceptron on the custom test set is {accuracy * 100}%')