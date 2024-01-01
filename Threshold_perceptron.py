# Authors: 
# Elmer Dema


import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, train_iter=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.train_iter = train_iter

    def out(self, inputs):
        inputs = np.insert(inputs, 0, 1)
        summation = np.dot(inputs, self.weights)
        activation = 1 if summation > 0 else 0 # Threshold activation function
        return activation

    def train(self, training_data):
        for epoch in range(self.train_iter):
            for inputs, label in training_data:
                prediction = self.out(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * np.insert(inputs, 0, 1)

# Generating the dataset
def generate_dataset(ideal, noise_percentage=60, size=100):
    rng = np.random.default_rng(123)
    ideal = [(np.array(xs), y) for (xs, y) in ideal]
    
    dataset = []
    for i in range(size * (100 - noise_percentage) // 100):
        for (xs, y) in ideal:
            noise = rng.random(4)
            noise = np.where(noise > 0.5, 1, 0)
            xs = np.where(noise == 1, np.where(xs == 0, 1, 0), xs)
            dataset.append((xs / 1, y))

    for i in range(size * noise_percentage // 100):
        for (xs, y) in ideal:
            dataset.append((xs / 1, y))

    return dataset

# Testing the trained Perceptron on some inputs
def test_perceptron(perceptron, test_inputs, truth):
    for test_input, expected_output in zip(test_inputs, truth):
        prediction = perceptron.out(test_input)
        print(f"Input: {test_input} Prediction: {prediction} Expected: {expected_output}")

    # Calculating the accuracy of the perceptron
    correct = sum(1 for i in range(len(test_inputs)) if truth[i] == perceptron.out(test_inputs[i]))
    accuracy = correct / len(test_inputs)
    print(f"Accuracy: {accuracy}")

def main():
    # Creating the dataset for training the perceptron
    ideal = [
        ([[0,1,1,0],
          [1,0,0,1],
          [1,0,0,1],
          [0,1,1,0]], 0),
        ([[0,0,1,0],
          [0,1,1,0],
          [0,0,1,0],
          [0,1,1,1]], 1),
    ]

    # Generating the dataset
    dataset = generate_dataset(ideal, noise_percentage=60, size=100)

    # Training the perceptron
    input_size = 16
    perceptron = Perceptron(input_size)
    perceptron.train(dataset)

    print(f"Weights: {perceptron.weights}")

    # Testing the perceptron
    test_inputs = [
        np.array([[0, 0, 1, 0],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]).flatten(),
        np.array([[0, 0, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]]).flatten(),
        np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 1]]).flatten(),
        np.array([[0, 1, 1, 0],
                  [0, 0, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0]]).flatten(),
        np.array([[0, 1, 1, 1],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 1, 1, 0]]).flatten(),
        np.array([[0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 1, 0]]).flatten(),
    ]

    truth = [0, 1, 0, 1, 0, 1]

    test_perceptron(perceptron, test_inputs, truth)

if __name__ == "__main__":
    main()
