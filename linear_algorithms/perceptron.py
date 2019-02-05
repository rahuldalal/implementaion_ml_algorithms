from random import seed, randrange
from math import exp
from csv import reader
from copy import deepcopy, copy
import os

# Local imports
from data_preparation.load_dataset import load_dataset, str_column_to_float, str_column_to_int
from data_preparation.cross_validation_harness import evaluate_algorithm
from data_preparation.evaluation_metrics import accuracy_metric


# Perceptron functions
# Make predictions
def predict(row, weights, activation_function):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += row[i] * weights[i+1]
    return activation_function(activation)

# A perceptron can have only step and sigmoid as activation functions
# Step activation function
def step_activation(z):
    return 1.0 if z >= 0.0 else 0.0


# Sigmoid function
def sigmoid_activation(z):
    return 1/(1+exp(-z))


# Train Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch, activation_function):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0 # since we are using step activation we cannot use cross entropy loss to plot
        for row in train:
            yhat = predict(row, weights, activation_function)
            error = yhat - row[-1]
            weights[0] = weights[0] - l_rate * error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] - l_rate * error * row[i]
            sum_error += error**2
        print('> epoch = {}, sum_error {:.3f}'.format(epoch, sum_error))
    return weights


# Perceptron algorithm
def perceptron(train, test, l_rate, n_epoch, activation_function):
    weights = train_weights(train, l_rate, n_epoch, activation_function)
    predicted = list()
    for row in test:
        predicted.append(predict(row, weights, activation_function))
    return predicted


if __name__ == '__main__':

    # # test predictions
    # dataset = [[2.7810836,2.550537003,0],
    #   [1.465489372,2.362125076,0],
    #   [3.396561688,4.400293529,0],
    #   [1.38807019,1.850220317,0],
    #   [3.06407232,3.005305973,0],
    #   [7.627531214,2.759262235,1],
    #   [5.332441248,2.088626775,1],
    #   [6.922596716,1.77106367,1],
    #   [8.675418651,-0.242068655,1],
    #   [7.673756466,3.508563011,1]]
    # weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
    # for row in dataset:
    #   prediction = predict(row, weights, step_activation)
    #   print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    # # Test weights training
    # dataset = [[2.7810836,2.550537003,0],
    #   [1.465489372,2.362125076,0],
    #   [3.396561688,4.400293529,0],
    #   [1.38807019,1.850220317,0],
    #   [3.06407232,3.005305973,0],
    #   [7.627531214,2.759262235,1],
    #   [5.332441248,2.088626775,1],
    #   [6.922596716,1.77106367,1],
    #   [8.675418651,-0.242068655,1],
    #   [7.673756466,3.508563011,1]]
    # l_rate = 0.1
    # n_epoch = 5
    # weights = train_weights(dataset, l_rate, n_epoch, step_activation)
    # print(weights)


    # Perceptron algorithm to train on sonar dataset
    seed(1)
    dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    filename = 'sonar.all-data.csv'
    dataset = load_dataset(os.path.join(dataset_base_path, filename))
    # Convert string to flat values
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # Convert categorical variables to integer values
    str_column_to_int(dataset,len(dataset[0])-1)
    # evaluate algorithm
    n_folds = 3
    l_rate = 0.01
    n_epoch = 500
    scores = evaluate_algorithm(dataset, perceptron, n_folds, accuracy_metric, l_rate, n_epoch, step_activation)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
