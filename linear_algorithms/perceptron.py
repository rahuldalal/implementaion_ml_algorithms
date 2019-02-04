from random import seed, randrange
from math import exp
from csv import reader
from copy import deepcopy, copy
import os


# Load dataset
def load_dataset(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_file = reader(file)
        for row in csv_file:
            # remove empty rows
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert all a column from string to float (all rows). Does not return anything. Makes changes in place
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert categorical data types in a column to integers
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    class_values_unique = set(class_values)
    lookup = dict()
    for index, val in enumerate(class_values_unique):
        lookup[val] = index
    for row in dataset:
        row[column] = lookup.get(row[column])
    return lookup # In case downstream users want to use the mapping


# Split a dataset into k-fold
def cross_validation_split(dataset, k=3):
    dataset_split = list()
    dataset_copy = copy(dataset)
    # print(len(dataset))
    fold_size = int(len(dataset)/k)
    # print(fold_size)
    for i in range(k):
        fold_data = list()
        while len(fold_data) < fold_size:
            index = randrange(len(dataset_copy))
            fold_data.append(dataset_copy.pop(index))
        dataset_split.append(fold_data)
    return dataset_split


# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross avalidation harness
def evaluate_algorithm(dataset, algorithm, n_folds, metric, *args):
    folds = cross_validation_split(dataset, n_folds)
    # print(folds)
    scores = list()
    for fold in folds:
        train_set = list(folds) # shallow copy of the folds
        train_set.remove(fold)

        train_set = sum(train_set, [])
        # print(train_set)
        test_set = deepcopy(fold)
        for row in test_set:
            row[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        scores.append(metric(actual, predicted))
    return scores


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
