from random import seed, randrange
from math import exp, log

from csv import reader
from math import sqrt
from random import randrange, seed
from copy import copy, deepcopy
import os


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


# Find the min and max values for given column
def dataset_minmax(dataset):
    minmax = list()
    for col in range(len(dataset[0])):
        col_values = [row[col] for row in dataset]
        col_min = min(col_values)
        col_max = max(col_values)
        minmax.append((col_min, col_max))
    return minmax


# Normalize dataset in place (Rescale values of each column between 0 -1)
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])


# Split a dataset into k-fold
def cross_validation_split(dataset, k=3):
    dataset_split = list()
    dataset_copy = copy(dataset)
    fold_size = int(len(dataset)/k)
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
def evaluate_algorithm(dataset, algorithm, n_folds, accuracy_metric, *args):
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
        scores.append(accuracy_metric(actual, predicted))
    return scores


# Make predictions
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1): # since last value in the row is the class label
        yhat += row[i]*coefficients[i+1]
    yhat = 1/(1+exp(-yhat))
    return yhat


# Test prediction with coefficients
def predict(row, coefficients):
  yhat = coefficients[0]
  for i in range(len(row)-1):
    yhat += coefficients[i + 1] * row[i]
  return 1.0 / (1.0 + exp(-yhat))


# Estimate linear regression coefficients using stochastic gradient descent
# JB has used mean squared error function we will use cross entropy
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        loss = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            loss += -(row[-1]*log(yhat)+(1-row[-1])*log(1-yhat))
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        print('>epoch={}, loss={:.3f}'.format(epoch, loss))
    return coef


# Multiple linear regression algo to suit the train test harness
def multiple_linear_regression(train_set, test_set, l_rate, n_epoch):
    coef = coefficients_sgd(train_set, l_rate, n_epoch)
    prediction = list()
    for row in test_set:
        prediction.append(round(predict(row, coef)))
    return prediction


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
# coef = [-0.406605464, 0.852573316, -1.104746259]
# for row in dataset:
#     yhat = predict(row, coef)
#     print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))


# # TestCalculate coefficients
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
# l_rate = 0.3
# n_epoch = 100
# coef = coefficients_sgd(dataset, l_rate, n_epoch)
# print(coef)


# Logistic regression for pima-indians-diabetes dataset
seed(1)
dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
filename = 'pima-indians-diabetes.csv'
dataset = load_dataset(os.path.join(dataset_base_path, filename))

# Convert string to float
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# Normalize dataset
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# Train and test multiple linear regression on dataset using cross validation harness
n_folds = 5
l_rate = 0.1
n_epoch = 100
accuracy_metric = accuracy_metric
scores = evaluate_algorithm(dataset, multiple_linear_regression, n_folds, accuracy_metric, l_rate, n_epoch)
print('Scores: {}'.format(scores))
print('Mean Accuracy: {:.3f}'.format(sum(scores)/float(len(scores))))


