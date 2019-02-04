from csv import reader
from copy import deepcopy, copy
from random import randrange, seed
from math import sqrt
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


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for a, p in zip(actual, predicted):
        sum_error += pow((p-a), 2)
    return sqrt(sum_error/len(actual))


# Train test split
def train_test_split(dataset, split=0.6):
    train = list()
    train_size = int(split*(len(dataset)))
    dataset_copy = copy(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        # Ensure an element is sampled only once
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# Linear regression related functions
# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values)/float(len(values))


# Calculate the variance of a list of numbers (not averaged)
def variance(values, mean):
    return sum((val - mean)**2 for val in values)


# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0
    for xi, yj in zip(x, y):
        covar += (xi-mean_x)*(yj-mean_y)
    return covar


# Calculate coefficients
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    mean_x, mean_y = mean(x), mean(y)
    covar = covariance(x, mean_x, y, mean_y)
    var_x = variance(x, mean_x)
    b1 = covar/ var_x
    b0 = mean_y - b1*mean_x
    return [b0, b1]


def simple_linear_regression(train, test):
    b0, b1 = coefficients(train)
    predicted = [b0+b1*row[0] for row in test]
    return predicted


# Evaluate an algorithm using a train/test split harness
def evaluate_algorithm(dataset, algorithm, split, metric,  *args):
    train, test = train_test_split(dataset, split)
    # Make a copy of test set to remove predictions while passing to the algorithm
    test_set = deepcopy(test)
    for row in test_set:
        row[-1] = None
    predicted = algorithm(train, test_set, *args)
    print('Predictions:\n{}'.format(predicted))
    actual = [row[-1] for row in test]
    print('Actual:\n{}'.format(actual))
    metric_val = metric(actual, predicted)
    return metric_val


# # Test mean and variance on small contrived dataset
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# x = [row[0] for row in dataset]
# y = [row[1] for row in dataset]
# mean_x, mean_y = mean(x), mean(y)
# var_x, var_y = variance(x, mean_x), variance(y, mean_y)
# covar = covariance(x, mean_x, y, mean_y)
# print('x stats: mean={:.3f} variance={:.3f}'.format(mean_x, var_x))
# print('y stats: mean={:.3f} variance={:.3f}'.format(mean_y, var_y))
# print('Covariance: {:.3f}'.format(covar))
#
# # Test calculate coefficients
# b0, b1 = coefficients(dataset)
# print('Coefficients: b0={:.3f}, b1={:.3f}'.format(b0, b1))

# Load sweedish insurance dataset
seed(1)
dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
filename = 'insurance.csv'
dataset = load_dataset(os.path.join(dataset_base_path, filename))
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
# Convert string to float for all such columns
for column in range(len(dataset[0])):
    str_column_to_float(dataset, column)

rmse = evaluate_algorithm(dataset, simple_linear_regression, 0.6, rmse_metric)
print('RMSE {:.3f}'.format(rmse))
