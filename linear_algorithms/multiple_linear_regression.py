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
    print(len(dataset))
    fold_size = int(len(dataset)/k)
    print(fold_size)
    for i in range(k):
        fold_data = list()
        while len(fold_data) < fold_size:
            index = randrange(len(dataset_copy))
            fold_data.append(dataset_copy.pop(index))
        dataset_split.append(fold_data)
    return dataset_split


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for a, p in zip(actual, predicted):
        sum_error += pow((p-a), 2)
    return sqrt(sum_error / len(actual))


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


# Make a prediction with coefficients
def predict(row, coefficents):
    yhat = coefficents[0]
    for i in range(len(row)-1): # The last column is dependent var
        yhat += row[i]*coefficents[i+1]
    return yhat


# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))] # Including constant b0
    current_epoch, sum_error = 1, 0
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error**2  # Mean squared error cost function (not averaged across num_training_ex and ignored 1/2)
            coef[0] = coef[0]-l_rate*error
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1] - l_rate*error*row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef


# Multiple linear regression algo to suit the train test harness
def multiple_linear_regression(train_set, test_set, l_rate, n_epoch):
    coef = coefficients_sgd(train_set, l_rate, n_epoch)
    prediction = list()
    for row in test_set:
        prediction.append(predict(row, coef))
    return prediction


# # Test predict function
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# coef = [0.4, 0.8]
# for row in dataset:
#   yhat = predict(row, coef)
#   print("Expected={:.3f}, Predicted={:.3f}".format(row[-1], yhat))


# # Test multiple linear regression using SGD
# # Calculate coefficients
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# l_rate = 0.001
# n_epoch = 50
# coef = coefficients_sgd(dataset, l_rate, n_epoch)
# print(coef)


# Multiple linear regression on wine quality dataset
seed(1)
dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
filename = 'winequality-white.csv'
dataset = load_dataset(os.path.join(dataset_base_path, filename))
# Convert string to float
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# Normalize dataset
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# Train and test multiple linear regression on dataset using cross validation harness
n_folds = 5
l_rate = 0.01
n_epoch = 50
accuracy_metric = rmse_metric
scores = evaluate_algorithm(dataset, multiple_linear_regression, n_folds, accuracy_metric, l_rate, n_epoch)
print('Scores: {}'.format(scores))
print('Mean RMSE: {:.3f}'.format(sum(scores)/float(len(scores))))

