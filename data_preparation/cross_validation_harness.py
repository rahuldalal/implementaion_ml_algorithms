from csv import reader

from random import seed, randrange
from copy import deepcopy, copy
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
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
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


# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for _ in range(len(test))]
    return predicted


# Test the train/test harness
seed(1)
dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
filename = 'pima-indians-diabetes.csv'
dataset = load_dataset(os.path.join(dataset_base_path, filename))
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
for column in range(len(dataset[0])):
    str_column_to_float(dataset, column)
scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, 5)
print('Scores: {}'.format(scores))
print('Mean Accuracy {:.3f}'.format(sum(scores)/len(scores)))