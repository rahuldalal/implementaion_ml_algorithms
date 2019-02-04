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


def train_test_split(dataset, split=0.6):
    train = list()
    train_size = int(split*(len(dataset)))
    dataset_copy = copy(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        # Ensure an element is sampled only once
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a train/test split harness
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    # Make a copy of test set to remove predictions while passing to the algorithm
    test_set = deepcopy(test)
    for row in test_set:
        row[-1] = None
    predicted = algorithm(train, test_set, *args)
    print(predicted)
    actual = [row[-1] for row in test]
    print(actual)
    accuracy = accuracy_metric(actual, predicted)
    return accuracy


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
accuracy = evaluate_algorithm(dataset, zero_rule_algorithm_classification, 0.6)
print('Accuracy {:.3f}'.format(accuracy))