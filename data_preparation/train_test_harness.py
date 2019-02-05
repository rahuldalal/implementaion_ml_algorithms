from csv import reader

from random import seed, randrange
from copy import deepcopy, copy
import os

# Local imports
from data_preparation.load_dataset import load_dataset, str_column_to_float
from data_preparation.evaluation_metrics import accuracy_metric
from data_preparation.resampling_methods import train_test_split


# Evaluate an algorithm using a train/test split harness
def evaluate_algorithm(dataset, algorithm, split, metric, *args):
    train, test = train_test_split(dataset, split)
    # Make a copy of test set to remove predictions while passing to the algorithm
    test_set = deepcopy(test)
    for row in test_set:
        row[-1] = None
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    accuracy = metric(actual, predicted)
    return accuracy


# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for _ in range(len(test))]
    return predicted


if __name__ == '__main__':
    # Test the train/test harness
    seed(1)
    dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    filename = 'pima-indians-diabetes.csv'
    dataset = load_dataset(os.path.join(dataset_base_path, filename))
    print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
    for column in range(len(dataset[0])):
        str_column_to_float(dataset, column)
    accuracy = evaluate_algorithm(dataset, zero_rule_algorithm_classification, 0.6, accuracy_metric)
    print('Accuracy {:.3f}'.format(accuracy))