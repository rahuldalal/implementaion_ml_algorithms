from csv import reader

from random import seed, randrange
from copy import deepcopy, copy
import os

from data_preparation.load_dataset import load_dataset, str_column_to_float
from data_preparation.resampling_methods import cross_validation_split
from data_preparation.evaluation_metrics import accuracy_metric


# Evaluate an algorithm using a cross validation harness
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


# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for _ in range(len(test))]
    return predicted


# Test the train/test harness
if __name__ == '__main__':
    seed(1)
    dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    filename = 'pima-indians-diabetes.csv'
    dataset = load_dataset(os.path.join(dataset_base_path, filename))
    print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
    for column in range(len(dataset[0])):
        str_column_to_float(dataset, column)
    scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, 5, accuracy_metric)
    print('Scores: {}'.format(scores))
    print('Mean Accuracy {:.3f}'.format(sum(scores)/len(scores)))