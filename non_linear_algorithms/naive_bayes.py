from collections import defaultdict
from math import sqrt, pi, exp
from random import seed
import os


# Local imports
from data_preparation.load_dataset import load_dataset, str_column_to_float, str_column_to_int
from data_preparation.cross_validation_harness import evaluate_algorithm
from data_preparation.evaluation_metrics import accuracy_metric


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = defaultdict(list)
    for row in dataset:
        separated[row[-1]].append(row)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / len(numbers)


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    mean_val = mean(numbers)
    variance = sum((num - mean_val) ** 2 for num in numbers) / (len(numbers) - 1)  # sample..hence divide by len(nums)-1
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])  # we don't need the statistics for the class label
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_label, class_dataset in separated.items():
        summaries[class_label] = summarize_dataset(class_dataset)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1/(sqrt(2*pi) * stdev)) * exponent


# Calculate the probabilities(not actually probabilites as denominator is ommited) of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum(summaries[class_label][0][2] for class_label in summaries)  # calculate total number of dta points
    probabilities = dict()
    for class_label, class_summaries in summaries.items():
        probabilities[class_label] = class_summaries[0][2]/total_rows
        for i in range(len(class_summaries)):
            col_mean, col_stdev, _ = class_summaries[i]
            probabilities[class_label] *= calculate_probability(row[i], col_mean, col_stdev)
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilites = calculate_class_probabilities(summaries, row)
    class_prediction = max(probabilites.keys(), key=probabilites.get)
    return class_prediction


# Naive Bayes Algorithm
def naive_bayes(train, test):
    train_summaries = summarize_by_class(train)
    predicted = list()
    for row in test:
        predicted.append(predict(train_summaries, row))
    return predicted

if __name__=='__main__':
    dataset = [[3.393533211, 2.331273381, 0],
               [3.110073483, 1.781539638, 0],
               [1.343808831, 3.368360954, 0],
               [3.582294042, 4.67917911, 0],
               [2.280362439, 2.866990263, 0],
               [7.423436942, 4.696522875, 1],
               [5.745051997, 3.533989803, 1],
               [9.172168622, 2.511101045, 1],
               [7.792783481, 3.424088941, 1],
               [7.939820817, 0.791637231, 1]]

    # # Test separating data by class
    # separated = separate_by_class(dataset)
    # for label in separated:
    #     print(label)
    #     for row in separated[label]:
    #         print(row)


    # # Test summarizing a dataset
    # summary = summarize_dataset(dataset)
    # print(summary)


    # # Test summarizing by class
    # summaries = summarize_by_class(dataset)
    # for class_label, class_summary in summaries.items():
    #     print('Class {}'.format(class_label))
    #     for col_summary in class_summary:
    #         print(col_summary)


    # # Test Gaussian PDF
    # print(calculate_probability(1.0, 1.0, 1.0))
    # print(calculate_probability(2.0, 1.0, 1.0))
    # print(calculate_probability(0.0, 1.0, 1.0))


    # # Test calculating class probabilities
    # summaries = summarize_by_class(dataset)
    # row = dataset[0]
    # probabilities = calculate_class_probabilities(summaries, row)
    # print(probabilities)
    # print(predict(summaries, row))


# Test Naive Bayes on Iris Dataset
    seed(1)

    dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    filename = 'iris.csv'
    dataset = load_dataset(os.path.join(dataset_base_path, filename))
    # Convert string to float
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)

    # evaluate algorithm
    n_folds = 5
    scores = evaluate_algorithm(dataset, naive_bayes, 5, accuracy_metric)
    print('Scores: {}'.format(scores))
    print('Mean Accuracy: {:.3f}'.format(sum(scores)/float(len(scores))))

