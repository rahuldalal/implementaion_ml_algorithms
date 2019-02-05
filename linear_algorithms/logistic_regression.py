from random import seed, randrange
from math import exp, log

from random import randrange, seed
import os

# Local imports
from data_preparation.load_dataset import load_dataset, str_column_to_float, dataset_minmax, normalize_dataset
from data_preparation.evaluation_metrics import accuracy_metric
from data_preparation.cross_validation_harness import evaluate_algorithm

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


