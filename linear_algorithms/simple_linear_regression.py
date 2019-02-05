from random import randrange, seed
import os


# Local imports
from data_preparation.load_dataset import load_dataset, str_column_to_float
from data_preparation.evaluation_metrics import rmse_metric
from data_preparation.train_test_harness import evaluate_algorithm


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


if __name__=='__main__':
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
