from random import randrange, seed
import os


from data_preparation.load_dataset import load_dataset, str_column_to_float, dataset_minmax, normalize_dataset
from data_preparation.cross_validation_harness import evaluate_algorithm
from data_preparation.evaluation_metrics import rmse_metric

# Make a prediction with coefficients
def predict(row, coefficents):
    yhat = coefficents[0]
    for i in range(len(row)-1): # The last column is dependent var
        yhat += row[i]*coefficents[i+1]
    return yhat


# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))] # Including constant b0
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


if __name__ == '__main__':
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
    metric = rmse_metric
    scores = evaluate_algorithm(dataset, multiple_linear_regression, n_folds, metric, l_rate, n_epoch)
    print('Scores: {}'.format(scores))
    print('Mean RMSE: {:.3f}'.format(sum(scores)/float(len(scores))))

