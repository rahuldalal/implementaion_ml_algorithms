from math import sqrt

# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
  correct = 0
  for i in range(len(actual)):
    if actual[i] == predicted[i]:
     correct += 1
  return correct / float(len(actual)) * 100.0


# Calculate confusion matrix
def confusion_matrix(actual, predicted):
    unique_classes = set(actual)
    matrix = [list() for _ in range(len(unique_classes))]
    for i in range(len(unique_classes)):
        matrix[i] = [0 for _ in range(len(unique_classes))]
    lookup = {val: index for index, val in enumerate(unique_classes)}
    # print(lookup)
    for a, p in zip(actual, predicted):
        i = lookup.get(p)  # predictions are row wise
        j = lookup.get(a)  # actual values are column wise
        matrix[i][j] += 1
    return unique_classes, matrix, lookup

def print_confusion_matrix(unique, matrix):
    print('(A) {}'.format('   '.join(str(x) for x in unique)))
    print('(P) {}'.format('---'*len(unique)))
    for index, val in enumerate(unique):
        print('{} | {}'.format(val, matrix[index]))


# Calculate mean absolute error
def mae_metric(actual, predicted):
    sum_error = 0.0
    for a, p in zip(actual, predicted):
        sum_error += abs(p-a)
    return sum_error/len(actual)


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for a, p in zip(actual, predicted):
        sum_error += pow((p-a), 2)
    return sqrt(sum_error / len(actual))

# # Test confusion matrix with integers
# actual = ['a','a','a','a','a','b','b','b','b','b']
# predicted = ['a','b','b','a','a','b','a','b','b','b']
# unique, matrix, lookup = confusion_matrix(actual, predicted)
# print(lookup, unique)
# # print(unique)
# # print(matrix)
# print_confusion_matrix(unique, matrix)


# # Test MAE
# actual = [0.1, 0.2, 0.3, 0.4, 0.5]
# predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
# mae = mae_metric(actual, predicted)
# print(mae)

# Test RMSE
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
rmse = rmse_metric(actual, predicted)
print(rmse)

