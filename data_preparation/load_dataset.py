from csv import reader
from math import sqrt
import os


# Load dataset
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


# Convert categorical data types in a column to integers
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    class_values_unique = set(class_values)
    lookup = dict()
    for index, val in enumerate(class_values_unique):
        lookup[val] = index
    for row in dataset:
        row[column] = lookup.get(row[column])
    return lookup # In case downstream users want to use the mapping


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


# Calculate column means
def column_means(dataset):
    col_means = [0 for _ in range(len(dataset[0]))]
    for col_index in range(len(dataset[0])):
        col_values = [row[col_index] for row in dataset]
        col_means[col_index] = sum(col_values)/len(col_values)
    return col_means


# Calculate column standard deviation
def column_stdevs(dataset, means):
    # to access stdevs by column index. For an empty list we cannot do that
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        term_variance = [pow(row[i] - means[i], 2) for row in dataset]
        stdevs[i] = sum(term_variance)
    stdevs = [sqrt(x/(len(dataset)-1)) for x in stdevs]
    return stdevs


# Standardize dataset
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i]-means[i])/stdevs[i]


# Small dataset to check normalization
# dataset = [[50, 30], [20, 90]]
# print(dataset)
# minmax = dataset_minmax(dataset)
# normalize_dataset(dataset, minmax)
# print(dataset)


# Small dataset to check standardization
# dataset = [[50, 30], [20, 90], [30, 50]]
# print(dataset)
# # Estimate mean and standard deviation
# means = column_means(dataset)
# stdevs = column_stdevs(dataset, means)
# print(means)
# print(stdevs)
# # standardize dataset
# standardize_dataset(dataset, means, stdevs)
# print(dataset)


# # Load pima diabetes dataset
# dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
# filename = 'pima-indians-diabetes.csv'
# dataset = load_dataset(os.path.join(dataset_base_path, filename))
# print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
# # print('pima-indians-diabete dataset string vals: \n{}'.format(dataset[0]))
# # Convert string to float for all such columns
# for column in range(len(dataset[0])):
#     str_column_to_float(dataset, column)
# print('pima-indians-diabete dataset float vals: \n{}'.format(dataset[0]))
# # Normalize pima diabetes dataset
# minmax = dataset_minmax(dataset)
# normalize_dataset(dataset, minmax)
# print('pima-indians-diabete dataset normalized: \n{}'.format(dataset[0]))
# # Standardize dataset
# # Estimate mean and standard deviation
# means = column_means(dataset)
# stdevs = column_stdevs(dataset, means)
# standardize_dataset(dataset, means, stdevs)
# print('pima-indians-diabete dataset standardized: \n{}'.format(dataset[0]))


# Load iris dataset
dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
filename = 'iris.csv'
dataset = load_dataset(os.path.join(dataset_base_path, filename))
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
# print('Iris dataset string vals: \n{}'.format(dataset[0]))
# Convert string to float for all such columns
for column in range(len(dataset[0])-1):
    str_column_to_float(dataset, column)
class_lookup = str_column_to_int(dataset, len(dataset[0])-1)
print('Iris dataset float vals + categorical (integer) encoding: \n{}'.format(dataset[0]))

