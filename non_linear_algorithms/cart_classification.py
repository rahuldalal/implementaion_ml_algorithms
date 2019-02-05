# Example of CART on the Banknote dataset
from random import seed, randrange
from csv import reader
from copy import copy, deepcopy
import os

# Local imports
from data_preparation.load_dataset import load_dataset, str_column_to_float
from data_preparation.evaluation_metrics import accuracy_metric
from data_preparation.cross_validation_harness import evaluate_algorithm


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # Count total number of data instances at the split point
    num_instances = sum([len(group) for group in groups])
    gini_value = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        # Get the class labels for each data point in the group
        class_labels_for_group = [row[-1] for row in group]
        for class_value in classes:
            # Calculate probability for each class in the group
            p = class_labels_for_group.count(class_value)/size
            # Simplified Gini formula: 1 - (p1**2+p2**2...). Score is calculating second term
            score += p**2
        # Weight the Gini index for the group
        group_weighted_score = (1-score)*(size/num_instances)
        gini_value += group_weighted_score
    return gini_value


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_groups, b_score = None, None, None, 999
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini_value = gini_index(groups, class_values)
            print('X{} = {} Gini={:.3f}'.format(index, row[index], gini_value))
            if gini_value < b_score:
                b_index, b_value, b_groups, b_score = index, row[index], groups, gini_value
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node.get('groups')
    del(node['groups'])
    # Check for no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# Build a decision tree
def build_tree(dataset, max_depth, min_size):
    root = get_split(dataset)
    split(root, max_depth, min_size, 1)
    return root


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('{}X{} < {:.3f}'.format(' '*depth, node.get('index'), node.get('value')))
        print_tree(node.get('left'), depth+1)
        print_tree(node.get('right'), depth+1)
    else:
        print('{}[{}]'.format(' '*depth, node))


# Make a prediction with a decision tree
def predict(node, row):
    if row[node.get('index')] < node.get('value'):
        if isinstance(node.get('left'), dict):
            return predict(node.get('left'), row)
        else:
            return node.get('left')
    else:
        if isinstance(node.get('right'), dict):
            return predict(node.get('right'), row)
        else:
            return node.get('right')


# Decision tree for classification
def decision_tree(train_set, test_set, max_depth, min_size):
    root = build_tree(train_set, max_depth, min_size)
    prediction = list()
    for row in test_set:
        prediction.append(predict(root, row))
    return prediction

if __name__ == '__main__':
    # # test Gini values
    # print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
    # print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

    # # Test getting the best split
    # dataset = [[2.771244718,1.784783929,0],
    #   [1.728571309,1.169761413,0],
    #   [3.678319846,2.81281357,0],
    #   [3.961043357,2.61995032,0],
    #   [2.999208922,2.209014212,0],
    #   [7.497545867,3.162953546,1],
    #   [9.00220326,3.339047188,1],
    #   [7.444542326,0.476683375,1],
    #   [10.12493903,3.234550982,1],
    #   [6.642287351,3.319983761,1]]
    # split = get_split(dataset)
    # print('Split: [X%d = %.3f]' % ((split['index']+1), split['value']))

    # Test building and printing a tree
    # dataset = [[2.771244718,1.784783929,0],
    #   [1.728571309,1.169761413,0],
    #   [3.678319846,2.81281357,0],
    #   [3.961043357,2.61995032,0],
    #   [2.999208922,2.209014212,0],
    #   [7.497545867,3.162953546,1],
    #   [9.00220326,3.339047188,1],
    #   [7.444542326,0.476683375,1],
    #   [10.12493903,3.234550982,1],
    #   [6.642287351,3.319983761,1]]
    # tree = build_tree(dataset, 3, 1)
    # print_tree(tree)

    # Test prediction with a decision tree
    # dataset = [[2.771244718,1.784783929,0],
    #   [1.728571309,1.169761413,0],
    #   [3.678319846,2.81281357,0],
    #   [3.961043357,2.61995032,0],
    #   [2.999208922,2.209014212,0],
    #   [7.497545867,3.162953546,1],
    #   [9.00220326,3.339047188,1],
    #   [7.444542326,0.476683375,1],
    #   [10.12493903,3.234550982,1],
    #   [6.642287351,3.319983761,1]]
    # stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
    # for row in dataset:
    #     prediction = predict(stump, row)
    #     print('Expected=%d, Got=%d' % (row[-1], prediction))

    # Load dataset
    seed(1)
    dataset_base_path =os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    filename = 'data_banknote_authentication.csv'
    dataset = load_dataset(os.path.join(dataset_base_path, filename))

    # Convert string to float for all such columns
    for column in range(len(dataset[0])):
        str_column_to_float(dataset, column)

    # evaluate algorithm
    n_folds = 5
    max_depth = 5
    min_size = 10
    scores = evaluate_algorithm(dataset, decision_tree, n_folds, accuracy_metric, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))