from random import seed, randrange
from copy import copy


def train_test_split(dataset, split=0.6):
    train = list()
    train_size = int(split*(len(dataset)))
    dataset_copy = copy(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        # Ensure an element is sampled only once
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# Split a dataset into k-fold
def cross_validation_split(dataset, k=3):
    dataset_split = list()
    dataset_copy = copy(dataset)
    print(len(dataset))
    fold_size = int(len(dataset)/k)
    print(fold_size)
    for i in range(k):
        fold_data = list()
        while len(fold_data) < fold_size:
            index = randrange(len(dataset_copy))
            fold_data.append(dataset_copy.pop(index))
        dataset_split.append(fold_data)
    return dataset_split

# # test train/test split
# seed(1)
# dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# train, test = train_test_split(dataset)
# print(train)
# print(test)


# test cross validation split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 4)
print(folds)


