import numpy as np
import pandas as pd

test = np.column_stack([[2, 1, 9, 4], [4, 2, 7, 4], [2, 1, 10, 10], [1, 1, 7, 10], [4, 2, 4, 2]])
train = np.column_stack([[1, 10, 2, 10, 3, 2], [3, 3, 3, 9, 5, 3], [1, 2, 1, 7, 2, 1], [1, 1, 1, 1, 2, 1], [2, 2, 2, 4, 4, 4]])

X_train = train[:, 0:4]
y_train = train[:, 4]
X_test = test[:, 0:4]
y_test = test[:, 4]

def prob_of_class(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    prob = counts / np.sum(counts)
    return dict(zip(unique, prob))


def prob_of_args(X_train, X_test, y_train, y_test):
    prob = prob_of_class(y_train)

    for row in np.arange(X_test.shape[0]):
        for col in np.arange(X_train.shape[1]):
            for decision in prob:
                ind = y_train == decision
                df = X_train[ind, col] == X_test[row, col]
                print(np.sum(df) / np.sum(ind))

# czy nie poprzestawiać elementów w pętlach? Jak zaimplementować wyjątki, gdzie w jednej z klas jest zero? -> słownik

prob_of_args(X_train, X_test, y_train, y_test)


