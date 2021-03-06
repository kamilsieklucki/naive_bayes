import numpy as np
import pandas as pd

test = np.column_stack([[2, 1, 9, 4], [4, 2, 7, 4], [2, 1, 10, 10], [1, 1, 7, 10], [4, 2, 4, 2]])
train = np.column_stack([[1, 10, 2, 10, 3, 2], [3, 3, 3, 9, 5, 3], [1, 2, 1, 7, 2, 1], [1, 1, 1, 1, 2, 1], [2, 2, 2, 4, 4, 4]])

X_train = train[:, 0:4]
y_train = train[:, 4]
X_test = test[:, 0:4]
y_test = test[:, 4]

# test iris
#Importing the Dataset
# url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
# dataset = pd.read_csv(url, names=names)
# #Preprocessing - The next step is to split our dataset into its attributes and labels.
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values
# #Train Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# names = ['target_names','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
#   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280-OD315 of diluted wines','Proline']
# dataset_wine = pd.read_csv(url, names=names)
#
# X = dataset_wine.iloc[:, :-1].values
# y = dataset_wine.iloc[:, 0].values
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


def prob_of_class(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    prob = counts / np.sum(counts)
    return (dict(zip(unique, prob)), dict(zip(unique, counts)))


def prob_of_args(X_train, X_test, y_train, y_test):
    import random

    prob, n = prob_of_class(y_train)

    rows = np.arange(X_test.shape[0])
    cols = np.arange(X_train.shape[1])

    wynik_final = []

    for row in rows:
        wynik = []
        for decision in prob:
            wynik_czesciowy = []
            index = y_train == decision
            for col in cols:
                df = X_train[index, col] == X_test[row, col]
                wynik_czesciowy.append(np.sum(df))
            wynik.append(wynik_czesciowy)

        wynik = np.array(wynik)
        for j in cols:
            if np.any(wynik[:, j]) == 0:
                wynik[wynik[:, j] != 0, j] += 1

        tmp = {}
        i = 0
        for c in prob:
            tmp[c] = ((np.sum(wynik[i, :]) / n[c]) * prob[c])
            i += 1

        if np.all(list(tmp.values()) == list(tmp.values())[0]):
            x = random.sample(list(tmp.keys()), 1)
            wynik_final.append(x[0])
        else:
            x = max(tmp, key=tmp.get)
            wynik_final.append(x)

    return np.array(wynik_final)


y_pred = prob_of_args(X_train, X_test, y_train, y_test)
# print(y_pred)
# print(y_test)

def cm(y_pred, y_test):
    unique = np.unique(y_test)
    wynik = np.zeros((len(unique), len(unique)))
    classes = len(unique)
    length = len(y_test)

    for i in range(len(unique)):
        for j in range(len(unique)):
            wynik[j, i] = np.sum((y_pred == unique[i]) & (y_test == unique[j]))

    print(wynik)

    acc = []
    acc_il = []
    for k in range(classes):
        licznik = wynik[k, k]
        mianownik = np.sum(wynik[:, k])
        y = 0 if mianownik == 0 else licznik / mianownik

        print("acc" + str(unique[k]), round(y, 2))

        acc.append(y)
        acc_il.append(wynik[k, k])

    print("acc_global", round(sum(acc_il) / length, 2))
    print("acc_balanced", round(sum(acc) / classes, 2))

    tpr = []
    for k in range(classes):
        licznik = wynik[k, k]
        mianownik = np.sum(wynik[k, :])
        y = 0 if mianownik == 0 else licznik / mianownik

        print("tpr" + str(unique[k]), round(y, 2))

        tpr.append(y)

    print("coverage", np.sum(wynik) / length)




# confusion matrix
print("Confusion Matrix (my implementation)")
cm(y_pred, y_test)


#Evaluating the Algorithm
print("Evaluating the Algorithm sklearn")
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


