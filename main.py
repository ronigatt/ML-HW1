import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.neighbors
from matplotlib.colors import ListedColormap

from classification_rules import ClassificationRules
from k_nearest_neighbors import KNearestNeighbors
from weigthed_k_nearest_neighbors import WeightedKNearestNeighbors

if __name__ == '__main__':
    dataset_dir = "https://raw.githubusercontent.com/probml/pmtkdata/master/knnClassify3c/"
    train_dataset_path = dataset_dir + "knnClassify3cTrain.txt"
    test_dataset_path = dataset_dir + "knnClassify3cTest.txt"

    # Train dataset
    train_dataset = pd.read_csv(train_dataset_path, names=["x1", "x2", "class"], delimiter=" ")
    X_train = train_dataset.iloc[:, :-1].values
    y_train = train_dataset.iloc[:, -1].values

    # Test dataset
    test_dataset = pd.read_csv(test_dataset_path, names=["x1", "x2", "class"], delimiter=" ")
    X_test = test_dataset.iloc[:, :-1].values
    y_test = test_dataset.iloc[:, -1].values

    # Question 1 #

    # create pairs plot
    df = pd.DataFrame(X_train, columns=["x1", "x2"])
    plt.figure()
    pd.plotting.scatter_matrix(df, alpha=0.7, c=y_train, cmap=ListedColormap([(1, 102/255, 1), (153/255, 0, 204/255), (0, 204/255, 1)]))
    plt.suptitle('Pairs plot', fontsize=16)

    # create whisker plot
    ncols = len(train_dataset.columns) - 1
    fig_1b, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(9.6, 4.8))
    for i, feature in enumerate(train_dataset.columns):
        if i < ncols:
            sns.boxplot(x="class", y=feature, data=train_dataset, ax=ax[i], palette=[(1, 102/255, 1), (153/255, 0, 204/255), (0, 204/255, 1)])
    fig_1b.suptitle('Whisker plot', fontsize=16)

    plt.show()

    # Question 3 #

    # Apply the rule based classifier in (2) to predict the test set
    classification_result = np.zeros(X_test.shape[0], dtype=np.int)
    for index, row in enumerate(X_test):
        classification_result[index] = ClassificationRules.predict(row[0], row[1])

    # Report the misclassification rate
    misclassification_rate = np.mean(classification_result != y_test)

    # Question 4 #

    knn_misclassification_rate = np.zeros(3)
    for index, k in enumerate([1,5,10]):
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        knn_result = knn.predict(X_test)
        knn_misclassification_rate[index] = np.mean(knn_result != y_test)
    plt.figure()
    plt.plot(1 / np.array([1, 5, 10]), knn_misclassification_rate, "-bo")
    plt.suptitle("Misclassification rate v.s. 1/k of KNN")
    plt.show()
    print(knn_misclassification_rate)

    # Question 5 #

    k_nearest_neighbors = KNearestNeighbors(1)
    k_nearest_neighbors.fit(X_train, y_train)
    knn_result = k_nearest_neighbors.predict(X_test)
    knn_naive_misclassification_rate = np.mean(knn_result != y_test)
    print(knn_naive_misclassification_rate)

    # Question 7 #
    w_knn_misclassification_rate = np.zeros(3)
    for index, k in enumerate([1, 5, 10]):
        knn = WeightedKNearestNeighbors(n_neighbors=k)
        knn.fit(X_train, y_train)
        knn_result = knn.predict(X_test)
        w_knn_misclassification_rate[index] = np.mean(knn_result != y_test)
    plt.figure()
    plt.plot(1 / np.array([1, 5, 10]), w_knn_misclassification_rate, "-bo")
    plt.suptitle("Misclassification rate v.s. 1/k of weighted KNN")
    plt.show()
    print(w_knn_misclassification_rate)
