import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler


class knn:

    def __init__(self, k):
        """
        Object instantiation, save k and define a scaler object.
        :param self:
        :param k:
        :return:
        """
        self.k = k
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit scaler and save X_train and y_train.
        :param self:
        :param X_train:
        :param y_train:
        :return:
        """
        self.scaler.fit(X_train)
        self.X_train = X_train
        self.y_train = y_train

    @abstractmethod
    def predict(self, X_test):
        """
        Predict labels for X_test and return predicted labels.
        :param self: self.
        :param X_test: Test data.
        :return: Predicted labels.
        """

    def neighbours_indices(self, x):
        """
        For a given point x, find indices of k closest points in the training set.
        :param self: self.
        :param x: Vector describing point.
        :return: Indices of k closest points.
        """

        neighbors = np.zeros(self.k)
        for i in range(self.k):
            neighbors[i] = i

        for i in range(self.k+1, len(self.X_train)):
            temp_i = i
            for j in range(self.k):
                if self.dist(self.X_train[temp_i], x) < self.dist(self.X_train[j], x):
                    temp_j = neighbors[j]
                    neighbors[j] = temp_i
                    temp_i = temp_j
        return neighbors

    @staticmethod
    def dist(x1, x2):
        """
        Returns Euclidean distance between x1 and x2.
        :param x1:
        :param x2:
        :return:
        """
        return np.linalg.norm(x1-x2)

class RegressionKNN(knn):
    def __init__(self, k):
        """
        Object instantiation, parent class instantiation.
        :param k:
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        Predict labels for X_test and return predicted labels.
        :param X_test: Test data.
        :return: Predicted labels.
        """


class ClassificationKNN(knn):
    def __init__(self, k):
        """
        Object instantiation, parent class instantiation.
        :param k: How many neighbors
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        Predict labels for X_test and return predicted labels.
        :param X_test: Test data.
        :return: Predicted labels
        """
