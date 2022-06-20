import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(2)


def load_data(path):
    """
    Function receives path to data and return the data as data frame.
    :param path: The path to the data.
    :return: The data as data frame.
    """
    df = pd.read_csv(path)
    return df


def adjust_labels(season_np):
    """
    Function receives np array of values from season and adjusts its
    labels from {0,1,2,3} to {0,1}
    :param season_np: A np array of the season column from the dataset.
    :return: The adjusted np array.
    """
    season_copy = season_np
    season_copy[season_np == 1] = 0
    season_copy[season_np > 1] = 1
    return season_copy


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


class StandardScaler:
    def __init__(self):
        """
        object instantiation.
        """
        self.mean = None
        self.std_div = None

    def fit(self, X):
        """
        Fit scaler by learning mean and standard deviation per feature.
        :param X: np array of data.
        :return: none.
        """
        self.mean = np.average(X, axis=0)
        self.std_div = np.std(X, ddof=1)

    def transform(self, X):
        """
        Transform X by learned mean and standard deviation, and return it.
        :param X: np array of data.
        :return: Transformed X.
        """
        X = (X-self.mean)/self.std_div
        return X

    def fit_transform(self, X):
        """
        fit scaler by learning mean and standard deviation per feature, and then transform X.
        :param X: np array of data.
        :return: Transformed X.
        """
        self.fit()
        return self.transform(X)
