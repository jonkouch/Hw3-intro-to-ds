import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """
    Calculates f1_score of binary classification task with true
    labels y_true and predicted labels y_pred.
    :param y_true: True classification of labels in np array.
    :param y_pred: Predicted classification of labels in np array.
    :return: The f1_score.
    """
    true_positive = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_positive = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    false_negative = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    f1 = (2*recall*precision)/(recall+precision)
    return f1


def rmse(y_true, y_pred):
    """
    RMSE of regression task with true labels y_true and predicted labels y_pred.
    :param y_true: True classification of labels in np array.
    :param y_pred: Predicted classification of labels in np array.
    :return: The RMSE
    """
    sigma = 0
    sigma += ((y_true[i]-y_pred[i])**2 for i in range(len(y_pred)))
    return (sigma*(1/len(y_true)))**0.5


def visualize_results(k_list, scores, metric_name, title, path):
    """
    Plot a results graph of cross validation scores
    :param k_list: Value list for k.
    :param scores: A list of equal length to k_list, where each entry is the results average
                   of the cross validation process for the model with the appropriate k.
    :param metric_name: A string that receives "f1_score" or "RMSE".
    :param title: A string that receives "classification" or "regression".
    :param path: The path for the file in which the plot will be saved.
    :return: none
    """
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.plot(k_list, scores)
    plt.xlabel("k")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig("plot.pdf")
    plt.show()





