import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """
    Run cross validation on X and y with specific model by given folds. Evaluate by given metric.
    :param model: A model object after instantiation.
    :param X: A matrix whose rows are observations and columns are features.
    :param y: An array the size of the observations, containing classification for each observation.
    :param folds: Output data.get_folds, an SKlearn KFold object.
    :param metric: Function that receives two arguments and returns a number.
    :return: A score for the cross validation process.
    """

    results = []
    for train_indices, validation_indices in folds.split(X):
        model.fit(X[train_indices], y[train_indices])
        model_train_predictions = model.predict(X[validation_indices])
        results.append(metric(y[validation_indices], model_train_predictions))
    return results


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    """
    run cross validation on X and y for every model induced by values from k_list by given folds.
    Evaluate each model by given metric.
    :param model:
    :param k_list:
    :param X:
    :param y:
    :param folds:
    :param metric:
    :return:
    """

    mean_results = []
    std_results = []

    for index in range(len(k_list)):
        model_obj = model(k_list[index])
        mean_results[index] = sum(cross_validation_score(model_obj, X, y, folds, metric))/len(folds)
        np_std = np.array(cross_validation_score(model_obj, X, y, folds, metric))
        std_results[index] = np.std(np_std)

    return mean_results, std_results
