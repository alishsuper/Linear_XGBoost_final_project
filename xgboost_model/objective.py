import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

def reg_linear(y_true, y_pred):
    N = y_true.shape[0]

    grad = y_pred-y_true
    hess = np.ones(N)
    return grad, hess

def binary_logistic(y_true, y_pred):
    y_pred = _sigmoid(y_pred)

    grad = y_true - y_pred
    hess = y_pred * (1.0-y_pred)

    return grad, hess

def logloss_score(y_true, y_pred):
    return log_loss(y_true, y_pred)

def mse_score(y_true, y_pred):
    N = y_true.shape[0]
    return np.sum((y_true-y_pred)**2)/N

def rmse_score(y_true, y_pred):
    N = y_true.shape[0]
    return np.sqrt(np.sum((y_true-y_pred)**2)/N)

def _get_objective(loss="reg:linear"):
    objectives_names = {"reg:linear": reg_linear,
                        "binary:logistic": binary_logistic,
                        "rmse": rmse_score}
    if loss not in objectives_names.keys():
        print(loss not in objectives_names.keys())
        raise Exception("{} does not found".format(loss))

    return objectives_names[loss]

def _get_loss(loss="mse"):
    objectives_names = {"reg:linear": mse_score,
                        "mse": mse_score,
                        "rmse": rmse_score,
                        "auc": roc_auc_score,
                        "accuracy": accuracy_score,
                        "logloss": logloss_score}
    if loss not in objectives_names.keys():
        raise Exception("{} does not found".format(loss))

    return objectives_names[loss]

def _sigmoid(x):
    return 1.0 / (1 + np.exp(-x))