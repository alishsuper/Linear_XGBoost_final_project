import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import linearXGB as lxgb
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from objective import _get_loss

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def plot(params, X, y, n_estimators, verbose=10, test_size=.3):
    if test_size <= .0 or test_size >= 1.:
        raise Exception("folds should be in range (0, 1)")

    if X.shape[1] != 1:
        raise Exception("The X dataset should be 1-dimensinal")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    model = lxgb.linearXBG(params, n_estimators, verbose=verbose, eval_sets=[([X_test, y_test], "test")])
    model.fit(X_train, y_train)

    train_losses = calculate_losses(model, X_train, y_train)
    test_losses = calculate_losses(model, X_test, y_test)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax1.plot(X, model.predict(X), marker=".")
    ax1.scatter(X, y, c="r", marker=".")
    ax2.plot(train_losses[0], train_losses[1], c="r")
    ax2.plot(test_losses[0], test_losses[1], c="b")
    ax1.set_title("Approximation by linear XGBoost on test")
    ax2.set_title("Losses on train and test")
    ax1.legend(["Original test function", "Approximation on test"])
    ax2.legend(["Train loss", "Test loss"])

    plt.show()


def calculate_losses(model, X, y, metric="rmse"):
    eval_metric = _get_loss(metric)
    losses = []
    for i in range(1, model.n_estimators):
        losses += [eval_metric(y, model.predict(X, ntree_limit=i))]

    return np.arange(1, model.n_estimators), losses

def params_text(params):
    text = ""
    for pk, pv in params.items():
        if pv == str(pv):
            continue
        if (float(pv)==pv and pv > 0):
            text += "{} : {}\n".format(pk, pv)
    return text


def compare_plot(params, X, y, n_estimators, verbose=10, test_size=.3):
    if test_size <= .0 or test_size >= 1.:
        raise Exception("folds should be in range (0, 1)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    linearXGB = lxgb.linearXBG(params, n_estimators, verbose=verbose, eval_sets=[([X_test, y_test], "test")])
    linearXGB.fit(X_train, y_train)

    originalXGB = xgb.XGBRegressor(n_estimators=n_estimators,
                                   max_depth=params["max_depth"],
                                   nthread=-1,
                                   learning_rate=params["learning_rate"],
                                   reg_alpha=params["alpha"],
                                   reg_lambda=params["lambda"])
    originalXGB.fit(X_train, y_train)


    linear_train_losses = calculate_losses(linearXGB, X_train, y_train)
    linear_test_losses = calculate_losses(linearXGB, X_test, y_test)

    original_train_losses = calculate_losses(originalXGB, X_train, y_train)
    original_test_losses = calculate_losses(originalXGB, X_test, y_test)


    plt.plot(linear_train_losses[0], linear_train_losses[1], c='r')
    plt.plot(linear_test_losses[0], linear_test_losses[1], c='b')
    plt.plot(original_train_losses[0], original_train_losses[1], c='g')
    plt.plot(original_test_losses[0], original_test_losses[1], c='y')

    plt.title("Losses of original and linear XGBoosts")
    plt.legend(["LinearXGB train loss", "LinearXGB test loss", "OriginalXGB train loss", "OriginalXGB test loss"])
    plt.show()

    info_string = color.BOLD + "Linear XGBoost: \n" + color.END
    info_string += " "*5 + "best score is {} on the {} iteration (train)".format(np.min(linear_train_losses[1]), np.argmin(linear_train_losses[1])) + '\n'
    info_string += " "*5 + "best score is {} on the {} iteration (test)".format(np.min(linear_test_losses[1]), np.argmin(linear_test_losses[1])) + '\n'
    info_string += color.BOLD + "Original XGBoost: \n" + color.END
    info_string += " "*5 + "best score is {} on the {} iteration (train)".format(np.min(original_train_losses[1]), np.argmin(original_train_losses[1])) + '\n'
    info_string += " "*5 + "best score is {} on the {} iteration (test)".format(np.min(original_test_losses[1]), np.argmin(original_test_losses[1])) + '\n'

    print(info_string)