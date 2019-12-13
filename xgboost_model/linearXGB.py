import numpy as np
import copy
from preprocessing import normalize
import objective as obj
from training import BoostingTree
from callbacks import Callback as cb

class linearXBG():
    def __init__(self, params, n_estimators=10, eval_sets=None, verbose=False, early_stopping_rounds=False):
        params = self._checkParams(params=params,
                                   n_estimators=n_estimators,
                                   eval_sets=eval_sets,
                                   verbose=verbose)

        self.model = []

        self.objective = obj._get_objective(params["objective"])

        self.eval_metric = obj._get_loss(params["eval_metric"])

        if params["objective"] in ["binary:logistic"]:
            self.sigmoid=True
        else: self.sigmoid=False

        if params["eval_metric"] in ["accuracy"]:
            self.binarise=True
        else: self.binarise=False

        self.n_estimators = n_estimators
        self.verbose = verbose
        self.params = params

        self.eval_sets = eval_sets
        self.early_stopping_rounds = early_stopping_rounds
        if self.eval_sets is None:
            self.early_stopping_rounds = False

        self.scores = []

        self.eval_preds = {}
        self.callbacks = cb()
        self._score_len = 0

        self.best_score = np.inf

    def fit(self, X, y):
        X, y = np.array(X), np.array(y).reshape(y.shape[0], )

        self.params["lambda"] = self._norm(self.params["lambda"], X.shape[0])

        if self.eval_sets is None:
            self.eval_sets = [([X, y], "train")]
        else:
            if self.eval_sets[0][1] != "train":
                self.eval_sets = [([X, y], "train")] + self.eval_sets

        for i in range(len(self.eval_sets)):
            self.eval_sets[i][0][0] = np.array(self.eval_sets[i][0][0])
            self.eval_sets[i][0][1] = np.array(self.eval_sets[i][0][1]).reshape(self.eval_sets[i][0][1].shape[0], )

        if self.params["normalize"]:
            X = normalize(X)

            for i in range(len(self.eval_sets)):
                self.eval_sets[i][0][0] = normalize(self.eval_sets[i][0][0])

        initial_params = self.params.copy()
        bst = BoostingTree(initial_params)
        bst.fit(X, y)
        self.model += [copy.deepcopy(bst)]

        for eval in self.eval_sets:
            self.eval_preds[eval[1]] = bst.predict(eval[0][0])

        ntree = 0
        while ntree in range(self.n_estimators):
            if self.early_stopping_rounds:
                if self.verbose is not False and (ntree%self.verbose==0 or ntree==self.n_estimators-1):
                    print_stdout=True
                else:
                    print_stdout=False
                self._score_len, score = self.callbacks.evaluate(model=self,
                                                                 eval_sets=self.eval_sets,
                                                                 eval_preds=self.eval_preds,
                                                                 ntree=ntree,
                                                                 score_len=self._score_len,
                                                                 n_estimators= self.n_estimators,
                                                                 binarise=self.binarise,
                                                                 print_stdout=print_stdout)

                if self.best_score > score:
                    self.best_score = score
                    self.best_ntree = ntree

                self.scores += [score]

                if self.best_score < np.min(self.scores[-self.early_stopping_rounds:]):
                    if self.verbose is not False:
                        print("Stopped after {} iterations".format(ntree))
                    return self
            else:
                if self.verbose is not False and self.eval_sets is not None and (ntree%self.verbose == 0 or ntree==self.n_estimators-1):
                    self._score_len, score = self.callbacks.evaluate(model=self,
                                                                     eval_sets=self.eval_sets,
                                                                     eval_preds=self.eval_preds,
                                                                     ntree=ntree,
                                                                     score_len=self._score_len,
                                                                     n_estimators= self.n_estimators,
                                                                     binarise=self.binarise,
                                                                     print_stdout=True)

            grad = self.objective(y, self.eval_preds["train"])[0]
            self._boostOneIter(X, -grad)
            ntree += 1

        return self


    def _boostOneIter(self, X, grad):
        bst = BoostingTree(self.params)
        bst.fit(X, grad)
        self.model += [copy.deepcopy(bst)]

        for eval_name in self.eval_preds.keys():
            self.eval_preds[eval_name] += self.params["learning_rate"] * bst.predict(self._get_eval_set(self.eval_sets, eval_name))


    def predict(self, X, ntree_limit=-1):

        X = np.array(X)

        if self.params["normalize"]:
            X = normalize(X)

        if ntree_limit == -1:
            ntree_limit = len(self.model)
        preds = self.model[0].predict(X)
        for ntree in np.arange(1, ntree_limit):
            preds += self.params["learning_rate"] * self.model[ntree].predict(X)

        return preds

    def _checkParams(self, params, n_estimators, eval_sets, verbose):
        params_keys = params.keys()
        if "learning_rate" not in params_keys:
            params["learning_rate"] = .1
        if "max_depth" not in params_keys:
            params["max_depth"] = 3
        if "objective" not in params_keys:
            params["objective"] = "reg:linear"
        if "eval_metric" not in params_keys:
            params["eval_metric"] = params["objective"]
        if "alpha" not in params_keys:
            params["alpha"] = 0
        if "lambda" not in params_keys:
            params["lambda"] = 1
        if "mu" not in params_keys:
            params["mu"] = .1
        if "nu" not in params_keys:
            params["nu"] = 0
        if "gamma" not in params_keys:
            params["gamma"] = 0
        if "normalize" not in params_keys:
            params["normalize"] = False
        if "eval_metric" not in params_keys:
            params["eval_metric"] = "rmse"

        if isinstance(n_estimators, int) is False:
            raise Exception("n_estimators should be integer.")
        if isinstance(params["max_depth"], int) is False:
            raise Exception("Depth should be integer.")
        if isinstance(params["learning_rate"], (float, int)) is False:
            raise Exception("Parameter alpha should be float.")
        if isinstance(params["alpha"], (float, int)) is False:
            raise Exception("Parameter alpha should be float.")
        if isinstance(params["lambda"], (float, int)) is False:
            raise Exception("Parameter lambda should be float.")
        if isinstance(params["mu"], (float, int)) is False:
            raise Exception("Parameter mu should be float.")
        if isinstance(params["nu"], (float, int)) is False:
            raise Exception("Parameter nu should be float.")
        if isinstance(params["gamma"], (float, int)) is False:
            raise Exception("Parameter gamma should be float.")
        if isinstance(params["normalize"], bool) is False:
            raise Exception("Parameter normalize should be bool.")
        if isinstance(verbose, (bool, int)) is False:
            raise Exception("Parameter verbose should be bool or integer.")

        if eval_sets is not None:
            if isinstance(eval_sets, list) is False:
                raise Exception("eval_sets should be None or list")

        return params

    def _get_eval_set(self, eval_sets, eval_name):
        for eval in eval_sets:
            if eval[1] == eval_name:
                return eval[0][0]


    def _norm(self, x, lenght):
        return x/lenght