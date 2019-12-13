import numpy as np
from tree import Tree
from objective import _get_objective

class BoostingTree():
    def __init__(self, params):
        self.params = params
        self.obj = _get_objective(params["objective"])

    def fit(self, X, y):
        self._set_lenght, self._feature_number = X.shape
        X = np.append(X, np.arange(self._set_lenght).reshape(self._set_lenght, 1), axis=1)
        self._X, self._y = X, y

        self._grad, self._hess = self.obj(y, np.zeros(X.shape[0]))

        init_value = {"a": np.zeros(self._feature_number),
                      "b": 0,
                      "feature": None,
                      "threshold": None,
                      "indicies": np.arange(self._set_lenght)}

        self.tree = Tree(init_value)
        self._fit()

    def _fit(self, nid=1):
        if nid < 2**self.params["max_depth"]:
            if self._updateTree(nid):
                self._fit(2*nid)
                self._fit(2*nid+1)

    def _updateTree(self, nid):
        value = self.tree.get_value(nid)

        X = self._selectByIndecies(value["indicies"])

        best_loss = -np.inf
        best_feature = None
        best_threshold = None
        best_matricies = None
        best_order=None

        for feature in np.arange(self._feature_number):
            order = np.argsort(X[:, feature])

            AL, BL = self._getMatricies(nid, "L", order, feature)
            AR, BR = self._getMatricies(nid, "R", order, feature)
            losses = self._getLosses(AL, BL, AR, BR, nid, order)
            max_val, argmax_val = self._max(losses, X[order, feature])

            if best_loss < max_val:
                best_loss = max_val
                best_feature = feature
                best_threshold = argmax_val
                best_matricies = (AL[best_threshold], BL[best_threshold], AR[best_threshold], BR[best_threshold])
                best_order = order


        if best_threshold == 0:
            daR, dbR = -np.linalg.solve(best_matricies[2], best_matricies[3].T)
            right_a = value["a"]
            right_a[best_feature] += daR
            value["a"] = right_a
            value["b"] = value["b"] + dbR
            return False
        elif best_threshold == X.shape[0]:
            daL, dbL = -np.linalg.solve(best_matricies[0], best_matricies[1].T)
            left_a = value["a"]
            left_a[best_feature] += daL
            value["a"] = left_a
            value["b"] = value["b"] + dbL
            return False


        best_threshold = X[best_order, best_feature][best_threshold]

        value["feature"] = best_feature
        value["threshold"] = best_threshold
        self.tree.set_value(nid, value)

        daL, dbL = -np.linalg.solve(best_matricies[0], best_matricies[1].T)
        daR, dbR = -np.linalg.solve(best_matricies[2], best_matricies[3].T)
        left_a, right_a = value["a"].copy(), value["a"].copy()
        left_a[best_feature] += daL
        right_a[best_feature] += daR
        left_indicies = X[X[:, best_feature] < best_threshold][:, -1]
        right_indicies = X[X[:, best_feature] >= best_threshold][:, -1]

        left_value = {"a": left_a,
                      "b": value["b"]+dbL,
                      "feature": None,
                      "threshold": None,
                      "indicies": left_indicies}

        right_value = {"a": right_a,
                       "b": value["b"]+dbR,
                       "feature": None,
                       "threshold": None,
                       "indicies": right_indicies}

        self.tree.get_split(nid, left_value, right_value)

        return True

    def _getMatricies(self, nid, side, order, feature):
        value = self.tree.get_value(nid)
        grad, hess = self._computeGradients(nid)
        X = self._selectByIndecies(value["indicies"])
        g, h, X = grad[order], hess[order], X[order]

        f = self._node_predict(nid)[order]
        x = X[:, feature]
        _lambda = self.params["lambda"]
        _mu = self.params["mu"]#*np.var(X)

        if side=="L":
            I = np.append(1, np.arange(1, x.shape[0]+1))
        elif side=="R":
            I = np.append(np.arange(x.shape[0], 0, -1), 1)
        else:
            I = Exception("Side should be L or R")
        A11 = self._cumsum(h * x ** 2, side) + _lambda * self._cumsum(x, side) ** 2 / I ** 2 + _mu
        A12 = self._cumsum(h * x, side) + _lambda * self._cumsum(x, side) / I
        A21 = A12
        A22 = self._cumsum(h, side) + _lambda

        B1 = self._cumsum((g + h * f) * x, side) + _lambda * self._cumsum(f, side) * \
             self._cumsum(x, side) / I ** 2 + _mu * value["a"][feature]
        B2 = self._cumsum((g + h * f), side) + _lambda * self._cumsum(f, side) / I

        A = self._stackMatricies(A11, A12, A21, A22)
        B = np.array([B1, B2]).T

        return A, B


    def _getLosses(self, AL, BL, AR, BR, nid, order):
        value = self.tree.get_value(nid)
        _lambda = self.params["lambda"]
        _mu = self.params["mu"]
        _nu = self.params["nu"]
        _gamma = self.params["gamma"]
        f = self._node_predict(nid)[order]
        IL = np.append(1, np.arange(1, AR.shape[0]))
        IR = np.append(np.arange(AR.shape[0]-1, 0, -1), 1)
        I = np.max(IL)

        losses = self._sideLoss(AL, BL) + self._sideLoss(AR, BR)
        losses -= (_gamma+.5*_mu*np.sum(value["a"]**2) + _nu * np.count_nonzero(value["a"]) -\
                   .5*_lambda*((np.sum(f)/I)**2-(self._cumsum(f, "L")/IL)**2-(self._cumsum(f, "R")/IR)**2))

        return losses

    def _sideLoss(self, A, B):
        B1 = np.append(B.reshape(B.shape[0],2,1), B.reshape(B.shape[0],2,1), axis=1).reshape(B.shape[0],2,2)
        B2 = np.append(B.reshape(B.shape[0],2,1), B.reshape(B.shape[0],2,1), axis=2).reshape(B.shape[0],2,2)
        losses_mat = .5*B1*np.linalg.inv(A)*B2

        return np.sum(losses_mat.reshape(B.shape[0], 4), axis=1)/2


    def _computeGradients(self, nid):
        value = self.tree.get_value(nid)
        indicies = np.sort(value["indicies"].astype(int))
        return self._grad[indicies], self._hess[indicies]

    def _node_predict(self, nid):
        value = self.tree.get_value(nid)
        X = self._selectByIndecies(value["indicies"])
        return np.dot(value["a"], X[:, :-1].T) + value["b"]

    def predict(self, X):
        self._prediction = {}
        preds = np.empty(X.shape[0])

        X = np.append(X, np.arange(X.shape[0]).reshape(X.shape[0], 1), axis=1)
        self._predict(X, 1, X[:, -1])
        for nid, inds in self._prediction.items():
            value = self.tree.get_value(nid)
            preds[inds.astype(int)] = self._predictOneLeaf(X[inds.astype(int)], value["a"], value["b"])
        return preds

    def _predict(self, X, nid, indicies):
        leaves = self.tree.tree.leaves()
        leaves = [leave.identifier for leave in leaves]
        if nid not in leaves:
            left_indicies, right_indicies = self._getPredictionSplit(X, nid, indicies)
            self._predict(X, 2 * nid, left_indicies)
            self._predict(X, 2 * nid + 1, right_indicies)
        else:
            if nid in self._prediction.keys():
                self._prediction[nid] = np.append(self._prediction[nid], indicies)
            else:
                self._prediction[nid] = indicies

    def _getPredictionSplit(self, X, nid, indicies):
        value = self.tree.get_value(nid)
        X = X[np.sort(indicies.astype(int))]
        left_indicies = X[X[:, value["feature"]] < value["threshold"]]
        right_indicies = X[X[:, value["feature"]] >= value["threshold"]]
        return left_indicies[:, -1], right_indicies[:, -1]

    def _predictOneLeaf(self, X, a, b):
        return np.dot(a, X[:, :-1].T) + b

    # ADDITIONAL FUNCTIONS

    def _selectByIndecies(self, indicies):
        return self._X[np.sort(indicies.astype(int))]

    def _cumsum(self, x, side):
        cumsum = np.cumsum(x)
        cumsum = np.append(0, cumsum)
        if side=="L":
            return cumsum
        elif side=="R":
            return np.sum(x)-cumsum
        else:
            raise Exception("Side should be L or R")

    def _stackMatricies(self, A11, A12, A21, A22):
        return np.vstack((A11, A12, A21, A22)).T.reshape(A11.shape[0], 2, 2)


    def _max(self, losses, x):
        _x = np.append(0, x)
        roll_x = np.roll(_x, -1)
        roll_x = roll_x - _x
        mask = (roll_x != 0).astype(int)
        return np.max(losses*mask), np.argmax(losses*mask)