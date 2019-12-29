import unittest
import numpy as np
import sys
sys.path.append("../")
import objective as obj


class TestObjective(unittest.TestCase):
    def _assertNumpyArrays(self, arr1, arr2):
        for el1, el2 in zip(arr1, arr2):
            self.assertAlmostEqual(el1, el2, delta=1e-5)

    def test_rmse_1(self):
        loss = obj.rmse_score(np.array([1, 2]),
                        np.array([3, 2]))

        self.assertAlmostEqual(loss, 1.41421356)
                               
    def test_get_metrics(self):
        rmse = obj._get_objective("rmse")
        loss = rmse(np.array([1, 2]),
                        np.array([3, 2]))
        
        self.assertAlmostEqual(loss, 1.41421356)

    def test_get_metrics_1(self):
        reg_linear = obj._get_objective("reg:linear")
        loss = reg_linear(np.array([1, 2]),
                        np.array([3, 2]))
        
        self.assertAlmostEqual(list(loss[0]), [2, 0])

    def test_reg_linear_1(self):
        loss = obj.reg_linear(np.array([1, 2]),
                        np.array([3, 2]))

        self.assertAlmostEqual(list(loss[0]), [2, 0])

