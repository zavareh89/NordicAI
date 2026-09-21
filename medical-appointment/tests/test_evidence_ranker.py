import unittest
import numpy as np

from dev.ranker_training import fit_ridge, predict


class RankerTests(unittest.TestCase):
    def test_ridge_learns_simple_ordering(self):
        X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        y = np.asarray([0.0, 0.2, 0.8, 1.0], dtype=np.float64)
        model = fit_ridge(X, y, 0.01)
        pred = predict(X, *model)
        self.assertGreater(pred[-1], pred[0])
        self.assertGreater(pred[2], pred[1])


if __name__ == "__main__":
    unittest.main()
