import unittest
import numpy as np

class APTest(unittest.TestCase):
    def run(self, ap):
        gt_items = np.asarray([2, 4, 5, 10])
        predictions_1 = np.asarray([1, 2, 3, 4, 5])
        predictions_2 = np.asarray([10, 5, 2, 4, 3])
        predictions_3 = np.asarray([1, 3, 6, 7, 8])
        predictions_4 = np.asarray([11, 12, 13, 14, 15, 16, 2, 4, 5, 10])
        predictions_5 = np.asarray([2, 11, 12, 13, 14, 15, 4, 5, 10, 16])
        self.assertTrue(np.allclose(ap(gt_items, predictions_1), (1. / 2 + 2. / 4 + 3. / 5) / 4))
        self.assertTrue(np.allclose(ap(gt_items, predictions_2), 1.0))
        self.assertTrue(np.allclose(ap(gt_items, predictions_3), 0.0))
        self.assertTrue(np.allclose(ap(gt_items, predictions_4), (1. / 7 + 2. / 8 + 3. / 9 + 4. / 10) / 4))
        self.assertTrue(np.allclose(ap(gt_items, predictions_5), (1. + 2. / 7 + 3. / 8 + 4. / 9) / 4))

        thresholds = [1, 2, 3, 4, 5]
        values = [
            0.0,
            1. / 2 / 2,
            1. / 2 / 3,
            (1. / 2 + 2. / 4) / 4,
            (1. / 2 + 2. / 4 + 3. / 5) / 4
        ]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(ap(gt_items, predictions_1, at)), val))

class RRTest(unittest.TestCase):
    def run(self, rr):
        gt_items = np.asarray([2, 4, 5, 10])
        predictions_1 = np.asarray([1, 2, 3, 4, 5])
        predictions_2 = np.asarray([10, 5, 2, 4, 3])
        predictions_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(rr(gt_items, predictions_1), 1. / 2))
        self.assertTrue(np.allclose(rr(gt_items, predictions_2), 1.))
        self.assertTrue(np.allclose(rr(gt_items, predictions_3), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1. / 2, 1. / 2, 1. / 2, 1. / 2]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(rr(gt_items, predictions_1, at)), val))

class HRTest(unittest.TestCase):
    def run(self, hr):
        gt_items = np.asarray([2, 4, 5, 10])
        predictions_1 = np.asarray([1, 2, 3, 4, 5])
        predictions_2 = np.asarray([10, 5, 2, 4, 3])
        predictions_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(hr(gt_items, predictions_1), 1))
        self.assertTrue(np.allclose(hr(gt_items, predictions_2), 1))
        self.assertTrue(np.allclose(hr(gt_items, predictions_3), 0))

        thresholds = [1, 2, 3]
        values = [0, 1, 1]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(hr(gt_items, predictions_1, at)), val))

def run_tests(*funcs):
    for cls, func in zip([APTest, RRTest, HRTest], funcs):
        cls().run(func)