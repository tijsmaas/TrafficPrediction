import unittest

import numpy as np
import tensorflow as tf
import torch

from lib.metrics import metrics_tf, metrics_np, metrics_torch


class MyTestCase(unittest.TestCase):
    def test_masked_mape_np(self):
        preds = np.array([
            [1, 2, 2],
            [3, 4, 5],
        ], dtype=np.float32)
        labels = np.array([
            [1, 2, 2],
            [3, 4, 4]
        ], dtype=np.float32)
        mape = metrics_np.masked_mape_np(preds=preds, labels=labels)
        self.assertAlmostEqual(1 / 24.0, mape, delta=1e-5)

    def test_masked_mape_np2(self):
        preds = np.array([
            [1, 2, 2],
            [3, 4, 5],
        ], dtype=np.float32)
        labels = np.array([
            [1, 2, 2],
            [3, 4, 4]
        ], dtype=np.float32)
        mape = metrics_np.masked_mape_np(preds=preds, labels=labels, null_val=4)
        self.assertEqual(0., mape)

    def test_masked_mape_np_all_zero(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float32)
        mape = metrics_np.masked_mape_np(preds=preds, labels=labels, null_val=0)
        self.assertEqual(0., mape)

    def test_masked_mape_np_all_nan(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ], dtype=np.float32)
        mape = metrics_np.masked_mape_np(preds=preds, labels=labels)
        self.assertEqual(0., mape)

    def test_masked_mape_np_nan(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [np.nan, np.nan],
            [np.nan, 3]
        ], dtype=np.float32)
        mape = metrics_np.masked_mape_np(preds=preds, labels=labels)
        self.assertAlmostEqual(1 / 3., mape, delta=1e-5)

    def test_masked_rmse_np_vanilla(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, 4],
            [3, 4]
        ], dtype=np.float32)
        mape = metrics_np.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertEqual(1., mape)

    def test_masked_rmse_np_nan(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, np.nan],
            [3, 4]
        ], dtype=np.float32)
        rmse = metrics_np.masked_rmse_np(preds=preds, labels=labels)
        self.assertEqual(0., rmse)

    def test_masked_rmse_np_all_zero(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float32)
        mape = metrics_np.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertEqual(0., mape)

    def test_masked_rmse_np_missing(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, 0],
            [3, 4]
        ], dtype=np.float32)
        mape = metrics_np.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertEqual(0., mape)

    def test_masked_rmse_np2(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32)
        rmse = metrics_np.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(np.sqrt(1 / 3.), rmse, delta=1e-5)


class TFRMSETestCase(unittest.TestCase):
    def test_masked_mse_null(self):
        with tf.Session() as sess:
            preds = tf.constant(np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float32))
            labels = tf.constant(np.array([
                [1, 0],
                [3, 3]
            ], dtype=np.float32))
            rmse = metrics_tf.masked_mse_tf(preds=preds, labels=labels, null_val=0)
            self.assertAlmostEqual(1 / 3.0, sess.run(rmse), delta=1e-5)

    def test_masked_mse_vanilla(self):
        with tf.Session() as sess:
            preds = tf.constant(np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float32))
            labels = tf.constant(np.array([
                [1, 0],
                [3, 3]
            ], dtype=np.float32))
            rmse = metrics_tf.masked_mse_tf(preds=preds, labels=labels)
            self.assertAlmostEqual(1.25, sess.run(rmse), delta=1e-5)

    def test_masked_mse_all_zero(self):
        with tf.Session() as sess:
            preds = tf.constant(np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float32))
            labels = tf.constant(np.array([
                [0, 0],
                [0, 0]
            ], dtype=np.float32))
            rmse = metrics_tf.masked_mse_tf(preds=preds, labels=labels, null_val=0)
            self.assertAlmostEqual(0., sess.run(rmse), delta=1e-5)

    def test_masked_mse_nan(self):
        with tf.Session() as sess:
            preds = tf.constant(np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float32))
            labels = tf.constant(np.array([
                [1, 2],
                [3, np.nan]
            ], dtype=np.float32))
            rmse = metrics_tf.masked_mse_tf(preds=preds, labels=labels)
            self.assertAlmostEqual(0., sess.run(rmse), delta=1e-5)

    def test_masked_mse_all_nan(self):
        with tf.Session() as sess:
            preds = tf.constant(np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float32))
            labels = tf.constant(np.array([
                [np.nan, np.nan],
                [np.nan, np.nan]
            ], dtype=np.float32))
            rmse = metrics_tf.masked_mse_tf(preds=preds, labels=labels, null_val=0)
            self.assertAlmostEqual(0., sess.run(rmse), delta=1e-5)

class TorchRMSETestCase(unittest.TestCase):
    def test_masked_mse_null(self):
        preds = torch.from_numpy(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = torch.from_numpy(np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32))
        rmse = metrics_torch.masked_mse_torch(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(1 / 3.0, rmse.item(), delta=1e-5)

    def test_masked_mse_vanilla(self):
        preds = torch.from_numpy(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = torch.from_numpy(np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32))
        rmse = metrics_torch.masked_mse_torch(preds=preds, labels=labels)
        self.assertAlmostEqual(1.25, rmse.item(), delta=1e-5)

    def test_masked_mse_all_zero(self):
        preds = torch.from_numpy(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = torch.from_numpy(np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float32))
        rmse = metrics_torch.masked_mse_torch(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(0., rmse.item(), delta=1e-5)

    def test_masked_mse_nan(self):
        preds = torch.from_numpy(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = torch.from_numpy(np.array([
            [1, 2],
            [3, np.nan]
        ], dtype=np.float32))
        rmse = metrics_torch.masked_mse_torch(preds=preds, labels=labels)
        self.assertAlmostEqual(0., rmse.item(), delta=1e-5)

    def test_masked_mse_all_nan(self):
        preds = torch.from_numpy(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = torch.from_numpy(np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ], dtype=np.float32))
        rmse = metrics_torch.masked_mse_torch(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(0., rmse.item(), delta=1e-5)

if __name__ == '__main__':
    unittest.main()
