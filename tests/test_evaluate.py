"""Testing methods in adad.evaluate.py
"""
import os

import numpy as np
import pytest
from adad.evaluate import (cumulative_accuracy, permutation_auc,
                           predictiveness_curves, roc_ad,
                           sensitivity_specificity)
from adad.utils import set_seed

PATH_TEST = os.path.join(os.getcwd(), 'tests', 'results')


@pytest.fixture(autouse=True)
def setup():
    set_seed(1234)

    if not os.path.exists(PATH_TEST):
        print('Creating dir:', PATH_TEST)
        os.mkdir(PATH_TEST)


@pytest.fixture
def input1():
    y_true = np.concatenate((np.ones(5), np.zeros(5))).astype(int)
    y_pred = np.copy(y_true)
    return y_true, y_pred


@pytest.fixture
def input2():
    y_true = np.concatenate((np.ones(5), np.zeros(5))).astype(int)
    y_pred = np.concatenate((np.zeros(5), np.ones(5))).astype(int)
    return y_true, y_pred


@pytest.fixture
def input3():
    y_true = np.ones(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    return y_true, y_pred


@pytest.fixture
def input4():
    y_true = np.ones(10, dtype=int)
    y_pred = np.concatenate(([1], np.zeros(9))).astype(int)
    return y_true, y_pred


@pytest.fixture
def input5():
    y_true = np.concatenate((np.ones(5), np.zeros(5))).astype(int)
    y_pred = np.concatenate((np.ones(3), np.zeros(5), np.ones(2))).astype(int)
    return y_true, y_pred


@pytest.fixture
def input6():
    y_true = np.array((np.ones(10, dtype=int),
                       np.zeros(10, dtype=int),
                       np.concatenate((np.ones(5), np.zeros(5))).astype(int),
                       np.concatenate((np.zeros(5), np.ones(5))).astype(int))
                      )
    y_score = np.random.rand(4, 10)
    return y_true, y_score


@pytest.fixture
def dist_measure1():
    return np.ones(10, dtype=float)


@pytest.fixture
def dist_measure2():
    return np.linspace(0.1, 1.0, num=10)


@pytest.fixture
def fprs_tprs():
    fprs = (
        np.linspace(0, 1, 10),
        np.array([0., .2, 1.]),
        np.linspace(0, 1, 10),
    )
    tprs = (
        np.linspace(0, 1, 10),
        np.array([0., .8, 1.]),
        np.concatenate(([0.], np.linspace(0.8, 1, 9),))
    )
    return fprs, tprs


class TestEvaluate:
    def test_cumulative_accuracy(self, input1, input2, input4, dist_measure1,
                                 dist_measure2):
        # All positive case
        acc, rate = cumulative_accuracy(*input1, dist_measure=dist_measure1)
        assert len(acc) == 10
        np.testing.assert_equal(acc, 1.0)
        assert acc.shape == rate.shape
        # Expecting 0.1, 0.2, ..., 1.0
        target = np.linspace(0.1, 1.0, num=10)
        np.testing.assert_array_almost_equal(rate, target)

        # All negative case
        acc, _ = cumulative_accuracy(*input2, dist_measure=dist_measure1)
        np.testing.assert_equal(acc, 0.)

        # Mixed case
        acc, _ = cumulative_accuracy(*input4, dist_measure=dist_measure2)
        target = 1. / np.linspace(1, 10, num=10)
        np.testing.assert_array_almost_equal(acc, target)

    def test_permutation_auc(self, input5, dist_measure2):
        # Implemented based on Matlab code from:
        # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0230-2
        # Do not handle extrema
        perm_auc, _ = permutation_auc(*input5, dist_measure2)
        # Value is based on observation:
        np.testing.assert_almost_equal(perm_auc, 0.88, decimal=2)

    def test_predictiveness_curves(self, input1, input2, input5, dist_measure2):
        # All positive case
        percentile, error_rate = predictiveness_curves(
            *input1, dist_measure2, n_quantiles=10)
        target = np.linspace(0.1, 1, num=10)
        np.testing.assert_array_almost_equal(percentile, target)
        np.testing.assert_array_equal(error_rate, np.zeros_like(error_rate))

        # All negative case
        _, error_rate = predictiveness_curves(
            *input2, dist_measure2, n_quantiles=10)
        np.testing.assert_array_equal(error_rate, np.ones_like(error_rate))

        # Mixed case
        _, error_rate = predictiveness_curves(
            *input5, dist_measure2, n_quantiles=10)
        target = np.array([0, 0, 0, 1 / 4, 1 / 2.5, 1 / 3,
                          1 / 3.5, 1 / 4, 1 / 3, 1 / 2.5])
        np.testing.assert_array_almost_equal(error_rate, target)

    def test_roc_ad(self, input5, dist_measure2):
        # The sklearn.metrics.roc_curve method does NOT handle extreme case.
        # It returns nan on tpr if there's no false positives.
        fpr, tpr = roc_ad(*input5, dist_measure2)
        assert fpr.shape == tpr.shape
        np.testing.assert_almost_equal(fpr, [0., 0., 0., 0.5, 0.5, 1.])
        np.testing.assert_array_almost_equal(tpr, [0., 0.25, 0.5, 0.5, 1., 1.])

    def test_sensitivity_specificity(self, input1, input2, input3):
        # All positive case
        tpr, tnr = sensitivity_specificity(*input1)
        assert tpr == 1
        assert tnr == 1

        # All negative case 1
        tpr, tnr = sensitivity_specificity(*input2)
        assert tpr == 0
        assert tnr == 0

        # All negative case 2
        tpr, tnr = sensitivity_specificity(*input3)
        assert tpr == 0
        assert tnr == 0
