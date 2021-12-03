"""Average Distance-based Applicability Domain (DA-Index Gamma)
"""
import logging

import numpy as np
from sklearn.neighbors import BallTree

from .app_domain_base import AppDomainBase

logger = logging.getLogger(__name__)

DISTANCE_METRICS = [
    # For floats
    'euclidean',
    'manhattan',
    'chebyshev',
    'minkowski',
    'wminkowski',
    'seuclidean',
    'mahalanobis',
    # For integers
    'hamming',
    'canberra',
    'braycurtis',
]


class DAIndexGamma(AppDomainBase):
    """Average distance based Applicability Domain (DA-Index Gamma)

    Parameters
    ----------
    clf: classifier, optional
        A pre-trained classifier. It should have `predict` and `predict_proba`
        methods.
    k: int, default=5
        The number of K-nearest neighbors.
    ci: float, default=0.95
        Confidence interval. It should in-between (0, 1].
    """

    def __init__(self, clf=None, k=5, dist_metric='euclidean'):
        super(DAIndexGamma, self).__init__()

        self.clf = clf
        self.k = k
        self.dist_metric = dist_metric
        assert dist_metric in DISTANCE_METRICS

        self.tree = None
        self.dist_measure_train = None

    def fit(self, X):
        self.tree = BallTree(X, metric=self.dist_metric)
        dist, _ = self.tree.query(X, k=self.k + 1)
        dist_mean = np.mean(dist[:, 1:], axis=1)
        self.dist_measure_train = np.sort(dist_mean)
        return self

    def measure(self, X):
        """Check AD on X. Returns True if a sample is within the domain."""
        dist, _ = self.tree.query(X, k=self.k)
        dist_measure = np.mean(dist, axis=1)
        return dist_measure


class DAIndexKappa(AppDomainBase):
    """Kth Nearest Distance based Applicability Domain (DA-Index Kappa)

    Parameters
    ----------
    clf: classifier, optional
        A pre-trained classifier. It should have `predict` and `predict_proba`
        methods.
    k: int, default=5
        The number of K-nearest neighbors.
    ci: float, default=0.95
        Confidence interval. It should in-between (0, 1].
    """

    def __init__(self, clf=None, k=5, dist_metric='euclidean'):
        super(DAIndexKappa, self).__init__()

        self.clf = clf
        self.k = k
        self.dist_metric = dist_metric
        assert dist_metric in DISTANCE_METRICS

        self.tree = None
        self.dist_measure_train = None

    def fit(self, X):
        n = len(X)
        self.tree = BallTree(X, metric=self.dist_metric)
        dist, _ = self.tree.query(X, k=self.k + 1)
        dist_at_k = dist[:, -1]
        dist_sorted = np.sort(dist_at_k)
        self.dist_measure_train = dist_sorted
        return self

    def measure(self, X):
        """Check AD on X. Returns True if a sample is within the domain."""
        dist, _ = self.tree.query(X, k=self.k)
        dist_at_k = dist[:, -1]
        return dist_at_k


class DAIndexDelta(AppDomainBase):
    """Length of mean vector based Applicability Domain (DA-Index Delta)

    Parameters
    ----------
    clf: classifier, optional
        A pre-trained classifier. It should have `predict` and `predict_proba`
        methods.
    k: int, default=5
        The number of K-nearest neighbors.
    ci: float, default=0.95
        Confidence interval. It should in-between (0, 1].
    """

    def __init__(self, clf=None, k=5, dist_metric='euclidean'):
        super(DAIndexDelta, self).__init__()

        self.clf = clf
        self.k = k
        self.dist_metric = dist_metric
        assert dist_metric in DISTANCE_METRICS

        self.tree = None
        self.dist_measure_train = None

    def fit(self, X):
        self.X = np.copy(X)
        n = len(X)
        self.tree = BallTree(X, metric=self.dist_metric)
        _, indices = self.tree.query(X, k=self.k + 1)
        dist_norm = np.array([np.linalg.norm(
            np.mean(self.X[indices[i, 1:]] - self.X[i], axis=0), ord=2) for i in range(n)])
        self.dist_measure_train = np.sort(dist_norm)
        return self

    def measure(self, X):
        """Check AD on X. Returns True if a sample is within the domain."""
        _, indices = self.tree.query(X, k=self.k)
        n = len(X)
        dist_norm = np.array([np.linalg.norm(
            np.mean(self.X[indices[i]] - X[i], axis=0), ord=2) for i in range(n)])
        return dist_norm
