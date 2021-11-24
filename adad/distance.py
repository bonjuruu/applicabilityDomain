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

    def __init__(self, clf=None, k=5, ci=0.95, dist_metric='euclidean'):
        super(DAIndexGamma, self).__init__()

        self.clf = clf
        self.k = k
        self.ci = ci
        self.dist_metric = dist_metric
        assert dist_metric in DISTANCE_METRICS

        self.tree = None
        self.threshold = np.inf

    def fit(self, X):
        n = len(X)
        self.tree = BallTree(X, metric=self.dist_metric)
        dist, _ = self.tree.query(X, k=self.k + 1)
        dist_mean = np.sum(dist, axis=1) / self.k
        dist_sorted = np.sort(dist_mean)
        idx = int(np.floor(self.ci * n))
        self.threshold = dist_sorted[idx]

    def measure(self, X):
        """Check AD on X. Returns True if a sample is within the domain."""
        dist, _ = self.tree.query(X, k=self.k + 1)
        dist_mean = np.sum(dist, axis=1) / self.k
        measure = dist_mean / self.threshold
        # Less than 1 indicates the sample within the domain.
        results = measure <= 1
        return results

    def predict(self, X):
        """Apply Applicability Domain and then perform classification on X using
        the given classifier.

        Returns
        -------
        pred: 1-D array
            The predictions of the samples that are passed AD test.
        idx: 1-D array
            The indices that are passed the AD test.
        """
        if self.clf == None or not hasattr(self.clf, 'predict'):
            raise RuntimeError(
                'This method is not supported.')
        if self.tree == None:
            raise RuntimeError(
                "Model hasn't trained. Call 'fit(X)' method first!")

        ad_measure = self.measure(X)
        idx = np.where(ad_measure)[0]
        pred = self.clf.predict(X[idx])
        return pred, idx

    def predict_proba(self, X):
        """Apply Applicability Domain and then compute probabilities of possible
        outcomes for X using the given classifier.

        Returns
        -------
        pred: 1-D array
            The predictions of the samples that are passed AD test.
        idx: 1-D array
            The indices that are passed the AD test.
        """
        if self.clf == None or not hasattr(self.clf, 'predict_proba'):
            raise RuntimeError(
                'This method is not supported.')
        if self.tree == None:
            raise RuntimeError(
                "Model hasn't trained. Call 'fit(X)' method first!")

        ad_measure = self.measure(X)
        idx = np.where(ad_measure)[0]
        pred = self.clf.predict_proba(X[idx])
        return pred, idx

    def score(self, X, y):
        """Apply Applicability Domain and return the accuracy on the subset data
        and labels.

        Returns
        -------
        score: float
            Mean accuracy of the subset data that have passed AD test.
        """
        if self.clf == None or not hasattr(self.clf, 'predict'):
            raise RuntimeError(
                'This method is not supported.')
        if self.tree == None:
            raise RuntimeError(
                "Model hasn't trained. Call 'fit(X)' method first!")

        pred, idx = self.predict(X)
        acc = np.mean(pred == y[idx])
        return acc
