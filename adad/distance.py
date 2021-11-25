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

    def __init__(self, clf=None, k=5, ci=0.95, dist_metric='euclidean'):
        super(DAIndexGamma, self).__init__()

        self.clf = clf
        self.k = k
        self.ci = ci
        self.dist_metric = dist_metric
        assert dist_metric in DISTANCE_METRICS

        self.tree = None
        self.threshold = np.inf

    def fit(self, X, y=None):
        n = len(X)
        self.tree = BallTree(X, metric=self.dist_metric)
        dist, _ = self.tree.query(X, k=self.k + 1)
        dist_mean = np.sum(dist, axis=1) / self.k
        dist_sorted = np.sort(dist_mean)
        idx = int(np.floor(self.ci * n))
        self.threshold = dist_sorted[idx]
        return self

    def measure(self, X):
        """Check AD on X. Returns True if a sample is within the domain."""
        dist, _ = self.tree.query(X, k=self.k + 1)
        dist_mean = np.sum(dist, axis=1) / self.k
        measure = dist_mean / self.threshold
        # Less than 1 indicates the sample within the domain.
        results = measure <= 1
        # TODO: We might need consider to return `measure`(float) instead of 
        # binary values.
        return results
