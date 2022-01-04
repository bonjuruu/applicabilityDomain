"""Modified Feature Squeezing for Applicability Domain
"""
import logging
import time

import numpy as np

from adad.app_domain_base import AppDomainBase
from adad.utils import time2str

logger = logging.getLogger(__name__)


class SklearnFeatureSqueezing(AppDomainBase):
    """Feature Squeezing for Applicability Domain (scikit-learn version)

    Parameters
    ----------
    clf: classifier
        A pre-trained scikit-learn classifier.
    Classifier: classifier
        The classifier that will be used for all descriptors.
    n_discriptors: int
        Number of discriptors per sample.
    clf_params: 
        List of parameters that are used by each classifier.
    """

    def __init__(self, clf, Classifier, n_discriptors, clf_params):
        self.clf = clf
        self.n_discriptors = n_discriptors

        assert n_discriptors == len(clf_params), \
            '# of discriptors does not match with # of parameter sets.'

        self.clfs = []
        for i in range(n_discriptors):
            self.clfs.append(Classifier(*clf_params[i]))

    def fit(self, X, y):
        time_start = time.perf_counter()
        for clf in self.clfs:
            clf.fit(X, y)
        time_elapsed = time.perf_counter() - time_start
        logger.info(f'Total training time: {time2str(time_elapsed)}')

    def measure(self, X, *X_alt):
        """Check AD on X.

        Parameters
        ----------
        X: array
            The unlabeled samples for testing.
        X_alt: list of array
            The unlabeled samples that use alternative descriptors
        """
        assert len(X_alt) == self.n_discriptors, \
            '# of alternative Xs does not match with # of expected discriptors.'

        pred = self.clf.predict(X)
        pred_alt = -np.ones((self.n_discriptors, len(pred)), dtype=pred.dtype)
        for i, clf in enumerate(self.clfs):
            pred_alt[i] = clf.predict(X)
        return pred, pred_alt
