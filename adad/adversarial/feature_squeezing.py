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
        Number of alternative discriptors.
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
            params = clf_params[i]
            self.clfs.append(Classifier(**params))

    def fit(self, Xs, y):
        time_start = time.perf_counter()
        for i, clf in enumerate(self.clfs):
            clf.fit(Xs[i], y)
        time_elapsed = time.perf_counter() - time_start
        logger.info(f'Total training time: {time2str(time_elapsed)}')
        return self

    def measure(self, X, Xs_alt):
        """Check AD on X.

        Parameters
        ----------
        X: array
            The unlabeled samples for testing.
        X_alt: list of array
            The unlabeled samples that use alternative descriptors
        """
        assert len(Xs_alt) == self.n_discriptors, \
            '# of alternative Xs does not match with # of expected discriptors.'

        pred = self.clf.predict(X)
        preds_alt = -np.ones((self.n_discriptors, len(pred)), dtype=pred.dtype)
        for i, clf in enumerate(self.clfs):
            preds_alt[i] = clf.predict(Xs_alt[i])
        
        result = np.sum(preds_alt == pred, axis=0) / self.n_discriptors
        return result
