import numpy as np

from .app_domain_base import AppDomainBase


class ProbabilityClassifier(AppDomainBase):
    def __init__(self, clf):
        super(ProbabilityClassifier, self).__init__()

        self.clf = clf

    def fit(self, X=None, y=None):
        return self

    def measure(self, X):
        all_p_x = np.max(self.clf.predict_proba(X), axis=1)
        error_p = 1 - all_p_x
        return error_p
