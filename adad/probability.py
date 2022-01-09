import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .app_domain_base import AppDomainBase

CLASSIFIERS = (RandomForestClassifier, KNeighborsClassifier, SVC)


class ProbabilityClassifier(AppDomainBase):
    def __init__(self, clf):
        super(ProbabilityClassifier, self).__init__()

        self.clf = clf
        assert isinstance(clf, CLASSIFIERS)

        self.dist_measure_train = None

    def fit(self, X, y=None):
        all_p_x = [max(x) for x in self.clf.predict_proba(X)]
        error_p = [1 - x for x in all_p_x]

        self.dist_measure_train = np.sort(error_p)
        return self

    def measure(self, X):
        all_p_x = [max(x) for x in self.clf.predict_proba(X)]
        all_error_p = np.array([1 - x for x in all_p_x])

        return all_error_p
