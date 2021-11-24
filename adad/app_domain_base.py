"""Base class for Applicability Domain classifier.
"""
from abc import ABC, abstractmethod
import pickle

import numpy as np


class AppDomainBase(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def measure(self, X):
        raise NotImplementedError

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
                'Predict is not supported.')

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
                'Predict_proba is not supported.')

        ad_measure = self.measure(X)
        idx = np.where(ad_measure)[0]
        pred = self.clf.predict_proba(X[idx])
        return pred, idx

    def score(self, X, y):
        """Apply Applicability Domain and return the accuracy on the subset data
        and labels. 
        """
        if self.clf == None or not hasattr(self.clf, 'predict'):
            raise RuntimeError(
                'This method is not supported.')

        pred, idx = self.predict(X)
        acc = np.mean(pred == y[idx])
        return acc

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)
