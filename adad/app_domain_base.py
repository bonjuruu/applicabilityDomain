"""Base class for Applicability Domain classifier.
"""
from abc import ABC, abstractmethod
import pickle


class AppDomainBase(ABC):
    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def measure(self, X):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)
