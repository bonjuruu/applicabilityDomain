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
        raise NotImplementedError

    @abstractmethod
    def score(self, X, y):
        raise NotImplementedError

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)
