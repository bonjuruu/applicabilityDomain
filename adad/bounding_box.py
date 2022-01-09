import numpy as np
from sklearn.decomposition import PCA

from .app_domain_base import AppDomainBase


class PCABoundingBox(AppDomainBase):
    """Apply Bounding box on PCA space. Use quantiles to estimate the confidence
    level.

    Parameters
    ----------
    clf: classifier, optional
        A pre-trained classifier. It should have `predict` and `predict_proba`
        methods. If clf is None, `filter_train` must be False.
    n_components: int, default=5
        The estimated number of components. 
    min_ci: float, default=0.9
        Minimal confidence interval
    n_stages: int, default=10
        Number of intervals to measure
    random_state: int, default=None
        Random state.
    filter_train: bool, default=True
        If set to True, only use the samples that are correctly classified by 
        the classifier to train.
    """

    def __init__(self,
                 clf=None,
                 n_components=5,
                 min_ci=0.9,
                 n_stages=10,
                 random_state=None,
                 filter_train=True):
        super(PCABoundingBox, self).__init__()

        self.clf = clf
        self.n_components = n_components
        self.min_ci = min_ci
        self.n_stages = n_stages
        if random_state == None:
            random_state = np.random.randint(0, 999999)
        self.random_state = random_state
        self.filter_train = filter_train

        self.pca = None
        self.boundingboxes = None

    def fit(self, X, y):
        n = X.shape[0]
        if self.filter_train:
            # Only use the samples that are correctly classified by the classifier.
            y_pred = self.clf.predict(X)
            idx = np.where(y_pred == y)[0]
            X = X[idx]
            y = y[idx]
            # print(f'Apply filter Before: {n} After: {X.shape[0]}')

        self.n_components = np.min([X.shape[0], X.shape[1], self.n_components])
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X)
        X_eigen = self.pca.transform(X)
        X_eigen_sorted = np.sort(X_eigen, axis=0)

        cis = np.linspace(1, self.min_ci, self.n_stages)
        self.boundingboxes = np.zeros((self.n_stages, 2, self.n_components), dtype=float)
        for i, ci in enumerate(cis):
            idx_min = np.max([int(np.floor((1 - ci) * X.shape[0])), 0])
            idx_max = np.min([int(np.floor(ci * X.shape[0])), X.shape[0] - 1])

            self.boundingboxes[i, 0] = X_eigen_sorted[idx_min, :]
            self.boundingboxes[i, 1] = X_eigen_sorted[idx_max, :]
        return self

    def measure(self, X):
        thresholds = np.zeros((X.shape[0], self.n_stages), dtype=float)
        X_eigen = self.pca.transform(X)

        for i, boundingbox in enumerate(self.boundingboxes):
            thresholds[:, i] = np.all((X_eigen >= boundingbox[0]) & (X_eigen <= boundingbox[1]), axis=1)
        confidence_lvl = np.sum(thresholds, axis=1) / self.n_stages
        error_lvl = 1 - confidence_lvl
        return error_lvl
