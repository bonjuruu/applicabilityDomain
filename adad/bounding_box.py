from .app_domain_base import AppDomainBase
import numpy as np
from sklearn.decomposition import PCA

class PCABoundingBox():
    def __init__(self,clf=None, n_pc=5, seed=None):
        super(PCABoundingBox, self).__init__()
        
        self.clf = clf
        self.n_pc = n_pc
        self.SEED = seed

    def fit(self, X, Y):
        # fit PCA to the second two parts (triggered rules and MACCS fingerprint)
        self.pca = PCA(n_components=min(self.n_pc, len(X)), random_state=self.SEED)
        self.pca.fit(X)
        X_trf = self.pca.transform(X)
        # construct axis-aligned bounding box in PCA-space
        self.AABB = np.array([[min(X_trf[:,i]), max(X_trf[:,i])] for i in range(len(X_trf[0]))])

    def measure(self, X):
        return self.pca.score_samples(X)
    
    def predict(self, X):
        # reshape single samples to a matrix to deal with both one
        #  and multiple samples
        if len(X.shape) == 1: X = X.reshape((1,len(X)))

        sample_trf = self.pca.transform(X)
        return np.all((sample_trf >= self.AABB[:,0]) & (sample_trf <= self.AABB[:,1]), axis=1).astype(float)