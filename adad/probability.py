import numpy as np
from numpy.core.fromnumeric import argmax

from .app_domain_base import AppDomainBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class ProbabilityRFC(AppDomainBase):
    def __init__(self, clf, ci=0.95):
        super(ProbabilityRFC, self).__init__()
        
        self.clf = clf
        assert isinstance(clf, RandomForestClassifier)
        self.no_trees = len(self.clf.estimators_)
        
        self.ci = ci
        
    def fit(self, X, y=None):
        self.X = X
        decision_trees = self.clf.estimators_
        self.terminals = self.clf.apply(X).transpose()
        
        all_p_RFC = []
        all_error_prob = []
        for n in range(len(X)):
            p1_RFC = []
            p2_RFC = []
            
            for i in range(len(decision_trees)):
                tree = decision_trees[i]
                leaf = self.terminals[i][n]
                k_indices = np.where(self.terminals[i] == leaf)[0]
                k_indices = np.delete(k_indices, np.where(k_indices == n))
                
                if len(k_indices) != 0:
                    p1_RFC.append(1/len(k_indices) * np.sum(tree.predict(X[k_indices]) == 0))
                    p2_RFC.append(1/len(k_indices) * np.sum(tree.predict(X[k_indices]) == 1))
                    
            if len(p1_RFC ) == 0 and len(p2_RFC) == 0:
                c_pred = [argmax(tree.predict_proba(X[n].reshape(1, -1))) for tree in decision_trees]
                v1 = 1/self.no_trees * np.sum(c_pred == 0)
                v2 = 1/self.no_trees * np.sum(c_pred == 1)
                v_RFC = max(v1, v2)
                all_p_RFC.append(v_RFC)
                all_error_prob.append(1 - v_RFC)
            
            else:
                p1_n_2 = [1/self.no_trees * np.sum(p1_RFC), 1/self.no_trees * np.sum(p2_RFC)]
                p_RFC = max(p1_n_2)
                all_p_RFC.append(p_RFC)
                all_error_prob.append(1 - p_RFC)

        error_p_sorted = np.sort(all_error_prob)
        idx = int(np.floor(self.ci * len(X)))
        self.threshold = error_p_sorted[idx]
        return self
        
    def measure(self, X):
        terminals = self.clf.apply(X).transpose()
        decision_trees = self.clf.estimators_
        
        all_error_prob = []
        for n in range(len(X)):
            p1_RFC = []
            p2_RFC = []
            
            for i in range(len(decision_trees)):
                tree = decision_trees[i]
                leaf = terminals[i][n]
                k_indices = np.where(self.terminals[i] == leaf)[0]
                
                if len(k_indices) != 0:
                    p1_RFC.append(1/len(k_indices) * np.sum(tree.predict(self.X[k_indices]) == 0))
                    p2_RFC.append(1/len(k_indices) * np.sum(tree.predict(self.X[k_indices]) == 1))
                    
            if len(p1_RFC) == 0 and len(p2_RFC) == 0:
                c_pred = [argmax(tree.predict_proba(X[n].reshape(1, -1))) for tree in decision_trees]
                v1 = 1/self.no_trees * np.sum(c_pred == 0)
                v2 = 1/self.no_trees * np.sum(c_pred == 1)
                v_RFC = max(v1, v2)
                all_error_prob.append(1 - v_RFC)
            
            else:
                p1_n_2 = [1/self.no_trees * np.sum(p1_RFC), 1/self.no_trees * np.sum(p2_RFC)]
                p_RFC = max(p1_n_2)
                all_error_prob.append(1 - p_RFC)
        
        measure = all_error_prob / self.threshold
        results = measure <= 1
        return results
    
class ProbabilityKNN(AppDomainBase):
    def __init__(self, clf, k, ci=0.95):
        super(ProbabilityKNN, self).__init__()
        
        self.clf = clf
        assert isinstance(clf, KNeighborsClassifier)

        self.k = k
        self.ci = ci
        
    def fit(self, X, y=0):
        self.X = X
        k_neighbours = self.clf.kneighbors(X, return_distance=False)
        all_p_error = []
        
        for i in range(len(X)):
            c_pred = self.clf.predict(X[k_neighbours[i]])[1:]
            p1_kNN = 1 / self.k * np.sum(c_pred == 0)
            p2_kNN = 1 / self.k * np.sum(c_pred == 1)
            
            all_p_error.append(1 - max(p1_kNN, p2_kNN))
        
        all_p_error_sorted = np.sort(all_p_error)
        idx = int(np.floor(self.ci * len(X)))
        self.threshold = all_p_error_sorted[idx]
        return self
        
    
    def measure(self, X):
        k_neighbours = self.clf.kneighbors(X, return_distance=False)
        all_p_error = []
        
        for i in range(len(X)):
            c_pred = self.clf.predict(self.X[k_neighbours[i]])[1:]
            p1_kNN = 1 / self.k * np.sum(c_pred == 0)
            p2_kNN = 1 / self.k * np.sum(c_pred == 1)
            
            all_p_error.append(1 - max(p1_kNN, p2_kNN))
        
        measure = all_p_error / self.threshold
        results = measure <= 1
        return results
            
    
class ProbabilitySVM(AppDomainBase):
    def __init__(self, clf, ci=0.95):
        super(ProbabilitySVM, self).__init__()

        self.clf = clf
        assert isinstance(clf, SVC)
        self.no_trees = len(self.clf.estimators_)
        
        self.ci = ci
    
    