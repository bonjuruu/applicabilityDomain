import os

import numpy as np
import pandas as pd
import pytest
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.distance import DAIndexGamma, DAIndexKappa, DAIndexDelta

SEED = 348
np.random.seed(348)

X = randint(5, size=(20, 5))
y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

classifier = RandomForestClassifier(random_state=SEED)
classifier.fit(X_train, y_train)


def test_Gamma_constructor():
    ad = DAIndexGamma(dist_metric='euclidean')
    assert ad.dist_metric == 'euclidean'

    with pytest.raises(AssertionError) as e_info:
        ad = DAIndexGamma(dist_metric='asdkjnas')


def test_Gamma_fit():
    ad = DAIndexGamma()
    ad.fit(X_train)

    dist, _ = ad.tree.query(X_train, k=ad.k + 1)
    dist_mean = np.sum(dist, axis=1) / ad.k
    dist_sorted = np.sort(dist_mean)
    idx = int(np.floor(ad.ci * len(X_train)))

    assert ad.threshold <= dist_sorted[len(
        X_train) - 1] and ad.threshold >= dist_sorted[idx - 1]


def test_Gamma_measure():
    ad = DAIndexGamma(clf=classifier)
    ad.fit(X_train)
    results = ad.measure(X_train)

    assert all([type(x) == np.bool_ for x in results])

    results2 = ad.measure(X_test)

    assert all([type(x) == np.bool_ for x in results2])

def test_Kappa_constructor():
    ad = DAIndexKappa(dist_metric='euclidean')
    assert ad.dist_metric == 'euclidean'

    with pytest.raises(AssertionError) as e_info:
        ad = DAIndexGamma(dist_metric='asdkjnas')
    
def test_Kappa_fit():
    ad = DAIndexKappa()
    ad.fit(X_train)
    
    dist, _ = ad.tree.query(X_train, k=ad.k + 1)
    dist_mean = np.sum(dist, axis=1) / ad.k
    dist_sorted = np.sort(dist_mean)
    idx = int(np.floor(ad.ci * len(X_train)))
    
    assert ad.threshold <= dist_sorted[len(X_train)- 1] and ad.threshold >= dist_sorted[idx - 1]
    
def test_Kappa_measure():
    ad = DAIndexKappa(clf=classifier)
    ad.fit(X_train)
    results = ad.measure(X_train)
    
    assert all([type(x) == np.bool_ for x in results])
    
    results2 = ad.measure(X_test)
    
    assert all([type(x) == np.bool_ for x in results2])   

def test_Delta_constructor():
    ad = DAIndexDelta(dist_metric='euclidean')
    assert ad.dist_metric == 'euclidean'

    with pytest.raises(AssertionError) as e_info:
        ad = DAIndexGamma(dist_metric='asdkjnas')

def test_Delta_fit():
    ad = DAIndexDelta()
    ad.fit(X_train)
    
    dist, _ = ad.tree.query(X_train, k=ad.k + 1)
    dist_mean = np.sum(dist, axis=1) / ad.k
    dist_sorted = np.sort(dist_mean)
    idx = int(np.floor(ad.ci * len(X_train)))
    
    assert ad.threshold <= dist_sorted[len(X_train)- 1] and ad.threshold >= dist_sorted[idx - 1]
    
def test_Delta_measure():
    ad = DAIndexDelta(clf=classifier)
    ad.fit(X_train)
    results = ad.measure(X_train)
    
    assert all([type(x) == np.bool_ for x in results])
    
    results2 = ad.measure(X_test)
    
    assert all([type(x) == np.bool_ for x in results2])

def test_ad_save():
    path = os.path.join(os.getcwd(), "tests//save_files")
    file_path = os.path.join(path, "test_save_ad")
    ad = DAIndexGamma(clf=classifier, dist_metric="seuclidean")
    ad.save(file_path)

    assert os.path.exists(file_path)


def test_ad_load():
    path = os.path.join(os.getcwd(), "tests//save_files")
    file_path = os.path.join(path, "test_save_ad")

    ad = DAIndexGamma()
    loaded_ad = ad.load(file_path)

    assert loaded_ad.dist_metric == "seuclidean"
    assert isinstance(loaded_ad.clf, RandomForestClassifier)
