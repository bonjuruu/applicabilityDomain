import os
import pandas as pd
import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.distance import DAIndexGamma

SEED = 348

files_path = os.path.join(os.getcwd(), 'data', 'maccs')
dataset_files = [os.path.join(files_path, file) for file in os.listdir(files_path)]
file_datapath = dataset_files[3]

df = pd.read_csv(file_datapath)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

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
    
    assert ad.threshold < dist_sorted[len(X_train)- 1] and ad.threshold > dist_sorted[idx - 1]
    
def test_Gamma_measure():
    ad = DAIndexGamma(clf=classifier)
    ad.fit(X_train)
    results = ad.measure(X_train)
    print(type(results[0]))
    
    assert all(results) == False
    assert all([type(x) == np.bool_ for x in results])
    
    results2 = ad.measure(X_test)
    
    assert all([type(x) == np.bool_ for x in results2])

    
def test_Gamma_predict():
    ad = DAIndexGamma(clf=classifier)
    ad.fit(X_train)
    predict, indexs = ad.predict(X_test)
    p_size, i_size = len(predict), len(indexs)
    
    assert i_size > 0 and i_size < len(X_test)
    assert p_size > 0 and p_size < len(X_test)
    assert i_size == p_size
    
    
def test_Gamma_predict_proba():
    ad = DAIndexGamma(clf=classifier)
    ad.fit(X_train)
    predict, indexs = ad.predict_proba(X_test)
    p_size, i_size = len(predict), len(indexs)
    
    assert i_size > 0 and i_size < len(X_test)
    assert p_size > 0 and p_size < len(X_test)
    assert i_size == p_size
    
def test_Gamma_score():
    ad = DAIndexGamma(clf=classifier)
    ad.fit(X_train)
    score = ad.score(X_test, y_test)
    
    assert score > 0.5
    
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