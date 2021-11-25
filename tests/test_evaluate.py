import os
import pandas as pd
import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.evaluate import sensitivity_specificity, save_roc
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

ad = DAIndexGamma(clf=classifier)
ad.fit(X_train)

def test_sensitivity_specificity():
    y_pred, idx = ad.predict(X_test)
    sensitivity, specificity = sensitivity_specificity(y_test[idx], y_pred)
    
    assert sensitivity > 0.5
    assert specificity > 0.5
    
def test_save_roc():
    path = os.path.join(os.getcwd(), "tests//results") 
    file_path = os.path.join(path, "test_result_CYP1A2.png")
    
    y_proba, idx = ad.predict_proba(X_test)
    
    save_roc(y_test[idx], y_proba, file_path, title="CYP1A2 ROC Curve")
    
    assert os.path.exists(file_path)