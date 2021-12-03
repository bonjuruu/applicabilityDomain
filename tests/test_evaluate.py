import os
import pandas as pd
import numpy as np
import pytest
from numpy.random import randint


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.evaluate import acc_vs_removed, calculate_auc, sensitivity_specificity, save_roc
from adad.distance import DAIndexGamma

SEED = 348
np.random.seed(348)

X = randint(5, size=(20, 5))
y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

classifier = RandomForestClassifier(random_state=SEED)
classifier.fit(X_train, y_train)

ad = DAIndexGamma(clf=classifier)
ad.fit(X_train)


def test_sensitivity_specificity():
    y_pred, idx = ad.predict(X_test)
    sensitivity, specificity = sensitivity_specificity(y_test[idx], y_pred)

    assert sensitivity >= 0 and sensitivity <= 1
    if str(specificity) == "nan":
        pass
    assert specificity >= 0 and specificity <= 1


def test_save_roc():
    path = os.path.join(os.getcwd(), "tests//results")
    file_path = os.path.join(path, "test_result.png")

    y_proba, idx = ad.predict_proba(X_test)

    save_roc(y_test[idx], y_proba, file_path, title="Test ROC Curve")

    assert os.path.exists(file_path)


def test_acc_vs_remove():
    y_pred, idx = ad.predict(X_test)
    acc_vs_removed(y_test, y_pred, y_pred)


def test_calculate_auc():
    y_pred, idx = ad.predict(X_test)
    y_true = y_test[idx]
    y_proba, idx = ad.predict_proba(X_test)

    sig_value, perm_AUC = calculate_auc(y_true, y_pred, y_proba, 1000)
    assert len(perm_AUC) == 1000
    assert sig_value >= 0 and sig_value <= 1
