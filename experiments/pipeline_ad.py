import os
import pickle
from pathlib import Path

# External
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc

# Local
from adad.distance import DAIndexDelta, DAIndexGamma, DAIndexKappa
from adad.evaluate import (cumulative_accuracy, permutation_auc,
                           predictiveness_curves, roc_ad,
                           sensitivity_specificity)
from adad.utils import create_dir, maccs2binary, open_json, to_json

N_ESTIMATORS = 200
SEED = np.random.randint(0, 999999)


def run_pipeline(dataset, cv_train, cv_test, Classifier, clf_params,
                 ApplicabilityDomain, ad_params, dataname, path_outputs):
    y = dataset['y'].to_numpy().astype(int)
    X = dataset.drop(['y'], axis=1).to_numpy().astype(np.float32)
    n_samples = X.shape[0]

    for col in cv_train.columns:
        idx_train = cv_train[col].dropna(axis=0).to_numpy().astype(int)
        idx_test = cv_test[col].dropna(axis=0).to_numpy().astype(int)
        assert len(idx_train) + len(idx_test) == n_samples
        assert not np.all(np.isin(idx_train, idx_test))

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # Train classifier
        clf = Classifier(**clf_params)  # TODO: Save `clf_params`
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_train = clf.score(X_train, y_train)  # TODO: Save this!
        acc_test = np.mean(y_pred == y_test)  # TODO: Save this!
        print('Accuracy Train: {:.2f}% Test: {:.2f}%'.format(
            acc_train * 100, acc_test * 100))

        # Train Applicability Domain
        # TODO: Save `ad_params`
        ad_params['clf'] = clf
        ad = ApplicabilityDomain(**ad_params)
        ad.fit(X_train, y_train)

        dist_measure = ad.measure(X_test)  # TODO: Save this!

        sensitivity, specificity = sensitivity_specificity(y_test, y_pred)  # TODO: Save this!
        cumulative_acc, cumulative_rate = cumulative_accuracy(y_test, y_pred, dist_measure)  # TODO: Save this!
        auc_significance, _ = permutation_auc(y_test, y_pred, dist_measure)  # TODO: Save this!
        fpr, tpr = roc_ad(y_test, y_pred, dist_measure)  # TODO: Save this!
        auc_roc = auc(fpr, tpr)  # TODO: Save this!
        percentile, err_rate = predictiveness_curves(y_test, y_pred, dist_measure, n_quantiles=50)  # TODO: Save this!


if __name__ == '__main__':
    print(f'The seed is {SEED}')
    PATH_ROOT = Path(os.getcwd()).absolute()
    print(PATH_ROOT)

    path_maccs = os.path.join(PATH_ROOT, 'data', 'maccs')
    path_maccs_files = np.sort(
        [os.path.join(path_maccs, file) for file in os.listdir(path_maccs) if file[-4:] == '.csv'])
    print(path_maccs_files)

    path_cv = os.path.join(PATH_ROOT, 'data', 'cv')
    path_cv_train = np.sort([os.path.join(path_cv, file) for file in os.listdir(
        path_cv) if file[-13:] == '_cv_train.csv'])
    path_cv_test = np.sort([os.path.join(path_cv, file) for file in os.listdir(
        path_cv) if file[-12:] == '_cv_test.csv'])
    print('Train:')
    print(path_cv_train)
    print('Test:')
    print(path_cv_test)

    datanames = [Path(f).stem.split('_')[0] for f in path_maccs_files]
    print(datanames)

    for i in range(len(datanames)):
        dataname = datanames[i]
        n_name = len(dataname)
        assert os.path.basename(path_maccs_files[i])[:n_name] == dataname
        assert os.path.basename(path_cv_train[i])[:n_name] == dataname
        assert os.path.basename(path_cv_test[i])[:n_name] == dataname

    path_outputs = os.path.join(PATH_ROOT, 'results')

    i = 0
    dataname = datanames[i]
    path_data = path_maccs_files[i]
    path_idx_train = path_cv_train[i]
    path_idx_test = path_cv_test[i]

    df = pd.read_csv(path_data)
    cv_train = pd.read_csv(path_idx_train, dtype=pd.Int64Dtype())
    cv_test = pd.read_csv(path_idx_test, dtype=pd.Int64Dtype())

    rfc = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)
    ad = DAIndexGamma(clf=rfc, dist_metric='jaccard')
    run_pipeline(df,
                 cv_train, cv_test,
                 RandomForestClassifier,
                 {'n_estimators': N_ESTIMATORS, 'random_state': SEED},
                 DAIndexGamma,
                 {'dist_metric': 'jaccard'},
                 dataname,
                 path_outputs)
