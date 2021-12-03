import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.adversarial import SklearnRegionBasedClassifier
from adad.utils import category2code, get_range, drop_redundant_col


SEED = 1234
TEST_RATIO = 0.4

# Parameters for Region-Based Classifier
R_MIN = 0.08
R_MAX = 0.1
STEP_SIZE = 1
EPS = 0.01


def run_dist():
    files_path = os.path.join(os.getcwd(), 'data', 'maccs')
    dataset_files = [os.path.join(files_path, file)
                     for file in os.listdir(files_path)]
    data_path = dataset_files[0]
    print(f'Read from: {data_path}')

    df = pd.read_csv(data_path)
    print(df.head())

    y = df['y'].to_numpy()
    df_X = df.drop(['y'], axis=1)
    df_X = drop_redundant_col(df_X)
    X = category2code(df_X).to_numpy()
    x_min, x_max = get_range(X)

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=SEED
    )
    print(X_train.shape, X_test.shape)

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    print(f'Train score: {score * 100:.2f}%')

    score = clf.score(X_test, y_test)
    print(f'[Without AD] Score: {score * 100:.2f}%')

    ad = SklearnRegionBasedClassifier(
        clf,
        sample_size=200,
        x_min=x_min,
        x_max=x_max,
        r_min=R_MIN,
        r_max=R_MAX,
        step_size=STEP_SIZE,
        eps=EPS,
        data_type='discrete',
        verbose=1)
    ad.fit(X_train, y_train)

    avg_dist_train = np.mean(ad.measure(X_train))
    print(f'Avg train dist: {avg_dist_train:.3f}')

    avg_dist_test = np.mean(ad.measure(X_test))
    print(f'Avg test dist: {avg_dist_test:.3f}')


if __name__ == '__main__':
    run_dist()
