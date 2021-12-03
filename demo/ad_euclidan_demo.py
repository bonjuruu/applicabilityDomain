import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.distance import DAIndexDelta, DAIndexGamma, DAIndexKappa

SEED = 1234
N_TREE = 100


def run_dist():
    files_path = os.path.join(os.getcwd(), 'data', 'maccs')
    dataset_files = [os.path.join(files_path, file)
                     for file in os.listdir(files_path)]
    data_path = dataset_files[0]
    print(f'Read from: {data_path}')

    df = pd.read_csv(data_path)
    y = df['y'].to_numpy()
    X = df.drop(['y'], axis=1).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print('Train shape: ', X_train.shape)
    print(' Test shape: ', X_test.shape)

    model = RandomForestClassifier(n_estimators=N_TREE)
    model.fit(X_train, y_train)

    for AD in [DAIndexGamma, DAIndexKappa, DAIndexDelta]:
        print(f'Running {AD.__name__}...')
        ad = AD(model)
        ad.fit(X_train)

        score = model.score(X_train, y_train)
        print(f'Train score: {score * 100:.2f}%')

        score = model.score(X_test, y_test)
        print(f'[Without AD] Score: {score * 100:.2f}%')

        avg_dist_train = np.mean(ad.dist_measure_train)
        print(f'Avg train dist: {avg_dist_train:.3f}')

        avg_dist_test = np.mean(ad.measure(X_test))
        print(f'Avg test dist: {avg_dist_test:.3f}')
        print(f"--------------------------------------------------------------")


if __name__ == '__main__':
    run_dist()
