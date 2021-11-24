import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.evaluate import sensitivity_specificity, save_roc
from adad.distance import DAIndexGamma


SEED = 1234


def run_dist():
    files_path = os.path.join(os.getcwd(), 'data', 'maccs')
    dataset_files = [os.path.join(files_path, file)
                     for file in os.listdir(files_path)]
    data_path = dataset_files[0]
    print(f'Read from: {data_path}')

    df = pd.read_csv(data_path)
    print(df.head())

    y = df['y'].to_numpy()
    X = df.drop(['y'], axis=1).to_numpy()

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(X_train.shape, X_test.shape)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    ad = DAIndexGamma(model)
    ad.fit(X_train)

    pred, idx = ad.predict(X_test)
    print(f'Pass rate: {len(idx) /len(X_test) * 100:.2f}%')

    score = model.score(X_train, y_train) * 100
    print(f'Train score: {score:.2f}%')

    score = model.score(X_test, y_test) * 100
    print(f'[Without AD] Score: {score:.2f}%')

    score = ad.score(X_test, y_test) * 100
    print(f'[With AD] Score: {score:.2f}%')

    sensitivity, specificity = sensitivity_specificity(y_test[idx], pred)
    print(f'Sensitivity: {sensitivity:.3f}')
    print(f'Specificity: {specificity:.3f}')

    proba, idx = ad.predict_proba(X_test)
    path_roc = os.path.join(os.getcwd(), 'results', 'ames_roc.pdf')
    save_roc(y_test[idx], proba, path_roc, title='Ames ROC Curve')


if __name__ == '__main__':
    run_dist()
