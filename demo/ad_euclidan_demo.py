import os, sys
sys.path.insert(0, os.getcwd()) 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adad.evaluate import sensitivity_specificity, save_roc, calculate_auc
from adad.distance import DAIndexGamma, DAIndexKappa, DAIndexDelta


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

    for AD in [DAIndexGamma, DAIndexKappa, DAIndexDelta]:
        print(f'Running {AD.__name__}...')
        ad = AD(model)
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

        y_true = y_test[idx]
        sig_value, perm_AUC = calculate_auc(y_true, pred, proba, 1000, SEED)
        print(f"Significance value: {sig_value:.3f}")
        print(f"--------------------------------------------------------------")


if __name__ == '__main__':
    run_dist()
