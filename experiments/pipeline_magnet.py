import argparse
import json
import os
import time
from pathlib import Path

# External
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Local
from adad.adversarial.feature_squeezing import SklearnFeatureSqueezing
from adad.adversarial.magnet import AutoEncoder, Magnet
from adad.evaluate import (cumulative_accuracy, permutation_auc,
                           predictiveness_curves, roc_ad,
                           sensitivity_specificity)
from adad.plot import plot_ca, plot_pc, plot_roc
from adad.torch_utils import NNClassifier
from adad.utils import create_dir, set_seed, time2str, to_json

CLASSIFIER_NAMES = ['rf', 'svm', 'knn', 'nn']
N_ENCODERS = 5


def plot_autoencoder_losses(df, path, title, fontsize=16, figsize=(8, 6)):
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def get_nn_params(X, clf_params):
    """Parameters for Neural Network need to generate dynamically."""
    clf_params = dict(clf_params)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Running on CPU!')
    clf_params['device'] = device

    # To handle # of features changes after dropping empty columns
    clf_params['input_dim'] = X.shape[1]
    if X.shape[1] < 128:
        # The encoded X in magnet is much smaller than MACCS descriptors
        clf_params['hidden_dim'] = 128
    return clf_params


def run_pipeline(X, y, cv_train, cv_test, Classifier, clf_params,
                 dataname, path_outputs):
    n_samples = X.shape[0]

    accs_train = []
    accs_test = []
    sensitivities = []
    specificities = []
    aucs = []
    perm_aucs = []
    df_dm = pd.DataFrame()
    df_cum_acc = pd.DataFrame()
    df_roc = pd.DataFrame()
    df_pc = pd.DataFrame()

    if Classifier.__name__ == NNClassifier.__name__:
        clf_params = get_nn_params(X, clf_params)

    for col in cv_train.columns:
        idx_train = cv_train[col].dropna(axis=0).to_numpy().astype(int)
        idx_test = cv_test[col].dropna(axis=0).to_numpy().astype(int)
        assert len(idx_train) + len(idx_test) == n_samples, 'Indices does not match samples!'
        assert not np.all(np.isin(idx_train, idx_test)), 'Training set and test set have overlap!'

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # Train classifier
        clf = Classifier(**clf_params)
        clf.fit(X_train, y_train)
        if Classifier.__name__ == NNClassifier.__name__:
            # Only save Neural Network models
            clf.save(os.path.join(path_outputs, f'{dataname}_{col}_NeuralNet'))

        y_pred = clf.predict(X_test)
        acc_train = clf.score(X_train, y_train)
        accs_train.append(acc_train)
        acc_test = np.mean(y_pred == y_test)
        accs_test.append(acc_test)
        print('[{}] Accuracy Train: {:.2f}% Test: {:.2f}%'.format(col, acc_train * 100, acc_test * 100))

        # Train Applicability Domain
        n_encoders = N_ENCODERS
        magnet = Magnet(AutoEncoder, n_encoders=n_encoders)
        path_magnet = os.path.join(path_outputs, f'{dataname}_magnet')
        if os.path.exists(os.path.join(path_magnet, 'magnet_autoencoder0.torch')):
            magnet.load(path_magnet)
        else:
            magnet.fit(X_train)
            magnet.save(path_magnet)

            # Save autoencoder losses
            autoencoder_losses = []
            for autoencoder in magnet.autoencoders:
                autoencoder_losses.append(autoencoder['losses'])
            autoencoder_losses = np.array(autoencoder_losses)
            df_autoencoder_losses = pd.DataFrame(autoencoder_losses.T, columns=[f'Autoencoder{i}' for i in range(n_encoders)])
            df_autoencoder_losses.to_csv(os.path.join(path_outputs, f'{dataname}_autoencoder_losses_{col}.csv'), index=False)
            plot_autoencoder_losses(
                df_autoencoder_losses,
                os.path.join(path_outputs, f'{dataname}_autoencoder_losses_{col}.pdf'),
                f'{dataname} Autoencoder Losses')

        # Use autoencoders to transform inputs
        Xs_encode_tr = magnet.measure(X_train)
        Xs_encode_te = magnet.measure(X_test)

        if Classifier.__name__ == NNClassifier.__name__:
            fs_clf_params = get_nn_params(Xs_encode_tr[0], clf_params)
        else:
            fs_clf_params = dict(clf_params)
        fs_clf_params = [fs_clf_params] * n_encoders
        fs = SklearnFeatureSqueezing(clf, Classifier, n_encoders, fs_clf_params)
        fs.fit(Xs_encode_tr, y_train)

        dist_measure = fs.measure(X_test, Xs_encode_te)

        # FS requires inverse DM!
        dist_measure = 1. / (dist_measure + 1e-6)
        df_dm[col] = pd.Series(dist_measure, dtype=float)

        sensitivity, specificity = sensitivity_specificity(y_test, y_pred)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

        cum_acc, cum_rate = cumulative_accuracy(y_test, y_pred, dist_measure)
        df_cum_acc[f'{col}_acc'] = pd.Series(cum_acc, dtype=float)
        df_cum_acc[f'{col}_rate'] = pd.Series(cum_rate, dtype=float)

        auc_significance, _ = permutation_auc(y_test, y_pred, dist_measure)
        perm_aucs.append(auc_significance)

        fpr, tpr = roc_ad(y_test, y_pred, dist_measure)
        df_roc[f'{col}_fpr'] = pd.Series(fpr, dtype=float)
        df_roc[f'{col}_tpr'] = pd.Series(tpr, dtype=float)

        auc_roc = auc(fpr, tpr)
        aucs.append(auc_roc)
        print(f'[{col}] AUC: {auc_roc:.3f} Permutation AUC: {auc_significance:.3f}')

        percentile, err_rate = predictiveness_curves(y_test, y_pred, dist_measure, n_quantiles=50)
        df_pc[f'{col}_percentile'] = pd.Series(percentile, dtype=float)
        df_pc[f'{col}_err_rate'] = pd.Series(err_rate, dtype=float)

    # Save results
    if Classifier.__name__ != NNClassifier.__name__:
        # Using the same parameters for all CV
        clf_params = clf.get_params()
        to_json(clf_params, os.path.join(path_outputs, f'{dataname}_{Classifier.__name__}.json'))

    # Save scores
    data1 = {
        'cv': cv_train.columns,
        'acc_train': np.round(accs_train, 6),
        'acc_test': np.round(accs_test, 6),
        'sensitivity': np.round(sensitivities, 6),
        'specificity': np.round(specificities, 6),
        'auc': np.round(aucs, 6),
        'perm_auc': np.round(perm_aucs, 6),
    }
    df_data1 = pd.DataFrame(data1)
    df_data1.to_csv(os.path.join(path_outputs, f'{dataname}_scores.csv'), index=False)

    # Save DM
    df_dm.to_csv(os.path.join(path_outputs, f'{dataname}_DistMeasure.csv'), index=False)

    # Save Cumulative Accuracy
    df_cum_acc.to_csv(os.path.join(path_outputs, f'{dataname}_CumulativeAccuracy.csv'), index=False)

    # Save ROC
    df_roc.to_csv(os.path.join(path_outputs, f'{dataname}_roc.csv'), index=False)

    # Save Predictiveness Curves
    df_pc.to_csv(os.path.join(path_outputs, f'{dataname}_PredictivenessCurves.csv'), index=False)

    # Plot Cumulative Acc
    path_ca_plot = os.path.join(path_outputs, f'{dataname}_ca.pdf')
    plot_ca(df_cum_acc, path_ca_plot, dataname)

    # Plot ROC
    path_roc_plot = os.path.join(path_outputs, f'{dataname}_roc.pdf')
    plot_roc(df_roc, df_data1['auc'], path_roc_plot, dataname)

    # Plot Predictiveness Curves
    path_pc_plot = os.path.join(path_outputs, f'{dataname}_pc.pdf')
    plot_pc(df_pc, path_pc_plot, dataname)


def get_clf(clfname):
    assert clfname in CLASSIFIER_NAMES, f'Received classifier: {clfname}'
    if clfname == 'rf':
        clf = RandomForestClassifier
    elif clfname == 'svm':
        clf = SVC
    elif clfname == 'knn':
        clf = KNeighborsClassifier
    elif clfname == 'nn':
        clf = NNClassifier
    else:
        clf = None
    return clf


def run_exp(path_input,
            path_output,
            dataname,
            clfname,
            clf_params,
            random_state):
    if random_state is None:
        random_state = np.random.randint(0, 999999)
    set_seed(random_state)
    print(f'Set random_state to: {random_state}')

    Classifier = get_clf(clfname)

    path_outputs = os.path.join(path_output, f'{Classifier.__name__}_{Magnet.__name__}')
    create_dir(path_outputs)
    print(f'Save results to: {path_outputs}')
    path_random_state = os.path.join(path_outputs, f'{dataname}_random_state.json')
    to_json({'random_state': random_state}, path_random_state)

    # Load data
    path_maccs = os.path.join(path_input, 'maccs')
    path_maccs_files = np.sort([os.path.join(path_maccs, file) for file in os.listdir(path_maccs) if file[-4:] == '.csv'])
    path_data = [file for file in path_maccs_files if dataname in file][0]
    df = pd.read_csv(path_data)
    y = df['y'].to_numpy().astype(int)
    X = df.drop(['y'], axis=1).to_numpy().astype(np.float32)

    # Load Cross-validation indices
    path_cv = os.path.join(path_input, 'cv')
    path_cv_train = np.sort([os.path.join(path_cv, file) for file in os.listdir(
        path_cv) if file[-13:] == '_cv_train.csv'])
    path_cv_test = np.sort([os.path.join(path_cv, file) for file in os.listdir(
        path_cv) if file[-12:] == '_cv_test.csv'])
    path_idx_train = [file for file in path_cv_train if dataname in file][0]
    path_idx_test = [file for file in path_cv_test if dataname in file][0]
    cv_train = pd.read_csv(path_idx_train, dtype=pd.Int64Dtype())
    cv_test = pd.read_csv(path_idx_test, dtype=pd.Int64Dtype())

    # RF and SVM need to set random_state
    if clfname == 'rf' or clfname == 'svm':
        clf_params['random_state'] = random_state

    time_start = time.perf_counter()
    run_pipeline(X, y,
                 cv_train, cv_test,
                 Classifier,
                 clf_params,
                 dataname,
                 path_outputs)
    time_elapsed = time.perf_counter() - time_start
    print(f'Total run time: {time2str(time_elapsed)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='./data',
                        help='Input file path.')
    parser.add_argument('-o', '--output', type=str, default='./results',
                        help='Output file path.')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Dataname.')
    parser.add_argument('--clf', type=str, required=True,
                        help='Classifier name.')
    parser.add_argument('--clfArg', type=str,
                        help='Classifier\'s parameters.')
    parser.add_argument('-r', '--random_state', type=int, help='Random state.')
    args = parser.parse_args()
    print('Received args:', args)
    path_input = Path(args.input).absolute()
    path_output = Path(args.output).absolute()
    dataname = args.data
    clfname = args.clf
    clf_params = json.loads(args.clfArg)
    random_state = args.random_state

    run_exp(path_input,
            path_output,
            dataname,
            clfname,
            clf_params,
            random_state)
