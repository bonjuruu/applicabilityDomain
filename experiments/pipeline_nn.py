import argparse
import json
import os
import time
from pathlib import Path

# External
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import RocCurveDisplay, auc
from torch.utils.data import DataLoader, TensorDataset

# Local
from adad.bounding_box import PCABoundingBox
from adad.distance import DAIndexDelta, DAIndexGamma, DAIndexKappa
from adad.evaluate import (cumulative_accuracy, permutation_auc,
                           predictiveness_curves, roc_ad,
                           sensitivity_specificity)
from adad.probability import ProbabilityClassifier
from adad.torch_utils import evaluate, predict, predict_proba, train_model
from adad.utils import create_dir, open_json, set_seed, time2str, to_json

AD_NAMES = ['gamma', 'kappa', 'delta', 'boundingbox', 'prob']


class NeuralNet(nn.Module):
    """A simple fullly-connected neural network with 1 hidden-layer"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=2):
        super(NeuralNet, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class NNClassifier:
    def __init__(self,
                 input_dim=167,
                 hidden_dim=512,
                 output_dim=2,
                 batch_size=128,
                 max_epochs=300,
                 lr=1e-3,
                 device='cuda'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = torch.device(device)

        self.clf = NeuralNet(input_dim, hidden_dim, output_dim).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    def fit(self, X, y):
        dataset = TensorDataset(
            torch.from_numpy(X).type(torch.float32),
            torch.from_numpy(y).type(torch.int64)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        train_model(self.clf, dataloader, self.optimizer, self.loss_fn,
                    self.device, self.max_epochs)

    def predict(self, X):
        return predict(X, self.clf, self.device, self.batch_size)

    def predict_proba(self, X):
        return predict_proba(X, self.clf, self.device, self.batch_size)

    def score(self, X, y):
        dataset = TensorDataset(
            torch.from_numpy(X).type(torch.float32),
            torch.from_numpy(y).type(torch.int64)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        acc, _ = evaluate(dataloader, self.clf, self.loss_fn, self.device)
        return acc

    def save(self, path):
        create_dir(path)
        torch.save(self.clf.state_dict(), os.path.join(path, 'NeuralNet.torch'))
        params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'lr': self.lr,
            'device': str(self.device),
        }
        to_json(params, os.path.join(path, 'NNClassifier.json'))

    def load(self, path):
        params = open_json(os.path.join(path, 'NNClassifier.json'))
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.batch_size = params['batch_size']
        self.max_epochs = params['max_epochs']
        self.lr = params['lr']
        self.device = torch.device(params['device'])
        self.clf.load_state_dict(
            torch.load(
                os.path.join(path, 'NeuralNet.torch'),
                map_location=self.device
            )
        )
        self.optimizer = torch.optim.SGD(self.clf.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)


def plot_ca(df_cum_acc, path_output, dataname, n_cv=5):
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=(8, 8))
    mean_x = np.linspace(0, 1, 100)
    ys = []
    for i in range(1, n_cv + 1):
        x = df_cum_acc[f'cv{i}_rate']
        y = df_cum_acc[f'cv{i}_acc']
        ax.plot(x, y, alpha=0.3, lw=1, label=f'Fold{i}')

        interp_acc = np.interp(mean_x, x, y)
        ys.append(interp_acc)

    # Draw mean value
    mean_y = np.mean(ys, axis=0)
    ax.plot(mean_x, mean_y, color='b', lw=2, alpha=0.8, label='Mean')

    # Fill standard error area
    std_y = np.std(ys, axis=0)
    y_upper = np.minimum(mean_y + std_y, 1)
    y_lower = np.maximum(mean_y - std_y, 0)
    ax.fill_between(mean_x, y_lower, y_upper, color='b', alpha=0.1,
                    label="$\pm$ 1 std. dev.")

    ax.set(xlim=[-0.01, 1.01], ylim=[0.6, 1.01])
    ax.legend(loc="lower right")
    ax.set_xlabel('Cumulative Rate')
    ax.set_ylabel('Cumulative Accuracy (%)')
    ax.set_title(f'{dataname} - Cumulative Accuracy')
    plt.tight_layout()
    plt.savefig(path_output, dpi=300)


def plot_roc(df_roc, roc_aucs, path_output, dataname, n_cv=5):
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=(8, 8))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # Draw each ROC curve
    for i in range(1, n_cv + 1):
        fpr = df_roc[f'cv{i}_fpr']
        tpr = df_roc[f'cv{i}_tpr']
        roc_auc = roc_aucs[i - 1]
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot(ax=ax, alpha=0.3, lw=1, label=f"ROC Fold{i} (AUC={roc_auc:.2f})")

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    # Draw mean value
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8,
            label=f"Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})")

    # Fill standard error area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='b', alpha=0.1,
                    label="$\pm$ 1 std. dev.")

    ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax.legend(loc="lower right")
    ax.set_title(f'{dataname} - ROC Curve')
    plt.tight_layout()
    plt.savefig(path_output, dpi=300)


def plot_pc(df, path_output, dataname, n_cv=5):
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=(8, 8))
    mean_x = np.linspace(0, 1, 100)
    ys = []
    for i in range(1, n_cv + 1):
        x = df[f'cv{i}_percentile']
        y = df[f'cv{i}_err_rate']
        ax.plot(x, y, alpha=0.3, lw=1, label=f'Fold{i}')

        interp_acc = np.interp(mean_x, x, y)
        ys.append(interp_acc)

    # Draw mean value
    mean_acc = np.mean(ys, axis=0)
    ax.plot(mean_x, mean_acc, color='b', lw=2, alpha=0.8, label='Mean')

    # Fill standard error area
    std_y = np.std(ys, axis=0)
    y_upper = np.minimum(mean_acc + std_y, 1)
    y_lower = np.maximum(mean_acc - std_y, 0)
    ax.fill_between(mean_x, y_lower, y_upper, color='b', alpha=0.1,
                    label="$\pm$ 1 std. dev.")

    ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 0.6])
    ax.legend(loc="upper left")
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Error Rate')
    ax.set_title(f'{dataname} - Predictiveness Curves (PC)')
    plt.tight_layout()
    plt.savefig(path_output, dpi=300)


def run_pipeline_nn(dataset,
                    cv_train,
                    cv_test,
                    ApplicabilityDomain,
                    ad_params,
                    dataname,
                    path_outputs):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Running on CPU!')

    y = dataset['y'].to_numpy().astype(int)
    X = dataset.drop(['y'], axis=1).to_numpy().astype(np.float32)
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
    for col in cv_train.columns:
        idx_train = cv_train[col].dropna(axis=0).to_numpy().astype(int)
        idx_test = cv_test[col].dropna(axis=0).to_numpy().astype(int)
        assert len(idx_train) + len(idx_test) == n_samples, 'Indices does not match samples!'
        assert not np.all(np.isin(idx_train, idx_test)), 'Training set and test set have overlap!'

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # Train classifier
        clf = NNClassifier(device=device)
        clf.fit(X_train, y_train)
        clf.save(os.path.join(path_outputs, f'{dataname}_{col}_NeuralNet'))

        y_pred = clf.predict(X_test)
        acc_train = clf.score(X_train, y_train)
        accs_train.append(acc_train)
        acc_test = np.mean(y_pred == y_test)
        accs_test.append(acc_test)
        print('Accuracy Train: {:.2f}% Test: {:.2f}%'.format(acc_train * 100, acc_test * 100))

        # Train Applicability Domain
        ad_params['clf'] = clf
        ad = ApplicabilityDomain(**ad_params)
        ad.fit(X_train, y_train)

        dist_measure = ad.measure(X_test)
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
        print(f'AUC: {auc_roc:.3f} Permutation AUC: {auc_significance:.3f}')

        percentile, err_rate = predictiveness_curves(y_test, y_pred, dist_measure, n_quantiles=50)
        df_pc[f'{col}_percentile'] = pd.Series(percentile, dtype=float)
        df_pc[f'{col}_err_rate'] = pd.Series(err_rate, dtype=float)

    # Save AD
    ad_params.pop('clf', None)
    to_json(ad_params, os.path.join(path_outputs, f'{dataname}_{ApplicabilityDomain.__name__}.json'))

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


def get_ad(adname):
    assert adname in AD_NAMES, f'Received AD: {adname}'
    if adname == 'gamma':
        ad = DAIndexGamma
    elif adname == 'kappa':
        ad = DAIndexKappa
    elif adname == 'delta':
        ad = DAIndexDelta
    elif adname == 'boundingbox':
        ad = PCABoundingBox
    elif adname == 'prob':
        ad = ProbabilityClassifier
    else:
        ad = None
    return ad


def run_exp(path_input,
            path_output,
            dataname,
            adname,
            ad_params,
            random_state):
    if random_state is None:
        random_state = np.random.randint(0, 999999)
    set_seed(random_state)
    print(f'Set random_state to: {random_state}')

    AD = get_ad(adname)

    path_outputs = os.path.join(path_output, f'{NeuralNet.__name__}_{AD.__name__}')
    create_dir(path_outputs)
    print(f'Save results to: {path_outputs}')
    path_random_state = os.path.join(path_outputs, f'{dataname}_random_state.json')
    to_json({'random_state': random_state}, path_random_state)

    # Load data
    path_maccs = os.path.join(path_input, 'maccs')
    path_maccs_files = np.sort([os.path.join(path_maccs, file) for file in os.listdir(path_maccs) if file[-4:] == '.csv'])
    path_data = [file for file in path_maccs_files if dataname in file][0]
    df = pd.read_csv(path_data)

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

    # BoundingBox requires random_state
    if adname == 'boundingbox':
        ad_params['random_state'] = random_state

    time_start = time.perf_counter()
    run_pipeline_nn(df, cv_train, cv_test,
                    AD,
                    ad_params,
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
    parser.add_argument('--ad', type=str, required=True,
                        help='Applicability Domain name.')
    parser.add_argument('--adArg', type=str, required=True,
                        help='AD\'s parameters.')
    parser.add_argument('-r', '--random_state', type=int, help='Random state.')
    args = parser.parse_args()
    print('Received args:', args)
    path_input = Path(args.input).absolute()
    path_output = Path(args.output).absolute()
    dataname = args.data
    adname = args.ad
    ad_params = json.loads(args.adArg)
    random_state = args.random_state

    run_exp(path_input,
            path_output,
            dataname,
            adname,
            ad_params,
            random_state)
