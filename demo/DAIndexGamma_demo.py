import os
from pathlib import Path
import time

# External
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, auc

# Local
from adad.distance import DAIndexGamma
from adad.evaluate import (cumulative_accuracy, permutation_auc,
                           predictiveness_curves, roc_ad,
                           sensitivity_specificity)
from adad.utils import create_dir, to_json, set_seed, time2str

N_ESTIMATORS = 200


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
    ax.legend(loc="lower right")
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Error Rate')
    ax.set_title(f'{dataname} - Predictiveness Curves (PC)')
    plt.tight_layout()
    plt.savefig(path_output, dpi=300)


def run_pipeline(dataset, cv_train, cv_test, Classifier, clf_params,
                 ApplicabilityDomain, ad_params, dataname, path_outputs):
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
        assert len(idx_train) + len(idx_test) == n_samples
        assert not np.all(np.isin(idx_train, idx_test))

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # Train classifier
        clf = Classifier(**clf_params)
        clf_params = clf.get_params()
        clf.fit(X_train, y_train)
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

        percentile, err_rate = predictiveness_curves(y_test, y_pred, dist_measure, n_quantiles=50)
        df_pc[f'{col}_percentile'] = pd.Series(percentile, dtype=float)
        df_pc[f'{col}_err_rate'] = pd.Series(err_rate, dtype=float)

    # Save results
    path_outputs = os.path.join(path_outputs, f'{Classifier.__name__}_{ApplicabilityDomain.__name__}')
    create_dir(path_outputs)

    # Using the same parameters for all CV
    to_json(clf_params, os.path.join(path_outputs, f'{dataname}_{Classifier.__name__}.json'))

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


if __name__ == '__main__':
    SEED = np.random.randint(0, 999999)
    set_seed(SEED)

    print(f'The seed is {SEED}')
    PATH_ROOT = Path(os.getcwd()).absolute()
    print(PATH_ROOT)

    path_maccs = os.path.join(PATH_ROOT, 'data', 'maccs')
    path_maccs_files = np.sort([os.path.join(path_maccs, file) for file in os.listdir(path_maccs) if file[-4:] == '.csv'])
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

    path_outputs = os.path.join(PATH_ROOT, 'demo')

    i = 0
    dataname = datanames[i]
    path_data = path_maccs_files[i]
    path_idx_train = path_cv_train[i]
    path_idx_test = path_cv_test[i]

    df = pd.read_csv(path_data)
    cv_train = pd.read_csv(path_idx_train, dtype=pd.Int64Dtype())
    cv_test = pd.read_csv(path_idx_test, dtype=pd.Int64Dtype())

    time_start = time.perf_counter()
    run_pipeline(df, cv_train, cv_test,
                 RandomForestClassifier,
                 {'n_estimators': N_ESTIMATORS, 'random_state': SEED},
                 DAIndexGamma,
                 {'dist_metric': 'jaccard'},
                 dataname,
                 path_outputs)
    time_elapsed = time.perf_counter() - time_start
    print(f'Total run time: {time2str(time_elapsed)}')
