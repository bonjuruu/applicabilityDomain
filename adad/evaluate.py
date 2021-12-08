import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import QuantileTransformer
import pandas as pd


def sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity (True Positive Rate TPR) and specificity (True 
    Negative Rate TNR)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if tp + fn != 0 else 0
    specificity = tn / (tn + fp) if tn + fp != 0 else 0
    return sensitivity, specificity


def cumulative_accuracy(y_true, y_pred, dist_measure):
    """Compute cumulative accuracy based on distance measures.

    Returns
    -------
    cumulative_acc: list
        Cumulative Accuracy
    cumulative_rate: list
        The rate the samples that are used to compute `cumulate_acc`.
    """
    assert (
        y_true.shape == y_pred.shape and y_true.shape == dist_measure.shape
    ), "True labels, predictions and distance measures must have same shape."

    idx_sorted = np.argsort(dist_measure)
    y_true_sorted = y_true[idx_sorted]
    y_pred_sorted = y_pred[idx_sorted]
    n_sample = len(y_true)
    cumulative_acc = np.zeros(n_sample)
    for i in range(n_sample):
        corrects = y_true_sorted[0: i + 1] == y_pred_sorted[0: i + 1]
        cumulative_acc[i] = np.mean(corrects)
    cumulative_rate = np.arange(1, n_sample + 1) / n_sample
    return cumulative_acc, cumulative_rate


def roc_ad(y_true, y_pred, dist_measure):
    """Compute Receiver Operating Characteristic (ROC) based on Applicability
    Domain distance.

    Parameters:
    -----------
    y_true: list
        True labels
    y_pred: list
        Predicted labels; Same shape as `y_true`
    dist_measure: list
        Distance measures from Applicability Domain; Same shape as `y_true`. 
        Lower value indicates the sample is within the AD.

    Returns
    -------
    fpr: list
        False Positive Rate for AD based on the distance measure
    tpr: list
        True Positive Rate for AD based on the distance measure
    """
    assert (
        y_true.shape == y_pred.shape and y_true.shape == dist_measure.shape
    ), "True labels, predictions and distance measures must have same shape."

    y_err = np.array((y_true != y_pred), dtype=int)
    fpr, tpr, _ = roc_curve(y_err, dist_measure, pos_label=1)
    return fpr, tpr


def save_roc(y_true, y_score, path, title=None, fontsize=14, figsize=(8, 8)):
    """Plot and save ROC curve.
    The ROC plot contains multiple results for comparision. 
    
    Parameters:
    -----------
    y_true: list:
        (m, n) matrix of true labels
    y_score: list:
        (m, n) matrix of scores
    path: string:
        Path to save ROC curve
    title: string:
        Title for ROC curve
    fontsize: int:
        Font size for graph
    figsize: tuple:
        (x, y) of the size of the graph
    """
    assert y_true.shape == y_score.shape

    num_rows, num_cols = y_true.shape
    results = pd.DataFrame(columns=['fpr', 'tpr', 'auc'])
    
    for row in range(num_rows):
        fpr, tpr, _ = roc_curve(y_true[row], y_score[row])
        roc_auc = auc(fpr, tpr)
        results = results.append({'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}, ignore_index=True)
    
    plt.rcParams["font.size"] = fontsize
    _, ax = plt.subplots(figsize=figsize)
    ax.tick_params(labelsize=fontsize)
    
    for i in range(results.shape[0]):
        ax.plot(results.loc[i]['fpr'], results.loc[i]['tpr'], label="AUC={:.3f}".format(results.loc[i]['auc']))
    
    ax.plot([0,1], [0,1], color='orange', linestyle='--')
    
    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xlabel("False Positive Rate")
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_ylabel("True Positive Rate")
    
    ax.legend(loc='lower right')
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def permutation_auc(y_true, y_pred, dist_measure, n_permutation=10000):
    """Computing randomized AUC value via permutation tests. This AUC result 
    indicates the baseline of AUC when distance measures are assigned to random
    values.

    Parameters:
    -----------
    y_true: list: 
        True labels
    y_pred: list
        Predicted labels; Same shape as `y_true`
    dist_measure: list
        Distance measures from Applicability Domain; Same shape as `y_true`.
        Lower value indicates the sample is within the AD.
    n_permutation: int, default=10000
        Number of permutations.

    Returns:
    --------
    significance_val: float
        95th percentile of permutation AUC value. If an algorithm reports an AUC
        value which is no better than this, it indicates the results from the 
        algorithm is no better than without AD (Insignificant).
    auc_perm: list
        AUC values for each permutation.
    """
    assert (
        y_true.shape == y_pred.shape and y_true.shape == dist_measure.shape
    ), "True labels, predictions and distance measures must have same shape."

    idx_pred0 = np.where(y_pred == 0)[0]
    idx_pred1 = np.where(y_pred == 1)[0]

    n_true0 = len(np.where(y_true == 0)[0])
    n_true1 = len(np.where(y_true == 1)[0])

    idx_true0 = np.where(y_true == 0)[0]

    # create array for permuted DM measure
    dist_measures_perm = np.zeros(len(dist_measure))
    auc_perm = np.zeros(n_permutation)

    for i in range(n_permutation):
        # randomly permute class indices based on predicted values
        idx_pred0_perm = np.random.permutation(idx_pred0)
        idx_pred1_perm = np.random.permutation(idx_pred1)

        # assign permuted DM measures
        dist_measures_perm[idx_pred0] = dist_measure[idx_pred0_perm]
        dist_measures_perm[idx_pred1] = dist_measure[idx_pred1_perm]

        # find rank of DM after permutations
        ranks = rankdata(dist_measures_perm)
        # sum of ranks for class 1
        s1 = np.sum(ranks[idx_true0])
        # calculate auc from sum of ranks and class population
        auc = (s1 - (n_true0 * (n_true0 + 1) / 2)) / (n_true0 * n_true1)
        auc_perm[i] = auc

    significance_val = np.percentile(auc_perm, 95)
    return significance_val, auc_perm


def predictiveness_curves(y_true, y_pred, dist_measure, n_quantiles=100):
    """Compute Predictiveness Curves for Applicability Domain. The X-axis is the
    percentile of the distance measure, and Y-axis is the error rate.

    Parameters:
    -----------
    y_true: list: 
        True labels
    y_pred: list
        Predicted labels; Same shape as `y_true`
    dist_measure: list
        Distance measures from Applicability Domain; Same shape as `y_true`.
        Lower value indicates the sample is within the AD.
    n_quantiles: int, default=100
        The number of the percentiles that the function will return. The value
        must smaller than the number of samples in the data, e.g. len(y_true).

    Returns:
    --------
    percentile: list
        The percentile of the distance measure
    error_rate: list
        The coresponding error rate at the percentile
    """
    assert n_quantiles <= len(y_true), \
        f'n_quantiles must smaller than {len(y_true)}. Got {n_quantiles}'
    assert (
        y_true.shape == y_pred.shape and y_true.shape == dist_measure.shape
    ), "True labels, predictions and distance measures must have same shape."

    transformer = QuantileTransformer(n_quantiles=n_quantiles)
    dm_quantile = transformer.fit_transform(dist_measure.reshape(-1, 1))
    dm_quantile = dm_quantile.reshape(-1)

    percentile = np.linspace(0, 1, n_quantiles + 1)[1:]

    y_err = np.array((y_true != y_pred), dtype=int)
    error_rate = np.zeros_like(percentile)
    for i, p in enumerate(percentile):
        idx = np.where(dm_quantile <= p)
        err = np.mean(y_err[idx]) if len(idx) > 0 else 0
        error_rate[i] = err
    return percentile, error_rate
