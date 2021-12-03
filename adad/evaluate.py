import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (RocCurveDisplay, auc, confusion_matrix,
                             hamming_loss, roc_curve)


def sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity and specificity"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def get_fpr_tpr(y_true, y_proba, y_pred):
    """TODO: Could you explain the expected behaviour for this method?"""
    tpr, fpr = [0.0], [0.0]

    proba_c1 = [x[1] for x in y_proba]
    idx_array = np.argsort(proba_c1)

    y_pred_sorted = np.flip(y_pred[idx_array]).tolist()
    y_true_sorted = np.flip(y_true[idx_array]).tolist()

    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(len(y_true)):
        tp += 1.0 if (y_true_sorted[i] == 1) & (y_pred_sorted[i] == 1) else 0.0
        tn += 1.0 if (y_true_sorted[i] == 0) & (y_pred_sorted[i] == 0) else 0.0
        fp += 1.0 if (y_true_sorted[i] == 0) & (y_pred_sorted[i] == 1) else 0.0
        fn += 1.0 if (y_true_sorted[i] == 1) & (y_pred_sorted[i] == 0) else 0.0

        if (fp != 0 or tn != 0):
            fpr.append(fp / (fp + tn))
        else:
            fpr.append(0.0)

        if (tp != 0 or fn != 0):
            tpr.append(tp / (tp + fn))
        else:
            tpr.append(0.0)

    fpr.sort()
    tpr.sort()
    return fpr, tpr


def create_roc_graph(fpr, tpr, path, title=None, fontsize=14, figsize=(8, 8)):
    """Plot and save ROC curve"""
    # TODO: What's the expected behaviour?

    plt.style.use('seaborn-whitegrid')
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)

    plt.rcParams["font.size"] = fontsize
    _, ax = plt.subplots(figsize=figsize)
    ax.tick_params(labelsize=fontsize)
    plt.plot(fpr, tpr, color='blue', linewidth=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='dashed',
             color='red', linewidth=2, label='random')
    if title:
        ax.set_title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def save_roc(y_true, y_score, path, title=None, fontsize=14, figsize=(8, 8)):
    """TODO: What's the difference between this and `create_roc_graph`?"""
    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1], pos_label=1)
    assert y_true.shape == y_score[:, 1].shape
    roc_auc = auc(fpr, tpr)
    plt.rcParams["font.size"] = fontsize
    _, ax = plt.subplots(figsize=figsize)
    ax.tick_params(labelsize=fontsize)
    RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def acc_vs_removed(y_true, y_pred, scores, plot=True, add=True, label=""):
    """TODO: Could you explain the expected behaviour for this method?"""
    s = np.argsort(scores)
    #acc, rem = [0], [0]
    acc_full = hamming_loss(y_true, y_pred)
    acc, rem = [], []
    for i in range(0, len(s) + 1):
        acc.append(
            (hamming_loss(y_true[s[i:]], y_pred[s[i:]]) - acc_full) / (1 - acc_full))
        rem.append(i / len(s))
    acc.append(1)
    rem.append(1)

    if plot:
        plt.step(rem, acc, label=label)
        plt.xlim([-0.05, 1.05])
        #plt.ylim([-0.05, 1.05])
        plt.xlabel("Ratio compounds removed")
        plt.ylabel("Hamming_Accuracy")
        if not add:
            plt.legend()
            plt.show()
            plt.savefig(dpi=300)

    # output area under curve
    return np.sum(np.array(acc)[1:] * (np.array(rem)[1:] - np.array(rem)[:-1]))


def calculate_auc(y_true, y_pred, dist_measure, n_permutation):
    """Calculates AUC via permutation test

    Parameters:
    -----------
    y_true: list: 
        True labels
    y_pred: list
        Predicted labels; Same shape as `y_true`
    dist_measure: list
        Distance measures from Applicability Domain; Same shape as `y_true`
    n_permutation: int
        Number of permutations.

    Returns:
    --------
    significance_val: float
        95th percentile of permutation distribution.
    auc_perm: list
        AUC values for each permutation.
    """

    idx_pred0 = np.where(y_pred == 0)[0]
    idx_pred1 = np.where(y_pred == 1)[0]

    n_pred0 = len(idx_pred0)
    n_pred1 = len(idx_pred1)
    n_true0 = len(np.where(y_true == 0)[0])
    n_true1 = len(np.where(y_true == 1)[0])

    idx_true0 = np.where(y_true == 0)[0]
    
    # create array for permuted DM measure
    dist_measures_perm = np.zeros(len(dist_measure))
    auc_perm = np.zeros(n_permutation)

    for i in range(n_permutation):
        # randomly permute class indices based on predicted values
        idx_pred0_perm = idx_pred0[np.random.permutation(n_pred0)]
        idx_pred1_perm = idx_pred1[np.random.permutation(n_pred1)]

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
