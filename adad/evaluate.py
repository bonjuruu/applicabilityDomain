import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, roc_curve


def sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity and specificity"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def save_roc(y_true, y_score, path, title=None, fontsize=14, figsize=(8, 8)):
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
