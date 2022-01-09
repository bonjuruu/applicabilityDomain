import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc

from adad.utils import create_dir


def plot_roc_list(fprs, tprs, legend, path, title=None,
                  fontsize=16, figsize=(8, 8)):
    """Plot and save ROC curve for multiple models for comparision. 

    Parameters:
    -----------
    fprs: tuple of list
        A tuple of False Positive Rates
    tprs: tuple of list
        A tuple of True Positive Rates
    legend: list
    path: string
        Output path
    title: string, default=None
        Title for the figure
    fontsize: int
        Font size for graph
    figsize: tuple, default=(8, 8)
        (width, height) of the size of the graph
    """
    n_models = len(fprs)
    assert n_models == len(tprs), 'Input pairs should have same length.'
    assert n_models == len(legend), \
        "The plot's legend should have same length as the input pairs"

    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    fprs = list(fprs)
    tprs = list(tprs)
    legend = list(legend)
    # Adding plot for each model
    for i in range(n_models):
        fpr = fprs[i]
        tpr = tprs[i]
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            estimator_name=legend[i]
        ).plot(ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax.legend(loc="lower right")
    plt.tight_layout()
    # Create new directory when it doesn't exist
    create_dir(Path(path).absolute().parent)
    plt.savefig(path, dpi=300)


def plot_ca(df, path, dataname, n_cv=5, fontsize=16, figsize=(8, 8)):
    """Plot Cumulative Accuracy"""
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    mean_x = np.linspace(0, 1, 100)
    ys = []
    for i in range(1, n_cv + 1):
        x = df[f'cv{i}_rate']
        y = df[f'cv{i}_acc']
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
    plt.savefig(path, dpi=300)


def plot_roc(df, roc_aucs, path, dataname, n_cv=5, fontsize=16, figsize=(8, 8)):
    """Plot ROC Curve"""
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # Draw each ROC curve
    for i in range(1, n_cv + 1):
        fpr = df[f'cv{i}_fpr']
        tpr = df[f'cv{i}_tpr']
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
    plt.savefig(path, dpi=300)


def plot_pc(df, path, dataname, n_cv=5, fontsize=16, figsize=(8, 8)):
    """Plot Predictiveness Curves"""
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
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
    plt.savefig(path, dpi=300)
