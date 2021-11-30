import matplotlib.pyplot as plt
import numpy as np
from numpy.random import permutation, seed
from scipy.stats import rankdata
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, roc_curve

from adad.utils import hamming_accuracy


def sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity and specificity"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

#also needs fixing
def get_fpr_tpr(y_true, y_proba, y_pred):
    tpr, fpr = [0.0], [0.0]
    
    proba_c1 = [x[1] for x in y_proba]
    idx_array = np.argsort(proba_c1)
    
    y_pred_sorted = np.flip(y_pred[idx_array]).tolist()
    y_true_sorted = np.flip(y_true[idx_array]).tolist()
    
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(len(y_true)):
        tp += 1.0 if (y_true_sorted[i]==1) & (y_pred_sorted[i]==1) else 0.0
        tn += 1.0 if (y_true_sorted[i]==0) & (y_pred_sorted[i]==0) else 0.0
        fp += 1.0 if (y_true_sorted[i]==0) & (y_pred_sorted[i]==1) else 0.0
        fn += 1.0 if (y_true_sorted[i]==1) & (y_pred_sorted[i]==0) else 0.0
        
        if (fp != 0 or tn != 0):
            fpr.append(fp/(fp+tn))
        else:
            fpr.append(0.0)
        
        if (tp != 0 or fn != 0):
            tpr.append(tp / (tp + fn))
        else:
            tpr.append(0.0)

    fpr.sort()
    tpr.sort()
    return fpr, tpr

#needs fixing
def create_roc_graph(fpr, tpr, path, title=None, fontsize=14, figsize=(8, 8)):
    plt.style.use('seaborn-whitegrid')
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)

    plt.rcParams["font.size"] = fontsize
    _, ax = plt.subplots(figsize=figsize)
    ax.tick_params(labelsize=fontsize)
    plt.plot(fpr, tpr, color='blue', linewidth=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='dashed', color='red', linewidth=2, label='random')
    if title:
        ax.set_title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)

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
      
#for multilabel
def acc_vs_removed(y_true, y_pred, scores, plot=True, add=True, label=""):
    s = np.argsort(scores)
    #acc, rem = [0], [0]
    acc_full = hamming_accuracy(y_true, y_pred)
    acc, rem = [], []
    for i in range(0, len(s)+1):
        acc.append((hamming_accuracy(y_true[s[i:]], y_pred[s[i:]]) - acc_full) / (1-acc_full))
        rem.append(i/len(s))
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
    return np.sum(np.array(acc)[1:] * (np.array(rem)[1:]-np.array(rem)[:-1]))

def calculate_auc(y_true, y_pred, DMmeasure, nPerm, SEED=0):
    """Calculates AUC via permutation test

    Parameters:
        y_true (list): A list of all the true values for y
        y_pred (list): A list of all the predicted values for y
        DMmeasure (n*2 matrix): A matrix of the predicted probability of the y value
        nPerm (int): Number of permutations
        SEED (int, optional): Seed to be able to obtain same value. Defaults to 0.

    Returns:
        sig_value[int], permAUC[list]: Returns the 95th percentile value and a list of all AUC values calculated from the permutation test
    """
    seed(SEED)
    #set up array for AUC per permutation
    permAUC = [0 for i in range(nPerm)]
    #find indices of each class 
    idxC1 = np.where(y_pred == 0)[0]
    idxC2 = np.where(y_pred == 1)[0]
    #determine size of each class
    n1Pred = len(idxC1)
    n2Pred = len(idxC2)
    #determine actual size of each class
    n1True = len(np.where(y_true == 0)[0])
    n2True = len(np.where(y_true == 1)[0])
    #find which indices are class 1
    idxY_true_C1 = np.where(y_true == 0)[0]
    #create array for permuted DM measure
    permDMmeas = np.zeros(len(DMmeasure))
    
    for i in range(nPerm):
        #randomly permute class indices based on predicted values
        permIdxC1 = np.array(idxC1[permutation(n1Pred)])
        permIdxC2 = np.array(idxC2[permutation(n2Pred)])
        #assign permuted DM measures
        permDMmeas[idxC1] = [x[0] for x in DMmeasure[permIdxC1]]
        permDMmeas[idxC2] = [x[0] for x in DMmeasure[permIdxC2]]
        
        #find rank of DM after permutations
        ranks = rankdata(permDMmeas)
        #sum of ranks for class 1
        S1 = np.sum(ranks[idxY_true_C1])
        #calculate auc from sum of ranks and class population
        auc = (S1 - (n1True * (n1True + 1) / 2)) / (n1True * n2True)
        permAUC[i] = auc
    
    sig_value = np.percentile(permAUC, 95)
    
    return sig_value, permAUC     